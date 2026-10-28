"""Graph Grid for ComfyUI.

The grid is edited in a dialog (web/graph_grid.js) and stored as JSON in the
node's hidden "config" widget:

    {"x": {"fields": [{"node": "5", "input": "text", "alias": "prompt", "class": "CLIPTextEncode"}],
           "entries": [{"label": "", "values": {"prompt": "1girl, park"}}],
           "label": "{auto}"},
     "y": {...}}

Each entry is one column (x) or row (y). A blank value leaves the widget as it
is, '""' sets an empty string, and {v} in a text value is the widget's value
before the override. X overrides are applied before Y overrides. Field node ids
are relative to the graph the Graph Grid node sits in.

At run time the node copies every node between the overridden widgets and the
image input once per cell (nodes no override touches are shared) and draws a
labelled grid.

Label templates: {auto}, {index}, {alias}, and {alias:SPEC} where SPEC is one of
name (file name without folder or extension), g, % (rounded percent), .Nf
(N decimals) or .N (first N characters). web/graph_grid.js implements the same
grammar for its preview; keep the two in step.
"""
import copy
import functools
import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import nodes as comfy_nodes

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

WEB_DIRECTORY = "./web"

FILE_EXT = re.compile(r"\.(safetensors|ckpt|pt|pth|bin|gguf|sft)$", re.I)
TOKEN = re.compile(r"\{([^{}:]+)(?::([^{}]*))?\}")
NUMBER = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
INTEGER = re.compile(r"^[-+]?\d+$")
TRUE_WORDS = {"1", "true", "yes", "on"}
FALSE_WORDS = {"0", "false", "no", "off"}
BATCH_MODES = ["side by side", "first image", "animate"]
MAX_DECIMALS = 20
MAX_CUT = 500
MAX_GRID_PIXELS = 100_000_000
MAX_OUTPUT_BYTES = 4 * 1024 ** 3

EMPTY_CONFIG = json.dumps({"x": {"fields": [], "entries": [], "label": "{auto}"},
                           "y": {"fields": [], "entries": [], "label": "{auto}"}})


def _is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int)


def _basename(s):
    return FILE_EXT.sub("", str(s).replace("/", "\\").rsplit("\\", 1)[-1])


def _slug(label):
    return re.sub(r"[^\w.@+-]+", "_", label).strip("_.")[:60]


def _as_number(v):
    s = str(v).strip()
    if not NUMBER.match(s):
        return None
    n = float(s)
    return n if math.isfinite(n) else None


def _number_text(value):
    """A number the way JavaScript's String(n) writes it, so labels match the editor preview."""
    s = str(value).strip()
    if INTEGER.match(s):
        return str(int(s))
    n = float(s)
    if n.is_integer() and abs(n) < 1e21:
        return str(int(n))
    if 1e-6 <= abs(n) < 1e21:
        return format(Decimal(repr(n)), "f")
    return re.sub(r"e([+-])0*(\d)", r"e\1\2", repr(n))


def _auto(value):
    s = str(value)
    if s == '""':
        return "(empty)"
    if FILE_EXT.search(s):
        return _basename(s)
    n = _as_number(s)
    if n is not None:
        return _number_text(s)
    s = re.sub(r"^[,\s]+|[,\s]+$", "", s.replace("{v}", ""))
    return s if len(s) <= 30 else s[:27] + "..."


def _format(value, spec):
    spec = (spec or "").strip()
    if not spec:
        return str(value)
    if spec == "name":
        return _basename(value)
    n = _as_number(value)
    cut = re.fullmatch(r"\.(\d+)", spec)
    if cut and n is None:
        return str(value)[:min(int(cut.group(1)), MAX_CUT)]
    if n is not None:
        fixed = re.fullmatch(r"\.(\d+)f", spec)
        if fixed and abs(n) >= 1e21:
            return _number_text(value)
        if fixed:
            # Same rounding as JS Number.toFixed: the exact binary value, ties away from zero.
            places = Decimal(1).scaleb(-min(int(fixed.group(1)), MAX_DECIMALS))
            return format(Decimal(n).quantize(places, ROUND_HALF_UP), "f")
        if spec == "g":
            return _number_text(value)
        if spec == "%":
            return f"{math.floor(n * 100 + 0.5)}%"
    return str(value)


def render_label(template, values, fields, index):
    def sub(m):
        key, spec = m.group(1).strip(), m.group(2)
        if key == "auto":
            return " ".join(_auto(values[f["alias"]]) for f in fields if f["alias"] in values)
        if key == "index":
            return str(index)
        if key in values:
            return _format(values[key], spec)
        return ""
    return re.sub(r"\s+", " ", TOKEN.sub(sub, template or "{auto}")).strip()


def _options(spec):
    if not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None


def _coerce(value, spec, current):
    """Turn an editor value into what the widget expects; raises ValueError with a reason."""
    kind = spec[0] if spec else None
    extra = spec[1] if spec and len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(value, str) and isinstance(current, str):
        value = value.replace("{v}", current)
    text = str(value).strip()

    options = _options(spec)
    if options is not None:
        for option in options:
            if option == value or str(option) == text:
                return option
        raise ValueError("is not one of the options")

    if kind == "BOOLEAN" or (kind is None and isinstance(current, bool)):
        if isinstance(value, bool):
            return value
        if text.lower() in TRUE_WORDS:
            return True
        if text.lower() in FALSE_WORDS:
            return False
        raise ValueError("is not true or false")

    if kind == "INT":
        if INTEGER.match(text):
            number = int(text)
        elif _as_number(text) is not None and float(text).is_integer():
            number = int(float(text))
        else:
            raise ValueError("is not a whole number")
    elif kind == "FLOAT" or (kind is None and isinstance(current, (int, float))):
        # Values inside object widgets have no declared type; rgthree writes a strength of 1.0 as 1.
        if INTEGER.match(text) and kind is None and isinstance(current, int):
            number = int(text)
        elif _as_number(text) is not None:
            number = float(text)
        else:
            raise ValueError("is not a number")
    else:
        return value if isinstance(value, str) else str(value)

    low, high = extra.get("min"), extra.get("max")
    if low is not None and number < low:
        raise ValueError(f"is below the minimum {low}")
    if high is not None and number > high:
        raise ValueError(f"is above the maximum {high}")
    return number


def _split_path(name):
    """'lora_1.strength' -> ('lora_1', ['strength']): a value inside a widget that holds an object or list."""
    base, _, rest = name.partition(".")
    return base, rest.split(".") if rest else []


def _unwrap(value, part):
    # The frontend sends list widget values as {"__value__": [...]} so they can't be mistaken for links.
    if isinstance(value, dict) and "__value__" in value and part not in value:
        return value["__value__"]
    return value


def _walk(container, path):
    parent = container
    for part in path[:-1]:
        parent = _unwrap(parent, part)
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    parent = _unwrap(parent, path[-1])
    key = int(path[-1]) if isinstance(parent, list) else path[-1]
    parent[key]
    return parent, key


def _check_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("Graph Grid: config must be a JSON object with 'x' and 'y'")
    for key in ("x", "y"):
        sec = cfg.get(key) or {}
        if not isinstance(sec, dict):
            raise ValueError(f"Graph Grid: config.{key} must be an object")
        fields, entries = sec.get("fields") or [], sec.get("entries") or []
        if not isinstance(fields, list) or not isinstance(entries, list):
            raise ValueError(f"Graph Grid: config.{key}.fields and .entries must be lists")
        aliases = set()
        for f in fields:
            if not isinstance(f, dict) or not all(isinstance(f.get(k), (str, int)) for k in ("node", "input", "alias")):
                raise ValueError(f"Graph Grid: every field in config.{key} needs node, input and alias")
            if f["alias"] in aliases:
                raise ValueError(f"Graph Grid: alias '{f['alias']}' is used twice in config.{key}")
            aliases.add(f["alias"])
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("values", {}) or {}, dict):
                raise ValueError(f"Graph Grid: entries in config.{key} must be objects with a 'values' object")
            if not isinstance(e.get("label") or "", str):
                raise ValueError(f"Graph Grid: entry labels in config.{key} must be text")
        if not isinstance(sec.get("label") or "", str):
            raise ValueError(f"Graph Grid: config.{key}.label must be text")
        if entries and not fields:
            raise ValueError(f"Graph Grid: config.{key} has entries but no fields, so every cell would be the same")


def _progress(owner, done, total, stage, columns=0):
    # Shown on the Graph Grid node by web/graph_grid.js; cached cells send nothing, so counts can jump.
    server = getattr(PromptServer, "instance", None) if PromptServer else None
    if server is not None and owner:
        server.send_sync("graphgrid.progress", {"node": owner, "done": done, "total": total, "stage": stage,
                                                "columns": columns}, server.client_id)


def _class_has_fingerprint(class_type):
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
    return cls is not None and (hasattr(cls, "IS_CHANGED") or hasattr(cls, "fingerprint_inputs"))


class GraphGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"lazy": True, "tooltip": "The image each cell shows. Everything upstream can be varied."}),
                "config": ("STRING", {"multiline": True, "default": EMPTY_CONFIG}),
                "cell_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "font_size": ("INT", {"default": 28, "min": 8, "max": 96}),
                "save_cells": ("BOOLEAN", {"default": False, "tooltip": "Also save every cell at full size."}),
                "cells_prefix": ("STRING", {"default": "graph_grid/cells",
                                            "tooltip": "Folder/prefix for saved cells; files get row, column and labels."}),
                "batch": (BATCH_MODES, {"default": "side by side",
                                        "tooltip": "Cells with several images: lay them side by side, keep the first, "
                                                   "or animate (one grid per frame, for a video or animated save node)."}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid",)
    FUNCTION = "plot"
    CATEGORY = "Graph Grid"

    @classmethod
    def IS_CHANGED(cls, config="", **kwargs):
        # A cached plot is reused before plot() runs, so if an override targets a node whose output
        # depends on outside state (a file it loads, say), only re-planning lets the clones notice.
        try:
            cfg = json.loads(config or EMPTY_CONFIG)
            classes = {f.get("class") for key in ("x", "y") for f in (cfg.get(key) or {}).get("fields", [])}
        except (ValueError, AttributeError, TypeError):
            return config
        return float("nan") if any(c and _class_has_fingerprint(c) for c in classes) else config

    def check_lazy_status(self, **kwargs):
        # Never ask for the image: the original chain must not run, each cell runs its own copy of it.
        return []

    def plot(self, config, cell_scale, font_size, dynprompt, unique_id, image=None,
             save_cells=False, cells_prefix="graph_grid/cells", batch="side by side"):
        try:
            cfg = json.loads(config or EMPTY_CONFIG)
        except ValueError as e:
            raise ValueError(f"Graph Grid: config is not valid JSON ({e})") from None
        _check_config(cfg)
        src = dynprompt.get_node(unique_id)["inputs"]["image"]
        if not _is_link(src):
            raise ValueError("Graph Grid: connect an image")
        src = [str(src[0]), src[1]]
        # Inside a subgraph the queued ids carry the subgraph's path; field ids are stored relative to it.
        scope = unique_id.rpartition(":")[0]

        upstream = {}
        stack = [src[0]]
        while stack:
            nid = stack.pop()
            if nid in upstream:
                continue
            upstream[nid] = dynprompt.get_node(nid)
            stack += [str(v[0]) for v in upstream[nid]["inputs"].values() if _is_link(v)]
        if any(n["class_type"] == "GraphGrid" for n in upstream.values()):
            raise ValueError("Graph Grid: another Graph Grid node feeds this one's image; nesting them is not supported")

        def title(nid):
            return upstream[nid].get("_meta", {}).get("title", upstream[nid]["class_type"])

        @functools.lru_cache(maxsize=None)
        def spec_of(nid, name):
            types = comfy_nodes.NODE_CLASS_MAPPINGS[upstream[nid]["class_type"]].INPUT_TYPES()
            for group in ("required", "optional"):
                if name in types.get(group, {}):
                    return types[group][name]
            return None

        def resolve(node_id):
            nid = str(node_id)
            for candidate in ((f"{scope}:{nid}", nid) if scope else (nid,)):
                if candidate in upstream:
                    return candidate
            return None

        def cells_for(key):
            section = cfg.get(key) or {}
            fields = section.get("fields") or []
            entries = section.get("entries") or []
            if not entries:
                return [("", [])]
            targets = {}
            for f in fields:
                nid = resolve(f["node"])
                if nid is None:
                    raise ValueError(f"Graph Grid: field '{f['alias']}' targets node #{f['node']}, which is not upstream "
                                     "of the image in the queued prompt (deleted, bypassed, muted or disconnected?)")
                name = str(f["input"])
                base, path = _split_path(name)
                current = upstream[nid]["inputs"].get(base)
                if spec_of(nid, base) is None and base not in upstream[nid]["inputs"]:
                    raise ValueError(f"Graph Grid: field '{f['alias']}': {title(nid)} has no input '{base}'")
                if _is_link(current):
                    raise ValueError(f"Graph Grid: field '{f['alias']}': {title(nid)}.{base} is wired to another node; "
                                     "vary that node instead")
                if path:
                    try:
                        _walk(current, path)
                    except (KeyError, IndexError, TypeError, ValueError):
                        raise ValueError(f"Graph Grid: field '{f['alias']}': {title(nid)}.{base} has no "
                                         f"'{'.'.join(path)}' in the queued prompt") from None
                targets[f["alias"]] = (nid, name)
            cells = []
            for i, entry in enumerate(entries):
                values = {k: v for k, v in (entry.get("values") or {}).items() if v not in (None, "") and k in targets}
                assigns = [(*targets[alias], "" if value == '""' else value, alias) for alias, value in
                           ((f["alias"], values[f["alias"]]) for f in fields if f["alias"] in values)]
                label = render_label(entry.get("label") or section.get("label"), values, fields, i + 1)
                cells.append((label, assigns))
            return cells

        xs, ys = cells_for("x"), cells_for("y")
        touched = {nid for _, a in xs + ys for nid, _, _, _ in a}
        if not touched:
            raise ValueError("Graph Grid: no row or column sets a value; open Edit grid and fill in some cells")

        cloned = set()
        changed = True
        while changed:
            changed = False
            for nid, n in upstream.items():
                if nid not in cloned and (nid in touched or any(
                        _is_link(v) and str(v[0]) in cloned for v in n["inputs"].values())):
                    cloned.add(nid)
                    changed = True
        if src[0] not in cloned:
            raise ValueError("Graph Grid: none of the fields affect the image input")

        graph, prev = {}, None
        total = len(xs) * len(ys)
        _progress(unique_id, 0, total, "cells")
        for r, (ylab, yas) in enumerate(ys):
            for c, (xlab, xas) in enumerate(xs):
                prefix = f"{unique_id}.gg.{r}.{c}."
                for nid in cloned:
                    n = copy.deepcopy(upstream[nid])
                    # ComfyUI stores the IS_CHANGED result on the prompt node and trusts it later;
                    # the clone's inputs differ, so it must be fingerprinted afresh.
                    n.pop("is_changed", None)
                    n["inputs"] = {k: [prefix + str(v[0]), v[1]] if _is_link(v) and str(v[0]) in cloned else v
                                   for k, v in n["inputs"].items()}
                    n["override_display_id"] = nid
                    graph[prefix + nid] = n
                for nid, name, value, alias in xas + yas:
                    inputs = graph[prefix + nid]["inputs"]
                    base, path = _split_path(name)
                    try:
                        if path:
                            parent, key = _walk(inputs[base], path)
                            parent[key] = _coerce(value, None, parent[key])
                        else:
                            inputs[base] = _coerce(value, spec_of(nid, base), inputs.get(base))
                    except ValueError as e:
                        raise ValueError(f"Graph Grid: cell (row '{ylab or r + 1}', column '{xlab or c + 1}'), "
                                         f"field '{alias}' ({title(nid)}.{name}): '{value}' {e}") from None
                img = [prefix + src[0], src[1]]
                if save_cells:
                    name = "_".join(p for p in (f"r{r + 1:02d}", f"c{c + 1:02d}", _slug(ylab), _slug(xlab)) if p)
                    graph[f"{unique_id}.gg.save.{r}.{c}"] = {"class_type": "SaveImage", "inputs": {
                        "images": img, "filename_prefix": "/".join(p for p in (cells_prefix.strip("/ "), name) if p)}}
                collect = f"{unique_id}.gg.collect.{r}.{c}"
                graph[collect] = {"class_type": "GraphGridCollect", "inputs": {
                    "image": img, "owner": unique_id, "index": r * len(xs) + c + 1, "total": total, "columns": len(xs),
                    **({"previous": [prev, 0]} if prev else {})}}
                prev = collect

        layout = json.dumps({"columns": [c[0] for c in xs], "rows": [r[0] for r in ys]})
        draw = f"{unique_id}.gg.draw"
        graph[draw] = {"class_type": "GraphGridDraw", "inputs": {
            "images": [prev, 0], "layout": layout, "cell_scale": cell_scale, "font_size": font_size,
            "batch": batch, "owner": unique_id}}
        return {"result": ([draw, 0],), "expand": graph}


class GraphGridCollect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}, "optional": {
            "previous": ("GRAPH_GRID_IMAGES",),
            "owner": ("STRING", {"default": ""}),
            "index": ("INT", {"default": 0, "min": 0, "max": 1 << 30}),
            "total": ("INT", {"default": 0, "min": 0, "max": 1 << 30}),
            "columns": ("INT", {"default": 0, "min": 0, "max": 1 << 30}),
        }}

    DEV_ONLY = True
    RETURN_TYPES = ("GRAPH_GRID_IMAGES",)
    FUNCTION = "collect"
    CATEGORY = "Graph Grid/internal"

    def collect(self, image, previous=None, owner="", index=0, total=0, columns=0):
        _progress(owner, index, total, "cells", columns)
        return ((previous or []) + [image],)


class GraphGridDraw:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("GRAPH_GRID_IMAGES",),
            "layout": ("STRING", {"default": "{}"}),
            "cell_scale": ("FLOAT", {"default": 0.5}),
            "font_size": ("INT", {"default": 28}),
        }, "optional": {
            "batch": (BATCH_MODES, {"default": "side by side"}),
            "owner": ("STRING", {"default": ""}),
        }}

    DEV_ONLY = True
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "Graph Grid/internal"

    @staticmethod
    def _font(size):
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    @staticmethod
    def _to_pil(frame):
        return Image.fromarray((frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))

    @staticmethod
    def _fit_text(face, text, width):
        if face.getlength(text) <= width:
            return text
        while text and face.getlength(text + "…") > width:
            text = text[:-1]
        return text + "…"

    def _wrap(self, face, text, width, lines=4):
        out, line = [], ""
        for word in text.split(" "):
            candidate = f"{line} {word}".strip()
            if line and face.getlength(candidate) > width:
                out.append(line)
                line = word
            else:
                line = candidate
        out.append(line)
        if len(out) > lines:
            out = out[:lines]
            out[-1] = self._fit_text(face, out[-1] + " …", width)
        return "\n".join(self._fit_text(face, ln, width) for ln in out)

    def draw(self, images, layout, cell_scale, font_size, batch="side by side", owner=""):
        _progress(owner, len(images), len(images), "drawing")
        spec = json.loads(layout)
        cols, rows = spec["columns"], spec["rows"]
        if not images or any(img.shape[0] == 0 for img in images):
            raise ValueError("Graph Grid: a cell produced no image")
        if len(images) != len(cols) * len(rows):
            raise ValueError(f"Graph Grid: got {len(images)} cells for a {len(cols)}x{len(rows)} grid")

        if batch == "animate":
            sources = [list(img) for img in images]
        elif batch == "first image":
            sources = [[img[0]] for img in images]
        else:
            sources = [[torch.cat(list(img), dim=1)] for img in images]
        frames = max(len(s) for s in sources)

        # Cells can differ in size; each is scaled by cell_scale and centred in a box that fits the largest.
        cw = max(1, int(max(f.shape[1] for s in sources for f in s) * cell_scale))
        ch = max(1, int(max(f.shape[0] for s in sources for f in s) * cell_scale))
        face = self._font(font_size)
        line_h = int(font_size * 1.25)
        col_text = [self._wrap(face, c, cw - 20) for c in cols]
        top = (max(t.count("\n") for t in col_text) + 1) * line_h + 20 if any(cols) else 0
        row_text = [self._fit_text(face, r, max(cw, 400)) for r in rows]
        left = int(max(face.getlength(r) for r in row_text)) + 30 if any(rows) else 0
        width, height = left + cw * len(cols), top + ch * len(rows)
        if width * height > MAX_GRID_PIXELS or width * height * 3 * 4 * frames > MAX_OUTPUT_BYTES:
            raise ValueError(f"Graph Grid: the grid would be {width}x{height} px x {frames} frames, too large; "
                             "lower cell_scale, use fewer cells, or pick batch 'first image'")

        base = Image.new("RGB", (width, height), "white")
        d = ImageDraw.Draw(base)
        for c, text in enumerate(col_text):
            d.multiline_text((left + c * cw + 10, 10), text, fill="black", font=face)
        for r, label in enumerate(row_text):
            d.text((10, top + r * ch + ch // 2 - font_size // 2), label, fill="black", font=face)

        fitted = {}

        def tile(i, f):
            # Shorter cells hold their last frame.
            k = min(f, len(sources[i]) - 1)
            if (i, k) not in fitted:
                im = self._to_pil(sources[i][k])
                s = min(cw / im.width, ch / im.height, cell_scale)
                fitted[(i, k)] = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
            return fitted[(i, k)]

        out = np.empty((frames, height, width, 3), dtype=np.uint8)
        for f in range(frames):
            grid = base.copy()
            for r in range(len(rows)):
                for c in range(len(cols)):
                    t = tile(r * len(cols) + c, f)
                    grid.paste(t, (left + c * cw + (cw - t.width) // 2, top + r * ch + (ch - t.height) // 2))
            out[f] = np.asarray(grid)
        _progress(owner, len(images), len(images), "done")
        return (torch.from_numpy(out).float().div_(255.0),)


NODE_CLASS_MAPPINGS = {"GraphGrid": GraphGrid, "GraphGridCollect": GraphGridCollect, "GraphGridDraw": GraphGridDraw}
NODE_DISPLAY_NAME_MAPPINGS = {"GraphGrid": "Graph Grid", "GraphGridCollect": "Graph Grid Collect", "GraphGridDraw": "Graph Grid Draw"}
