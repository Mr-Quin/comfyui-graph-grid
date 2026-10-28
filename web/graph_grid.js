import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EMPTY = () => ({ fields: [], entries: [], label: "{auto}" });
const FILE_EXT = /\.(safetensors|ckpt|pt|pth|bin|gguf|sft)$/i;
const TOKEN = /\{([^{}:]+)(?::([^{}]*))?\}/g;
const NUMBER = /^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
const RESERVED_ALIASES = new Set(["auto", "index", "__proto__", "constructor", "prototype"]);
const MAX_DECIMALS = 20;
const MAX_CUT = 500;
const MAX_FILL = 1000;
const MAX_ENTRIES = 256;
const MANY_CELLS = 64;
const MAX_SUBFIELDS = 60;
const SUMMARY_HEIGHT = 54;

// ---------- label grammar, mirrored in __init__.py ----------

const has = (obj, key) => Object.prototype.hasOwnProperty.call(obj ?? {}, key);
const basename = (s) => String(s).replace(/\//g, "\\").split("\\").pop().replace(FILE_EXT, "");
const asNumber = (v) => {
    const s = String(v ?? "").trim();
    const n = NUMBER.test(s) ? Number(s) : NaN;
    return Number.isFinite(n) ? n : null;
};
const numberText = (v) => {
    const s = String(v).trim();
    if (/^[-+]?\d+$/.test(s)) return BigInt(s).toString();
    const n = Number(s);
    return Number.isInteger(n) && Math.abs(n) < 1e21 ? BigInt(n).toString() : String(n);
};
const cutChars = (s, n) => Array.from(String(s)).slice(0, n).join("");

function autoText(value) {
    const s = String(value);
    if (s === '""') return "(empty)";
    if (FILE_EXT.test(s)) return basename(s);
    const n = asNumber(s);
    if (n !== null) return numberText(s);
    const t = s.replaceAll("{v}", "").replace(/^[,\s]+|[,\s]+$/g, "");
    return Array.from(t).length <= 30 ? t : cutChars(t, 27) + "...";
}

function formatValue(value, spec) {
    spec = (spec ?? "").trim();
    if (!spec) return String(value);
    if (spec === "name") return basename(value);
    const n = asNumber(value);
    const cut = spec.match(/^\.(\d+)$/);
    if (cut && n === null) return cutChars(value, Math.min(Number(cut[1]), MAX_CUT));
    if (n !== null) {
        const fixed = spec.match(/^\.(\d+)f$/);
        if (fixed) return Math.abs(n) >= 1e21 ? numberText(value) : n.toFixed(Math.min(Number(fixed[1]), MAX_DECIMALS));
        if (spec === "g") return numberText(value);
        if (spec === "%") return `${Math.floor(n * 100 + 0.5)}%`;
    }
    return String(value);
}

function setValues(values) {
    return Object.fromEntries(Object.entries(values || {}).filter(([, v]) => v !== "" && v != null));
}

function renderLabel(template, values, fields, index) {
    const set = setValues(values);
    return (template || "{auto}").replace(TOKEN, (_, key, spec) => {
        key = key.trim();
        if (key === "auto") return fields.filter((f) => has(set, f.alias)).map((f) => autoText(set[f.alias])).join(" ");
        if (key === "index") return String(index);
        return has(set, key) ? formatValue(set[key], spec) : "";
    }).replace(/\s+/g, " ").trim();
}

const entryLabel = (sec, e, i) => renderLabel(e.label || sec.label, e.values, sec.fields, i + 1);

// ---------- graph helpers ----------

// Node ids in the queued prompt are paths: a node inside a subgraph is "<subgraph node id>:<inner id>".
// Field ids are stored relative to the graph the Graph Grid sits in; the backend adds that graph's own path.
// Bypassed (mode 4) and muted (mode 2) nodes are left out of the prompt; Reroutes, Notes and the
// legacy Primitive node only exist in the editor, so none of these can be varied.
const MODE_MUTED = 2;
const MODE_BYPASS = 4;

function getLink(graph, id) {
    const links = graph?.links;
    return links instanceof Map ? links.get(id) : links?.[id];
}

function nodeByPath(root, path) {
    let graph = root;
    let node = null;
    for (const part of String(path).split(":")) {
        node = graph?.getNodeById(Number(part)) ?? graph?.getNodeById(part);
        if (!node) return null;
        graph = node.subgraph;
    }
    return node;
}

const isSubgraph = (n) => Boolean(n.subgraph) || n.isSubgraphNode?.() === true;
const usable = (n) => !n.isVirtualNode && n.mode !== MODE_MUTED && n.mode !== MODE_BYPASS;

function innerNodes(n, prefix, out) {
    for (const inner of n.subgraph?.nodes ?? n.subgraph?._nodes ?? []) {
        if (!usable(inner)) continue;
        const path = `${prefix}:${inner.id}`;
        if (isSubgraph(inner)) innerNodes(inner, path, out);
        else out.set(path, { path, node: inner });
    }
}

function upstreamNodes(plot) {
    const out = new Map();
    const seen = new Set();
    const graph = plot.graph ?? app.graph;
    const queue = [];
    const image = plot.inputs?.find((i) => i.name === "image");
    if (image?.link != null) queue.push(getLink(graph, image.link)?.origin_id);
    while (queue.length) {
        const id = queue.pop();
        if (id == null || seen.has(id)) continue;
        seen.add(id);
        const n = graph.getNodeById(id);
        if (!n) continue;
        // Every node inside an upstream subgraph is listed, whether or not it feeds the image.
        if (usable(n)) {
            if (isSubgraph(n)) innerNodes(n, String(n.id), out);
            else out.set(String(n.id), { path: String(n.id), node: n });
        }
        for (const i of n.inputs ?? []) if (i.link != null) queue.push(getLink(graph, i.link)?.origin_id);
    }
    return [...out.values()].sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }));
}

let loraFiles = null;

async function loadLoraFiles() {
    if (loraFiles) return loraFiles;
    try {
        const info = await (await api.fetchApi("/object_info/LoraLoader")).json();
        loraFiles = info?.LoraLoader?.input?.required?.lora_name?.[0] ?? [];
    } catch {
        loraFiles = [];
    }
    return loraFiles;
}

// A widget whose value is an object or list is offered as one field per primitive inside it,
// named "<widget>.<key>[.<key>]"; the backend resolves the same path in the queued prompt.
function subFields(name, value, out, depth = 0) {
    if (out.length >= MAX_SUBFIELDS || depth > 3) return;
    if (value === null || typeof value !== "object") {
        const key = name.split(".").pop();
        const kind = typeof value === "boolean" ? "toggle" : typeof value === "number" ? "number" : "text";
        const isLora = key === "lora" && typeof value === "string";
        out.push({ name, type: isLora ? "combo" : kind, value,
            options: isLora ? { values: () => loraFiles ?? [] } : {} });
        return;
    }
    const entries = Array.isArray(value) ? value.map((v, i) => [String(i), v]) : Object.entries(value);
    for (const [k, v] of entries) subFields(`${name}.${k}`, v, out, depth + 1);
}

function fieldWidgets(n) {
    // A widget driven by a legacy Primitive node gets the Primitive's value inlined into the prompt,
    // so it can still be overridden; any other link means the value comes from a real node.
    const linked = new Set((n.inputs ?? []).filter((i) => {
        if (i.link == null || !i.widget) return false;
        const origin = n.graph?.getNodeById(getLink(n.graph, i.link)?.origin_id);
        return !(origin?.isVirtualNode && origin.type === "PrimitiveNode");
    }).map((i) => i.widget.name));
    const out = [];
    for (const w of n.widgets ?? []) {
        if (!w.name || w.value === undefined || linked.has(w.name) || w.type === "button"
            || w.name === "control_after_generate" || w.serialize === false) continue;
        if (w.value !== null && typeof w.value === "object") {
            const parts = [];
            subFields(w.name, w.value, parts);
            out.push(...parts);
        } else {
            out.push(w);
        }
    }
    return out;
}

function widgetInfo(graph, field) {
    const n = nodeByPath(graph, field.node);
    const w = n && fieldWidgets(n).find((x) => x.name === field.input);
    if (!w) return { kind: "text", missing: true, title: `#${field.node}`, options: [] };
    let options = w.options?.values;
    if (typeof options === "function") options = options();
    const kind = w.type === "combo" ? "combo" : w.type === "number" || w.type === "slider" ? "number"
        : w.type === "toggle" || typeof w.value === "boolean" ? "boolean" : "text";
    return { kind, options: Array.isArray(options) ? options : [], current: w.value, title: n.title || n.type,
        integer: w.options?.precision === 0 || (w.options?.round === 1) };
}

function globRegex(pattern) {
    const esc = pattern.replace(/\//g, "\\").replace(/[.+^${}()|\\]/g, "\\$&")
        .replace(/\*/g, ".*").replace(/\?/g, ".");
    return new RegExp("^" + esc + "$", "i");
}

function moveItem(list, from, to) {
    if (from === to || from < 0 || to < 0 || from >= list.length || to >= list.length) return false;
    const [item] = list.splice(from, 1);
    list.splice(to, 0, item);
    return true;
}

// ---------- tiny DOM helper ----------

function h(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (v === undefined || v === null || v === false) continue;
        if (k === "class") e.className = v;
        else if (k === "style") e.style.cssText = v;
        else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
        else if (k === "value") e.value = v;
        else e.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid instanceof Node ? kid : String(kid));
    return e;
}

const CSS = `
.gg-dialog { width: min(1200px, 96vw); height: 88vh; max-height: 88vh; padding: 0; overflow: hidden;
  border: 1px solid var(--border-color, #444); border-radius: 8px;
  background: var(--comfy-menu-bg, #202020); color: var(--fg-color, #ddd); font: 13px sans-serif; }
.gg-dialog::backdrop { background: rgba(0,0,0,.55); }
.gg-wrap { display: flex; flex-direction: column; height: 100%; }
.gg-head, .gg-foot { flex: none; display: flex; align-items: center; gap: 8px; padding: 10px 14px; }
.gg-head { border-bottom: 1px solid var(--border-color, #444); }
.gg-foot { border-top: 1px solid var(--border-color, #444); }
.gg-head h3 { margin: 0; font-size: 15px; flex: 1; }
.gg-body { flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; padding: 12px 14px;
  display: flex; flex-direction: column; gap: 18px; scrollbar-gutter: stable; }
.gg-sec { border: 1px solid var(--border-color, #444); border-radius: 6px; padding: 10px; }
.gg-sec h4 { margin: 0 0 8px; font-size: 14px; display: flex; gap: 8px; align-items: center; }
.gg-sec h4 .gg-count { font-weight: normal; opacity: .7; }
.gg-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 8px; }
.gg-chip { display: inline-flex; gap: 4px; align-items: center; padding: 3px 6px; border-radius: 12px;
  background: var(--comfy-input-bg, #2a2a2a); border: 1px solid var(--border-color, #444); }
.gg-chip input { width: 90px; }
.gg-dialog input, .gg-dialog select, .gg-dialog textarea { background: var(--comfy-input-bg, #2a2a2a);
  color: var(--input-text, #ddd); border: 1px solid var(--border-color, #555); border-radius: 4px; padding: 3px 5px;
  font: inherit; box-sizing: border-box; }
.gg-dialog textarea { width: 100%; min-width: 200px; resize: vertical; }
.gg-dialog button { background: var(--comfy-input-bg, #333); color: inherit; border: 1px solid var(--border-color, #555);
  border-radius: 4px; padding: 3px 8px; cursor: pointer; font: inherit; }
.gg-dialog button:hover:not(:disabled) { filter: brightness(1.25); }
.gg-dialog button:disabled { opacity: .45; cursor: default; }
.gg-dialog button.gg-primary { background: #2a6fdb; border-color: #2a6fdb; color: #fff; }
.gg-dialog button.gg-icon { padding: 1px 6px; }
.gg-tablewrap { overflow-x: auto; }
.gg-table { border-collapse: collapse; width: 100%; table-layout: auto; }
.gg-table th, .gg-table td { border-bottom: 1px solid var(--border-color, #3a3a3a); padding: 4px; vertical-align: top; text-align: left; }
.gg-table th { font-weight: 600; white-space: nowrap; }
.gg-table td.gg-idx { opacity: .6; width: 34px; white-space: nowrap; }
.gg-table td.gg-acts { white-space: nowrap; width: 1%; }
.gg-handle { cursor: grab; user-select: none; opacity: .6; padding-right: 4px; }
.gg-handle:hover { opacity: 1; }
.gg-table tr.gg-dragging, .gg-chip.gg-dragging { opacity: .35; }
.gg-table tr.gg-over-before td { box-shadow: inset 0 2px 0 #2a6fdb; }
.gg-table tr.gg-over-after td { box-shadow: inset 0 -2px 0 #2a6fdb; }
.gg-chip.gg-over-before { box-shadow: -3px 0 0 #2a6fdb; }
.gg-chip.gg-over-after { box-shadow: 3px 0 0 #2a6fdb; }
.gg-cell-key { white-space: nowrap; opacity: .8; width: 1%; }
.gg-cell-val { white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, monospace; font-size: 12px; }
.gg-preview { display: block; font-size: 11px; opacity: .65; margin-top: 2px; min-height: 14px; }
.gg-help { font-size: 11px; opacity: .65; }
.gg-fill { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; padding: 8px; margin-top: 8px;
  border: 1px dashed var(--border-color, #555); border-radius: 6px; }
.gg-err { color: #ff8080; }
.gg-warn { color: #f5b041; opacity: 1; }
.gg-mini { font: 11px sans-serif; color: var(--fg-color, #ddd); padding: 2px 4px; height: 54px;
  overflow: hidden; box-sizing: border-box; }
.gg-mini-line { line-height: 16px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.gg-mini-dim { opacity: .7; }
.gg-bar { height: 6px; margin-top: 4px; border-radius: 3px; background: var(--comfy-input-bg, #333); overflow: hidden; }
.gg-bar > div { height: 100%; background: #2a6fdb; transition: width .2s; }
`;

function injectCss() {
    if (document.getElementById("gg-css")) return;
    document.head.append(h("style", { id: "gg-css" }, CSS));
}

// Only the handle makes the element draggable, so text selection in the inputs keeps working.
function makeSortable(el, handle, index, group, onMove) {
    handle.addEventListener("mousedown", () => {
        el.draggable = true;
        window.addEventListener("mouseup", () => { el.draggable = false; }, { once: true });
    });
    el.addEventListener("dragstart", (ev) => {
        ev.dataTransfer.effectAllowed = "move";
        ev.dataTransfer.setData("text/plain", JSON.stringify({ group, index }));
        el.classList.add("gg-dragging");
    });
    el.addEventListener("dragend", () => {
        el.draggable = false;
        el.classList.remove("gg-dragging");
    });
    const side = (ev) => {
        const r = el.getBoundingClientRect();
        return el.tagName === "TR" ? ev.clientY > r.top + r.height / 2 : ev.clientX > r.left + r.width / 2;
    };
    const clear = () => el.classList.remove("gg-over-before", "gg-over-after");
    el.addEventListener("dragover", (ev) => {
        ev.preventDefault();
        clear();
        el.classList.add(side(ev) ? "gg-over-after" : "gg-over-before");
    });
    el.addEventListener("dragleave", clear);
    el.addEventListener("drop", (ev) => {
        ev.preventDefault();
        clear();
        let data;
        try { data = JSON.parse(ev.dataTransfer.getData("text/plain")); } catch { return; }
        if (data.group !== group || !Number.isInteger(data.index)) return;
        let to = index + (side(ev) ? 1 : 0);
        if (data.index < to) to -= 1;
        onMove(data.index, to);
    });
}

// ---------- config ----------

function normalizeSection(s) {
    const sec = { ...EMPTY(), ...(s && typeof s === "object" ? s : {}) };
    sec.fields = Array.isArray(sec.fields) ? sec.fields.filter((f) => f && typeof f === "object") : [];
    sec.entries = (Array.isArray(sec.entries) ? sec.entries : []).filter((e) => e && typeof e === "object")
        .map((e) => ({ ...e, values: e.values && typeof e.values === "object" ? e.values : {} }));
    sec.label = typeof sec.label === "string" ? sec.label : "{auto}";
    return sec;
}

function parseConfig(text) {
    try {
        const c = JSON.parse(text || "{}");
        if (!c || typeof c !== "object") throw new Error("not an object");
        return { x: normalizeSection(c.x), y: normalizeSection(c.y) };
    } catch (e) {
        return { x: EMPTY(), y: EMPTY(), error: String(e.message || e) };
    }
}

const cellCount = (cfg) => Math.max(cfg.x.entries.length, 1) * Math.max(cfg.y.entries.length, 1);

function axisLine(name, sec) {
    if (!sec.entries.length) return h("div", { class: "gg-mini-line" }, h("b", {}, `${name} `), "none");
    const fields = sec.fields.map((f) => f.alias).join(", ") || "-";
    const labels = sec.entries.map((e, i) => entryLabel(sec, e, i) || `#${i + 1}`).join(" · ");
    return h("div", { class: "gg-mini-line", title: `${fields}\n${labels.replaceAll(" · ", "\n")}` },
        h("b", {}, `${name} `), `${sec.entries.length} × ${fields}: `, h("span", { class: "gg-mini-dim" }, labels));
}

function renderSummary(el, cfg) {
    if (cfg.error) {
        el.replaceChildren(h("div", { class: "gg-mini-line gg-err" }, `Grid config is unreadable: ${cfg.error}`));
        return;
    }
    if (!cfg.x.entries.length && !cfg.y.entries.length) {
        el.replaceChildren(h("div", { class: "gg-mini-line" }, "No grid yet. Click Edit grid."));
        return;
    }
    el.replaceChildren(axisLine("X", cfg.x), axisLine("Y", cfg.y),
        h("div", { class: "gg-mini-line gg-mini-dim" }, `${cellCount(cfg)} cells`));
}

// ---------- editor ----------

function openEditor(plot, configWidget, onSave) {
    injectCss();
    const graph = plot.graph ?? app.graph;
    const cfg = parseConfig(configWidget.value);
    const loadError = cfg.error;
    delete cfg.error;
    const dialog = h("dialog", { class: "gg-dialog" });
    const body = h("div", { class: "gg-body" });
    const total = h("span", { class: "gg-help" });
    const candidates = upstreamNodes(plot);

    const snapshot = JSON.stringify(cfg);
    const close = () => { dialog.close(); dialog.remove(); };
    const cancel = () => {
        if (JSON.stringify(cfg) !== snapshot && !confirm("Discard your changes to the grid?")) return;
        close();
    };
    const save = () => {
        if (loadError && !confirm("The stored grid config could not be read. Replace it with this grid?")) return;
        configWidget.value = JSON.stringify(cfg);
        configWidget.callback?.(configWidget.value);
        onSave(cfg);
        try {
            plot.graph?.setDirtyCanvas?.(true, true);
            app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.();
        } catch { /* change tracking is best effort */ }
        close();
    };

    function refreshTotal() {
        const n = cellCount(cfg);
        total.textContent = n > MANY_CELLS ? `${n} cells; each one is a full sampler run` : `${n} cells`;
        total.classList.toggle("gg-warn", n > MANY_CELLS);
    }

    const cellView = h("div", { class: "gg-sec" });
    const pick = { x: 0, y: 0 };

    // The backend's order: column overrides, then row overrides, {v} being the value before each one.
    function cellValues() {
        const out = new Map();
        for (const key of ["x", "y"]) {
            const sec = cfg[key];
            const entry = sec.entries[pick[key]];
            if (!entry) continue;
            for (const f of sec.fields) {
                let v = entry.values?.[f.alias];
                if (v === "" || v == null) continue;
                if (v === '""') v = "";
                const id = `${f.node}|${f.input}`;
                const before = out.has(id) ? out.get(id).value : widgetInfo(graph, f).current;
                if (typeof before === "string") v = String(v).replaceAll("{v}", before);
                out.set(id, { field: f, value: v });
            }
        }
        return [...out.values()];
    }

    function renderCell() {
        const select = (key) => {
            const sec = cfg[key];
            if (!sec.entries.length) return h("span", { class: "gg-help" }, "none");
            pick[key] = Math.min(pick[key], sec.entries.length - 1);
            const sel = h("select", { onchange: (ev) => { pick[key] = Number(ev.target.value); renderCell(); } },
                sec.entries.map((e, i) => h("option", { value: i }, `${i + 1}: ${entryLabel(sec, e, i) || "(no label)"}`)));
            sel.value = String(pick[key]);
            return sel;
        };
        const rows = cellValues().map(({ field, value }) => h("tr", {},
            h("td", { class: "gg-cell-key" }, `${widgetInfo(graph, field).title} › ${field.input}`),
            h("td", {}, h("div", { class: "gg-cell-val" }, String(value)))));
        cellView.replaceChildren(
            h("h4", {}, "Cell preview", h("span", { class: "gg-count" }, "what one cell will use")),
            h("div", { class: "gg-row" }, "Column", select("x"), "Row", select("y")),
            rows.length ? h("table", { class: "gg-table" }, rows) : h("div", { class: "gg-help" }, "This cell changes nothing."));
    }

    function render() {
        const scroll = body.scrollTop;
        const kids = [section("x", "Columns (X)"), section("y", "Rows (Y)"), cellView];
        if (loadError) {
            kids.unshift(h("div", { class: "gg-err" }, `The stored grid config could not be read (${loadError}); `
                + "saving will replace it."));
        }
        if (!candidates.length) {
            kids.unshift(h("div", { class: "gg-err" }, "Connect the image input first; fields come from the nodes that feed it."));
        }
        renderCell();
        body.replaceChildren(...kids);
        body.scrollTop = scroll;
        refreshTotal();
    }

    function uniqueAlias(fields, base) {
        let clean = String(base).replace(/[^\w]+/g, "_").replace(/^_+|_+$/g, "") || "field";
        if (RESERVED_ALIASES.has(clean)) clean = `${clean}_field`;
        let alias = clean;
        let i = 2;
        while (fields.some((f) => f.alias === alias)) alias = `${clean}_${i++}`;
        return alias;
    }

    const renameToken = (template, from, to) =>
        template ? template.replace(TOKEN, (m, key, spec) => (key.trim() === from ? `{${to}${spec != null ? ":" + spec : ""}}` : m))
            : template;

    function section(key, title) {
        const sec = cfg[key];
        const noun = key === "x" ? "column" : "row";
        const wrap = h("div", { class: "gg-sec" });
        const previews = [];

        const updatePreviews = () => {
            sec.entries.forEach((e, i) => {
                if (previews[i]) previews[i].textContent = "→ " + (entryLabel(sec, e, i) || "(no label)");
            });
            refreshTotal();
            renderCell();
        };

        // fields
        const nodeSel = h("select", {}, h("option", { value: "" }, "node…"),
            candidates.map((c) => h("option", { value: c.path }, `${c.node.title || c.node.type} #${c.path}`)));
        const inputSel = h("select", {}, h("option", { value: "" }, "field…"));
        nodeSel.onchange = () => {
            const n = nodeByPath(graph, nodeSel.value);
            inputSel.replaceChildren(h("option", { value: "" }, "field…"),
                ...(n ? fieldWidgets(n).map((w) => h("option", { value: w.name }, w.name)) : []));
        };
        const addField = h("button", { onclick: () => {
            if (!nodeSel.value || !inputSel.value) return;
            const target = nodeByPath(graph, nodeSel.value);
            sec.fields.push({ node: String(nodeSel.value), input: inputSel.value,
                class: target?.comfyClass ?? target?.type,
                alias: uniqueAlias(sec.fields, inputSel.value.replace(/_model$|_name$/, "")) });
            render();
        } }, "+ add field");

        const chips = sec.fields.map((f, fi) => {
            const info = widgetInfo(graph, f);
            const alias = h("input", { value: f.alias, title: `Name used in label templates, like {${f.alias}}`,
                onchange: (ev) => {
                    const next = uniqueAlias(sec.fields.filter((x) => x !== f), ev.target.value || f.input);
                    if (next === f.alias) { render(); return; }
                    for (const e of sec.entries) {
                        if (has(e.values, f.alias)) {
                            e.values[next] = e.values[f.alias];
                            delete e.values[f.alias];
                        }
                        e.label = renameToken(e.label, f.alias, next);
                    }
                    sec.label = renameToken(sec.label, f.alias, next);
                    f.alias = next;
                    render();
                } });
            const handle = h("span", { class: "gg-handle", title: "Drag to reorder" }, "⠿");
            const chip = h("span", { class: "gg-chip", title: info.missing ? "This node or widget no longer exists" : "" },
                handle, h("span", { class: info.missing ? "gg-err" : "" }, `${info.title} › ${f.input}`), "as", alias,
                h("button", { class: "gg-icon", title: "Remove field", onclick: () => {
                    if (sec.fields.length === 1 && sec.entries.length
                        && !confirm(`Removing the last field also removes all ${sec.entries.length} ${noun}s. Continue?`)) return;
                    sec.fields.splice(fi, 1);
                    for (const e of sec.entries) delete e.values?.[f.alias];
                    if (!sec.fields.length) sec.entries = [];
                    render();
                } }, "✕"));
            makeSortable(chip, handle, fi, `${key}-fields`, (from, to) => { if (moveItem(sec.fields, from, to)) render(); });
            return chip;
        });

        const datalists = sec.fields.map((f, fi) => {
            const info = widgetInfo(graph, f);
            return info.kind === "combo"
                ? h("datalist", { id: `gg-dl-${key}-${fi}` }, info.options.map((o) => h("option", { value: o }))) : null;
        });

        // entries table
        const table = h("table", { class: "gg-table" },
            h("tr", {}, h("th", {}, "#"), h("th", {}, "label"), sec.fields.map((f) => h("th", {}, f.alias)), h("th", {})));

        sec.entries.forEach((entry, ei) => {
            entry.values = entry.values || {};
            const preview = h("span", { class: "gg-preview" });
            previews[ei] = preview;
            const labelIn = h("input", { value: entry.label || "", placeholder: "template", style: "width: 150px",
                oninput: (ev) => { entry.label = ev.target.value; updatePreviews(); } });
            const handle = h("span", { class: "gg-handle", title: "Drag to reorder" }, "⠿");
            const tr = h("tr", {},
                h("td", { class: "gg-idx" }, handle, ei + 1),
                h("td", {}, labelIn, preview),
                sec.fields.map((f, fi) => h("td", {}, valueEditor(f, entry, updatePreviews, datalists[fi]?.id))),
                h("td", { class: "gg-acts" },
                    h("button", { class: "gg-icon", title: "Duplicate", disabled: sec.entries.length >= MAX_ENTRIES, onclick: () => {
                        sec.entries.splice(ei + 1, 0, JSON.parse(JSON.stringify(entry)));
                        render();
                    } }, "⧉"),
                    h("button", { class: "gg-icon", title: "Remove", onclick: () => { sec.entries.splice(ei, 1); render(); } }, "−")));
            makeSortable(tr, handle, ei, `${key}-entries`, (from, to) => { if (moveItem(sec.entries, from, to)) render(); });
            table.append(tr);
        });

        const labelTpl = h("input", { value: sec.label, style: "width: 260px",
            oninput: (ev) => { sec.label = ev.target.value; updatePreviews(); } });

        const fillArea = h("div");
        const fillBtn = h("button", { disabled: !sec.fields.length, onclick: () => {
            fillArea.replaceChildren(fillPanel(sec, () => { fillArea.replaceChildren(); render(); }));
        } }, "Fill…");

        wrap.append(
            h("h4", {}, title, h("span", { class: "gg-count" }, `${sec.entries.length} ${noun}s`)),
            h("div", { class: "gg-row" }, chips, nodeSel, inputSel, addField),
            h("div", { class: "gg-row" }, "Label template", labelTpl,
                h("span", { class: "gg-help" }, "{auto} {index} {alias} {alias:name} {alias:.2f} {alias:.30} {alias:%}")),
            sec.fields.length ? h("div", { class: "gg-tablewrap" }, table)
                : h("div", { class: "gg-help" }, "Add a field to start, e.g. LoRA › strength_model."),
            h("div", { class: "gg-row", style: "margin-top: 8px" },
                h("button", { disabled: !sec.fields.length || sec.entries.length >= MAX_ENTRIES,
                    onclick: () => { sec.entries.push({ label: "", values: {} }); render(); } },
                    `+ ${noun}`),
                fillBtn,
                h("button", { disabled: !sec.entries.length, onclick: () => {
                    if (confirm(`Remove all ${sec.entries.length} ${noun}s?`)) { sec.entries = []; render(); }
                } }, "Clear")),
            fillArea,
            ...datalists.filter(Boolean),
        );
        updatePreviews();
        return wrap;
    }

    function valueEditor(f, entry, changed, listId) {
        const info = widgetInfo(graph, f);
        const v = entry.values[f.alias] ?? "";
        const set = (val) => {
            if (val === "" || val == null) delete entry.values[f.alias];
            else entry.values[f.alias] = val;
            changed();
        };
        const hint = "blank = unchanged";
        if (info.kind === "combo") {
            return h("input", { list: listId, value: v, placeholder: hint, style: "width: 260px",
                oninput: (ev) => set(ev.target.value) });
        }
        if (info.kind === "number") {
            return h("input", { value: v, placeholder: hint, inputmode: "decimal", style: "width: 90px",
                oninput: (ev) => set(ev.target.value.trim()) });
        }
        if (info.kind === "boolean") {
            const sel = h("select", { onchange: (ev) => set(ev.target.value) },
                h("option", { value: "" }, "unchanged"), h("option", { value: "true" }, "true"), h("option", { value: "false" }, "false"));
            sel.value = String(v);
            return sel;
        }
        return h("textarea", { rows: 2, placeholder: `${hint}; {v} = current text`, oninput: (ev) => set(ev.target.value) }, v);
    }

    function fillPanel(sec, done) {
        const fieldSel = h("select", {}, sec.fields.map((f) => h("option", { value: f.alias }, f.alias)));
        const inputs = h("span", { class: "gg-row", style: "margin: 0" });
        const status = h("span", { class: "gg-help" });
        // getValues returns the values, or a string saying why there are none.
        let getValues = () => [];
        const count = () => {
            const v = getValues();
            status.textContent = typeof v === "string" ? v : `${v.length} values`;
        };

        const build = () => {
            const info = widgetInfo(graph, sec.fields.find((x) => x.alias === fieldSel.value));
            inputs.replaceChildren();
            if (info.kind === "number") {
                const cur = Number.isFinite(Number(info.current)) ? Number(info.current) : 0;
                const integer = info.integer || Number.isInteger(cur);
                const step0 = integer ? 1 : 0.1;
                const a = h("input", { value: String(cur), style: "width: 80px", oninput: count });
                const b = h("input", { value: String(+(cur + 4 * step0).toFixed(6)), style: "width: 80px", oninput: count });
                const s = h("input", { value: String(step0), style: "width: 70px", oninput: count });
                inputs.append("from", a, "to", b, "step", s);
                getValues = () => {
                    const x = Number(a.value), y = Number(b.value), st = Math.abs(Number(s.value)) * (y >= x ? 1 : -1);
                    if (!isFinite(x) || !isFinite(y) || !st) return "enter from, to and step";
                    // The small epsilon keeps 0.6..1.0 step 0.2 from losing its end point to float error.
                    const n = Math.floor((y - x) / st + 1e-9) + 1;
                    if (n > MAX_FILL) return `over ${MAX_FILL} values`;
                    const vals = Array.from({ length: Math.max(n, 0) }, (_, i) => {
                        const v = x + i * st;
                        return String(info.integer ? Math.round(v) : +v.toFixed(6));
                    });
                    return [...new Set(vals)];
                };
            } else if (info.kind === "combo") {
                const cur = String(info.current ?? "");
                const sep = Math.max(cur.lastIndexOf("/"), cur.lastIndexOf("\\"));
                const p = h("input", { value: sep >= 0 ? `${cur.slice(0, sep + 1)}*` : "", style: "width: 300px",
                    oninput: count, placeholder: "wildcard, e.g. *epoch[468]*" });
                inputs.append("matching", p);
                getValues = () => {
                    if (!p.value.trim()) return "type a wildcard";
                    let re;
                    try { re = globRegex(p.value.trim()); } catch { return "invalid wildcard"; }
                    const hits = info.options.filter((o) => re.test(String(o).replace(/\//g, "\\")))
                        .sort((m, n) => String(m).localeCompare(String(n), undefined, { numeric: true }));
                    return hits.length > MAX_FILL ? `over ${MAX_FILL} values` : hits;
                };
            } else if (info.kind === "boolean") {
                inputs.append("true and false");
                getValues = () => ["true", "false"];
            } else {
                const t = h("textarea", { rows: 4, placeholder: "one value per line; {v} = current text", oninput: count });
                inputs.append(t);
                getValues = () => {
                    const lines = t.value.split("\n").map((v) => v.trim()).filter(Boolean);
                    return lines.length > MAX_FILL ? `over ${MAX_FILL} values` : lines;
                };
            }
            count();
        };
        fieldSel.onchange = build;
        build();

        const apply = (mode) => {
            const alias = fieldSel.value;
            const values = getValues();
            if (typeof values === "string" || !values.length) { count(); return; }
            const next = mode === "cross" && sec.entries.length
                ? sec.entries.flatMap((e) => values.map((v) => ({ ...e, values: { ...e.values, [alias]: v } })))
                : mode === "prepend"
                    ? values.map((v) => ({ label: "", values: { [alias]: v } })).concat(sec.entries)
                    : sec.entries.concat(values.map((v) => ({ label: "", values: { [alias]: v } })));
            if (next.length > MAX_ENTRIES) {
                status.textContent = `that makes ${next.length} entries; the limit is ${MAX_ENTRIES}`;
                return;
            }
            sec.entries = next;
            done();
        };
        return h("div", { class: "gg-fill" }, "Fill", fieldSel, inputs, status,
            h("button", { onclick: () => apply("append"), title: "Add one entry per value at the end" }, "Append"),
            h("button", { onclick: () => apply("prepend"), title: "Add one entry per value at the start" }, "Prepend"),
            h("button", { onclick: () => apply("cross"), title: "Every existing entry times every value" }, "Cross with existing"),
            h("button", { onclick: () => done() }, "Cancel"));
    }

    dialog.append(h("div", { class: "gg-wrap" },
        h("div", { class: "gg-head" }, h("h3", {}, "Graph Grid"),
            h("button", { title: "Swap rows and columns", onclick: () => { [cfg.x, cfg.y] = [cfg.y, cfg.x]; render(); } }, "⇄ swap X/Y"),
            h("button", { class: "gg-icon", title: "Close without saving", onclick: cancel }, "✕")),
        body,
        h("div", { class: "gg-foot" }, total, h("span", { style: "flex: 1" }),
            h("button", { onclick: cancel }, "Cancel"),
            h("button", { class: "gg-primary", onclick: save }, "Save"))));
    dialog.addEventListener("cancel", (ev) => { ev.preventDefault(); cancel(); });
    document.body.append(dialog);
    render();
    dialog.showModal();
    if (!loraFiles) loadLoraFiles().then(() => { if (dialog.open) render(); });
    return dialog;
}

function hideWidget(w) {
    w.hidden = true;
    w.options = { ...(w.options || {}), hidden: true };
    w.computeSize = () => [0, -4];
    if (w.element) w.element.style.display = "none";
    if (w.inputEl) w.inputEl.style.display = "none";
}

// ---------- run progress ----------

const PROGRESS_DONE_MS = 2500;
const running = new Set();

function progressText({ done, total, stage, columns }) {
    if (stage === "drawing") return `Drawing the grid (${total} cells)…`;
    if (stage === "done") return `Done: ${total} cells`;
    if (!done || !columns) return `Rendering cell ${done} / ${total}`;
    const row = Math.floor((done - 1) / columns) + 1;
    const col = ((done - 1) % columns) + 1;
    return `Rendering cell ${done} / ${total} · row ${row}, column ${col}`;
}

function showProgress(el, detail) {
    const { done, total, stage } = detail;
    const pct = total ? Math.min(100, Math.round((done / total) * 100)) : 0;
    el.replaceChildren(h("div", { class: "gg-mini-line" }, progressText(detail)),
        h("div", { class: "gg-bar" }, h("div", { style: `width: ${stage === "cells" ? pct : 100}%` })));
}

// The summary is part of the node body, which zoomed-out views (and Nodes 2.0's simplified view) leave
// out, so the count also goes in the title. serialize() keeps the real title in saved workflows.
function setTitleProgress(node, detail) {
    if (node.ggTitle === undefined) node.ggTitle = node.title;
    const tag = detail.stage === "cells" ? `${detail.done}/${detail.total}` : "drawing…";
    node.title = `${node.ggTitle} · ${tag}`;
}

function restoreTitle(node) {
    if (node.ggTitle === undefined) return;
    node.title = node.ggTitle;
    node.ggTitle = undefined;
}

api.addEventListener("graphgrid.progress", ({ detail }) => {
    const node = nodeByPath(app.graph, detail?.node);
    if (!node?.ggProgress) return;
    running.add(node);
    node.ggProgress(detail);
    if (detail.stage === "done") {
        running.delete(node);
        setTimeout(() => node.ggRefresh?.(), PROGRESS_DONE_MS);
    }
});

for (const ev of ["execution_error", "execution_interrupted"]) {
    api.addEventListener(ev, () => {
        for (const node of running) node.ggRefresh?.();
        running.clear();
    });
}

app.registerExtension({
    name: "graphgrid.editor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GraphGrid") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;
            const configWidget = node.widgets.find((w) => w.name === "config");
            if (!configWidget) return r;
            injectCss();
            hideWidget(configWidget);

            const mini = h("div", { class: "gg-mini" });
            const refresh = () => {
                restoreTitle(node);
                renderSummary(mini, parseConfig(configWidget.value));
                node.setDirtyCanvas?.(true, true);
            };
            node.addWidget("button", "Edit grid…", null, () => openEditor(node, configWidget, refresh), { serialize: false });
            if (typeof node.addDOMWidget === "function") {
                node.addDOMWidget("gg_summary", "div", mini, {
                    serialize: false, getValue: () => "", setValue: () => {},
                    getMinHeight: () => SUMMARY_HEIGHT, getMaxHeight: () => SUMMARY_HEIGHT,
                });
            }
            node.ggRefresh = refresh;
            node.ggProgress = (detail) => {
                showProgress(mini, detail);
                if (detail.stage === "done") restoreTitle(node);
                else setTitleProgress(node, detail);
                node.setDirtyCanvas?.(true, true);
            };
            const serialize = node.serialize;
            node.serialize = function () {
                const data = serialize.apply(this, arguments);
                if (node.ggTitle !== undefined) data.title = node.ggTitle;
                return data;
            };
            const onConfigure = node.onConfigure;
            node.onConfigure = function () {
                const res = onConfigure?.apply(this, arguments);
                hideWidget(configWidget);
                refresh();
                return res;
            };
            refresh();
            if (node.size[0] < 320) node.setSize([320, node.size[1]]);
            return r;
        };
    },
});
