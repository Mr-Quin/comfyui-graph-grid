"""Build the example workflow and render the demo grid used for the README and banner.

    python build_example.py [--queue]

Needs a ComfyUI with this pack and the Anima models at COMFY (default http://127.0.0.1:8188).
--queue renders the demo and downloads the grid and cells into docs/src/demo/.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HOST = os.environ.get("COMFY", "http://127.0.0.1:8188")
HERE = Path(__file__).parent
ROOT = HERE.parent.parent

defs = json.loads(urllib.request.urlopen(HOST + "/object_info", timeout=60).read())


class Graph:
    def __init__(self):
        self.nodes, self.links = [], []

    def add(self, cls, pos, title=None, size=(320, 120), **inputs):
        """inputs: (node, slot) tuples become links, everything else is a widget value."""
        node = {"id": len(self.nodes) + 1, "type": cls, "pos": list(pos), "size": list(size),
                "flags": {}, "order": len(self.nodes), "mode": 0, "inputs": [], "outputs": [],
                "properties": {"Node name for S&R": cls}, "widgets_values": [], "_in": inputs}
        if title:
            node["title"] = title
        d = defs[cls]
        for i, (typ, name) in enumerate(zip(d["output"], d["output_name"])):
            node["outputs"].append({"name": name, "type": typ, "links": [], "slot_index": i})
        self.nodes.append(node)
        return node["id"]

    def finish(self):
        api = {}
        for node in self.nodes:
            d = defs[node["type"]]
            spec = {**d["input"].get("required", {}), **d["input"].get("optional", {})}
            order = d["input_order"].get("required", []) + d["input_order"].get("optional", [])
            inputs = node.pop("_in")
            api[str(node["id"])] = {"class_type": node["type"], "inputs": {},
                                    "_meta": {"title": node.get("title", node["type"])}}
            for name in order:
                if name not in inputs:
                    continue
                val = inputs[name]
                typ = spec[name][0]
                is_widget = isinstance(typ, list) or typ in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")
                if isinstance(val, tuple):
                    src, slot = val
                    lid = len(self.links) + 1
                    entry = {"name": name, "type": "COMBO" if isinstance(typ, list) else typ, "link": lid}
                    if is_widget:
                        entry["widget"] = {"name": name}
                    node["inputs"].append(entry)
                    srcnode = self.nodes[src - 1]
                    srcnode["outputs"][slot]["links"].append(lid)
                    self.links.append([lid, src, slot, node["id"], len(node["inputs"]) - 1, entry["type"]])
                    api[str(node["id"])]["inputs"][name] = [str(src), slot]
                    if is_widget:
                        node["widgets_values"].append(0 if typ == "INT" else "")
                else:
                    node["widgets_values"].append(val)
                    api[str(node["id"])]["inputs"][name] = val
                if is_widget and len(spec[name]) > 1 and spec[name][1].get("control_after_generate"):
                    node["widgets_values"].append("fixed")
        ui = {"last_node_id": len(self.nodes), "last_link_id": len(self.links), "nodes": self.nodes,
              "links": self.links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
        return ui, api


TAIL = "masterpiece, best quality, newest, safe, no humans, scenery"
PROMPTS = {
    "lighthouse": "year 2025, lighthouse on a cliff, ocean, dusk, orange sky, waves, " + TAIL,
    "fox": "year 2025, red fox, snow, winter forest, morning light, animal focus, " + TAIL,
    "street": "year 2025, rainy city street, night, neon signs, reflections, umbrella stand, " + TAIL,
}
MODELS = ["anima-base-v1.0.safetensors", "anima-aesthetic-v1.1.safetensors", "anima-preview3-base.safetensors"]
NEG = "worst quality, low quality, score_1, score_2, score_3, artist name, nsfw, text, watermark, people"

g = Graph()
unet = g.add("UNETLoader", (0, 0), title="Model", unet_name=MODELS[0], weight_dtype="default")
clip = g.add("CLIPLoader", (0, 150), clip_name="qwen_3_06b_base.safetensors", type="stable_diffusion", device="default")
vae = g.add("VAELoader", (0, 300), vae_name="qwen_image_vae.safetensors")
pos = g.add("CLIPTextEncode", (360, 0), title="Positive", size=(420, 180), clip=(clip, 0), text=PROMPTS["lighthouse"])
neg = g.add("CLIPTextEncode", (360, 220), title="Negative", size=(420, 160), clip=(clip, 0), text=NEG)
latent = g.add("EmptyLatentImage", (360, 420), width=1024, height=1024, batch_size=1)
ks = g.add("KSampler", (820, 0), title="Sampler", size=(320, 300), model=(unet, 0), seed=7, steps=30, cfg=4.5,
           sampler_name="er_sde", scheduler="simple", positive=(pos, 0), negative=(neg, 0),
           latent_image=(latent, 0), denoise=1.0)
dec = g.add("VAEDecode", (820, 350), samples=(ks, 0), vae=(vae, 0))
config = {
    "x": {"label": "{auto}", "fields": [{"node": str(pos), "input": "text", "alias": "prompt", "class": "CLIPTextEncode"}],
          "entries": [{"label": k, "values": {"prompt": v}} for k, v in PROMPTS.items()]},
    "y": {"label": "{model:name}", "fields": [{"node": str(unet), "input": "unet_name", "alias": "model", "class": "UNETLoader"}],
          "entries": [{"label": "", "values": {"model": m}} for m in MODELS]},
}
xy = g.add("GraphGrid", (1180, 0), title="Graph Grid", size=(360, 260), image=(dec, 0), config=json.dumps(config),
           cell_scale=0.4, font_size=28, save_cells=False, cells_prefix="graph_grid/cells", batch="side by side")
save = g.add("SaveImage", (1580, 0), title="Grid", size=(600, 600), images=(xy, 0), filename_prefix="graph_grid/grid")

ui, api = g.finish()
(ROOT / "example_workflows" / "scenery_by_model.json").write_text(json.dumps(ui, indent=1), encoding="utf-8")
print("wrote example_workflows/scenery_by_model.json")

if "--queue" in sys.argv:
    xy_id = str(xy)
    api[xy_id]["inputs"].update(save_cells=True, cells_prefix="graph_grid_demo/cells", cell_scale=0.5)
    api[str(save)]["inputs"]["filename_prefix"] = "graph_grid_demo/grid"
    req = urllib.request.Request(HOST + "/prompt", json.dumps({"prompt": api}).encode(), {"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    while True:
        h = json.loads(urllib.request.urlopen(f"{HOST}/history/{pid}").read()).get(pid)
        if h and h["status"].get("status_str"):
            break
        time.sleep(5)
    if h["status"]["status_str"] != "success":
        sys.exit(json.dumps(h["status"]["messages"])[-2000:])
    out = HERE / "demo"
    out.mkdir(exist_ok=True)
    for v in h["outputs"].values():
        for im in v.get("images", []):
            q = urllib.parse.urlencode({k: im[k] for k in ("filename", "subfolder", "type")})
            (out / im["filename"]).write_bytes(urllib.request.urlopen(f"{HOST}/view?{q}").read())
            print("saved", im["filename"])
