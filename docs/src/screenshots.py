"""Take the README screenshots from a running ComfyUI that has this pack and the example's models.

    COMFY=http://127.0.0.1:8188 python screenshots.py

The browser session is read-only: every non-GET request is answered locally, so nothing on the
server changes. Needs Playwright and Microsoft Edge.
"""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

HOST = os.environ.get("COMFY", "http://127.0.0.1:8188")
REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "docs"
workflow = json.loads((REPO / "example_workflows" / "scenery_by_model.json").read_text(encoding="utf-8"))
blocked = []


EXAMPLE_MODELS = sorted({n["widgets_values"][0] for n in workflow["nodes"] if n["type"] == "UNETLoader"}
                        | {v for n in workflow["nodes"] if n["type"] == "GraphGrid"
                           for e in json.loads(n["widgets_values"][0])["y"]["entries"] for v in e["values"].values()})


def guard(route, request):
    # Read-only session: nothing may change settings, saved workflows or the queue on the shared server.
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        blocked.append(f"{request.method} {request.url.replace(HOST, '')}")
        route.fulfill(status=200, content_type="application/json", body="{}")
    elif request.url.split("?")[0].endswith("/object_info"):
        # Offer the example's model names even where they aren't installed, so the Model node isn't flagged.
        response = route.fetch()
        info = response.json()
        names = info["UNETLoader"]["input"]["required"]["unet_name"][0]
        names[:] = sorted(set(names) | set(EXAMPLE_MODELS))
        route.fulfill(response=response, json=info)
    else:
        route.continue_()


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=2, color_scheme="dark")
    page = ctx.new_page()
    page.route("**/*", guard)
    page.goto(HOST, wait_until="networkidle")
    page.wait_for_function("window.app && window.app.graph", timeout=60000)
    page.wait_for_timeout(2000)
    page.keyboard.press("Escape")
    page.evaluate("async (wf) => { await window.app.loadGraphData(wf, true, true, 'scenery_by_model'); }", workflow)
    page.wait_for_timeout(1500)
    # Loading a workflow whose models aren't installed opens a "missing models" dialog; close it.
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)
    # Show the demo grid on the Save Image node and a mid-run progress state on Graph Grid.
    page.evaluate("""() => {
        const g = window.app.graph;
        const grid = g._nodes.find((n) => n.type === "GraphGrid");
        const save = g._nodes.find((n) => n.type === "SaveImage");
        save.onExecuted?.({ images: [{ filename: "grid_00001_.png", subfolder: "graph_grid_demo", type: "output" }] });
        window.__gg = grid;
        g.setDirtyCanvas(true, true);
    }""")
    page.wait_for_timeout(1500)
    # Third-party overlays (system monitors, progress readouts) describe this machine, not the node.
    page.evaluate("""() => {
        const hide = (el) => el && (el.style.display = "none");
        document.querySelectorAll('[class*="crystools" i], [id*="crystools" i]').forEach(hide);
        for (const el of document.querySelectorAll("body *")) {
            if (el.children.length === 0 && /Current node/.test(el.textContent)) {
                let p = el;
                while (p.parentElement && p.parentElement !== document.body && getComputedStyle(p).position === "static") p = p.parentElement;
                hide(p);
            }
        }
    }""")
    page.wait_for_timeout(300)

    def node_box(expr, pad=24, extra_h=0):
        return page.evaluate("""([expr, pad, extraH]) => {
            const n = eval(expr), c = window.app.canvas, ds = c.ds;
            const rect = c.canvas.getBoundingClientRect();
            const title = window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
            const [x, y] = ds.convertOffsetToCanvas([n.pos[0], n.pos[1] - title]);
            const w = n.size[0] * ds.scale, h = (n.size[1] + title) * ds.scale;
            return { x: rect.left + x - pad, y: rect.top + y - pad, width: w + pad * 2, height: h + pad * 2 + extraH };
        }""", [expr, pad, extra_h])

    # 1. Whole workflow, framed inside a safe area below the toolbars so no menus, monitors or queue badges show.
    SAFE = {"x": 90, "y": 140, "width": 1480, "height": 820}
    box = page.evaluate("""(safe) => {
        const c = window.app.canvas, rect = c.canvas.getBoundingClientRect();
        const nodes = window.app.graph._nodes.filter((n) => n.type !== "SaveImage");
        const xs = nodes.flatMap((n) => [n.pos[0], n.pos[0] + n.size[0]]);
        const ys = nodes.flatMap((n) => [n.pos[1] - 34, n.pos[1] + n.size[1]]);
        const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
        const scale = Math.min((safe.width - 40) / (x1 - x0), (safe.height - 40) / (y1 - y0));
        c.ds.scale = scale;
        const sx = safe.x - rect.left + 20 + ((safe.width - 40) - (x1 - x0) * scale) / 2;
        const sy = safe.y - rect.top + 20;
        c.ds.offset = [sx / scale - x0, sy / scale - y0];
        c.setDirty(true, true);
        return { x: safe.x, y: safe.y, width: safe.width, height: Math.min(safe.height, (y1 - y0) * scale + 40) };
    }""", SAFE)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / "workflow.png"), clip=box)

    # 2. The node itself, showing a run in progress.
    page.evaluate("() => window.__gg.ggProgress?.({ done: 7, total: 9, stage: 'cells' })")
    page.wait_for_timeout(400)
    page.screenshot(path=str(OUT / "node.png"), clip=node_box("window.__gg"))
    page.evaluate("() => window.__gg.ggRefresh?.()")

    # 3. The editor dialog, with the Fill panel open on the rows (model) axis.
    page.evaluate("() => window.__gg.widgets.find((w) => w.name.startsWith('Edit grid')).callback()")
    page.wait_for_selector("dialog.gg-dialog[open]")
    page.locator("dialog.gg-dialog .gg-sec").nth(1).get_by_role("button", name="Fill…").click()
    page.locator("dialog.gg-dialog .gg-fill input").first.fill("anima-*")
    page.wait_for_timeout(800)
    page.locator("dialog.gg-dialog").screenshot(path=str(OUT / "editor.png"))
    browser.close()

print(f"done; answered {len(blocked)} write requests locally")
