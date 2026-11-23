"""The Royale results page, served by Comfy itself: http://<comfy>:8188/h3royale

  GET  /h3royale                      the page (royale_page/index.html, read from disk on every request)
  GET  /h3royale/runs                 every run folder in output/h3_royale, newest first
  GET  /h3royale/run/<id>/<file>      royale.json, votes.json or a take's mp4 (range requests work, so video seeks)
  POST /h3royale/run/<id>/votes       saves the thumbs and comments as votes.json in the run folder
"""
import json
import os
import re

from aiohttp import web

import folder_paths

PAGE = os.path.join(os.path.dirname(__file__), "royale_page", "index.html")
RUN_ID = re.compile(r"^[0-9]{8}_[0-9]{6}$")
FILE = re.compile(r"^[A-Za-z0-9_.-]+\.(json|mp4)$")


def _root():
    return os.path.join(folder_paths.get_output_directory(), "h3_royale")


def _run_dir(request):
    rid = request.match_info["run"]
    if not RUN_ID.match(rid):
        raise web.HTTPNotFound()
    d = os.path.join(_root(), rid)
    if not os.path.isdir(d):
        raise web.HTTPNotFound()
    return d


def register(routes):
    @routes.get("/h3royale")
    async def page(request):
        with open(PAGE, encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html", headers={"Cache-Control": "no-store"})

    @routes.get("/h3royale/runs")
    async def runs(request):
        out = []
        root = _root()
        for rid in sorted(os.listdir(root) if os.path.isdir(root) else [], reverse=True):
            if not RUN_ID.match(rid):
                continue
            try:
                with open(os.path.join(root, rid, "royale.json"), encoding="utf-8") as f:
                    m = json.load(f)
            except (OSError, ValueError):
                continue
            takes = [t for a in m.get("arms", []) for t in a.get("takes", [])]
            out.append({"id": rid, "status": m.get("status"), "arms": len(m.get("arms", [])), "takes": len(takes),
                        "done": sum(t.get("status") == "done" for t in takes),
                        "what": "; ".join(" | ".join(a.get("diff") or []) for a in m.get("arms", [])[1:])[:160]})
        return web.json_response(out, headers={"Cache-Control": "no-store"})

    @routes.get("/h3royale/run/{run}/{file}")
    async def run_file(request):
        name = request.match_info["file"]
        if not FILE.match(name):
            raise web.HTTPNotFound()
        path = os.path.join(_run_dir(request), name)
        if not os.path.isfile(path):
            if name == "votes.json":
                return web.json_response({})
            raise web.HTTPNotFound()
        headers = {"Cache-Control": "no-store"} if name.endswith(".json") else {}
        return web.FileResponse(path, headers=headers)

    @routes.post("/h3royale/run/{run}/votes")
    async def votes(request):
        d = _run_dir(request)
        data = await request.json()
        if not isinstance(data, dict) or len(json.dumps(data)) > 200000:
            raise web.HTTPBadRequest()
        tmp = os.path.join(d, "votes.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, os.path.join(d, "votes.json"))
        return web.json_response({"ok": True})
