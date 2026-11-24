"""The Royale results page, served by Comfy itself: http://<comfy>:8188/h3royale

  GET  /h3royale                      the page (royale_page/index.html, read from disk on every request)
  GET  /h3royale/runs                 every run folder in output/h3_royale, newest first
  GET  /h3royale/run/<id>/<file>      royale.json, votes.json or a take's mp4 (range requests work, so video seeks)
  POST /h3royale/run/<id>/votes       saves the thumbs and comments as votes.json in the run folder
  POST /h3royale/run/<id>/queue       {arm, take, what: upscale|royale, arms, mode: run|plan}: builds a job from
                                      the run's job.json with that take as the baseline (royale_jobs.py) and
                                      puts it in Comfy's queue
"""
import asyncio
import importlib
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
                with open(os.path.join(root, rid, "royale.json"), "rb") as f:
                    raw = f.read()
                m = json.loads(raw)
            except (OSError, ValueError):
                continue
            takes = [t for a in m.get("arms", []) for t in a.get("takes", [])]
            out.append({"id": rid, "status": m.get("status"), "arms": len(m.get("arms", [])), "takes": len(takes),
                        "origin": m.get("origin"),
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
        if name.endswith(".json"):
            # read it and let go at once: a run rewrites royale.json after every take, and on Windows the
            # rewrite can't happen while this file is open
            data = None
            for _ in range(20):
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                    break
                except OSError:
                    await asyncio.sleep(0.05)
            if data is None:
                raise web.HTTPServiceUnavailable()
            return web.Response(body=data, content_type="application/json", headers={"Cache-Control": "no-store"})
        return web.FileResponse(path)

    @routes.post("/h3royale/run/{run}/queue")
    async def queue(request):
        import aiohttp

        from . import royale_jobs
        importlib.reload(royale_jobs)
        d = _run_dir(request)

        def fail(msg, status=400):
            return web.json_response({"error": msg}, status=status)
        try:
            body = await request.json()
            with open(os.path.join(d, "royale.json"), "rb") as f:
                man = json.loads(f.read())
        except (OSError, ValueError):
            return fail("couldn't read the request or this run's royale.json")
        try:
            with open(os.path.join(d, "job.json"), "rb") as f:
                job = json.loads(f.read())
        except (OSError, ValueError):
            return fail("This run has no saved job, so a new render can't be started from it. Runs made from now on have one.", 409)
        try:
            graph, extra = royale_jobs.build_job(job, man, int(body.get("arm", -1)), int(body.get("take", -1)),
                                                 body.get("what"), body.get("arms", ""), body.get("mode", "run"))
        except (ValueError, KeyError, TypeError) as e:
            return fail(str(e))
        url = f"http://127.0.0.1:{request.url.port or 8188}/prompt"
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json={"prompt": graph, "client_id": "h3royale",
                                             "extra_data": {"extra_pnginfo": extra}}) as r:
                    data = await r.json(content_type=None)
                    if r.status != 200:
                        err = data.get("error") if isinstance(data, dict) else None
                        msg = (err.get("message") if isinstance(err, dict) else str(err or data))[:400]
                        nodes_err = data.get("node_errors") if isinstance(data, dict) else None
                        if nodes_err:
                            msg += " | " + "; ".join(f"{k}: " + ", ".join(e.get("details", e.get("message", "")) for e in v.get("errors", []))
                                                     for k, v in nodes_err.items())[:600]
                        return fail("Comfy didn't take the job: " + msg)
        except aiohttp.ClientError as e:
            return fail(f"couldn't reach Comfy's queue: {e}", 502)
        return web.json_response({"ok": True, "prompt_id": data.get("prompt_id"), "number": data.get("number"),
                                  "node_id": str(job["node_id"])})

    @routes.post("/h3royale/run/{run}/votes")
    async def votes(request):
        d = _run_dir(request)
        data = await request.json()
        if not isinstance(data, dict) or len(json.dumps(data)) > 200000:
            raise web.HTTPBadRequest()
        tmp = os.path.join(d, "votes.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        for _ in range(40):
            try:
                os.replace(tmp, os.path.join(d, "votes.json"))
                break
            except PermissionError:
                await asyncio.sleep(0.05)
        else:
            raise web.HTTPServiceUnavailable()
        return web.json_response({"ok": True})
