"""
Unified FastAPI backend for the merged Input (DCM Studio) + Output (Results
Dashboard) front end. Combines:

  * the websocket-driven estimation / search endpoints from input_gui
  * the REST endpoints that drive the dashboard from output_gui

The React app is served from ``frontend/dist`` (a single SPA build that
contains both sub-apps). Run on port 8000.

Run:
    ~/.python-venv/bin/python main.py
"""

import asyncio
import os

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from web_helpers import build_payload, discover_runs, paths_for, DIR, DIST


app = FastAPI(title="Unified GUI")

# ------------------------------------------------------------------ Output API

@app.get("/api/runs")
def list_runs():
    """Run ids available on disk (one per siman_results file)."""
    return {"runs": discover_runs()}


@app.get("/api/dashboard")
def get_dashboard(run_id: str | None = None):
    """Full dashboard payload for a run. Defaults to the first run found."""
    runs = discover_runs()
    if run_id is None:
        if not runs:
            raise HTTPException(status_code=404, detail="No siman_results files found.")
        run_id = runs[0]

    results_file, pert_file = paths_for(run_id)
    if not os.path.exists(results_file):
        raise HTTPException(status_code=404, detail=f"Results file not found for run '{run_id}'.")

    return build_payload(results_file, pert_file, run_id)


# -------------------------------------------------------------- Input sockets

@app.websocket("/ws/run_estimation")
async def ws_run_estimation(ws: WebSocket):
    await ws.accept()
    try:
        await handle_estimation_ws(ws)
    except WebSocketDisconnect:
        pass


async def handle_estimation_ws(ws: WebSocket):
    """
    Dummy estimation websocket — receives appData + csv, replies running/done.
    Mirrors the placeholder behaviour from input_gui until the real estimator
    is wired in.
    """
    appData = await ws.receive_json()
    csv = await ws.receive_bytes()
    print(f"[estimation] appData keys: {list(appData.keys())}")
    print(f"[estimation] csv bytes:    {len(csv)}")

    await ws.send_json({"status": "running"})
    await asyncio.sleep(1)
    await ws.send_json({"status": "done", "ll": -921.4, "bic": -1874.3, "aic": -1750.7})

    await ws.close()


@app.websocket("/ws/run_search")
async def ws_run_search(ws: WebSocket):
    await ws.accept()
    try:
        await handle_search_ws(ws)
    except WebSocketDisconnect:
        pass


async def handle_search_ws(ws: WebSocket):
    """Dummy search websocket — receives appData + csv, streams progress."""
    appData = await ws.receive_json()
    csv = await ws.receive_bytes()
    print(f"[search] appData keys: {list(appData.keys())}")
    print(f"[search] csv bytes:    {len(csv)}")

    for n in range(100):
        await asyncio.sleep(0.04)
        await ws.send_json({"status": "running", "total": 100, "complete": n + 1})

    await ws.send_json({"status": "done"})
    await ws.close()


# --------------------------------------------------------------- Static SPA

if os.path.isdir(os.path.join(DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")


@app.get("/")
def home():
    index = os.path.join(DIST, "index.html")
    if not os.path.exists(index):
        return JSONResponse(
            {"detail": "Frontend not built. Run `npm install && npm run build` in frontend/."},
            status_code=503,
        )
    return FileResponse(index)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=True,
        reload_dirs=[DIR, os.path.join(DIR, "frontend", "src")],
    )
