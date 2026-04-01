import asyncio
import json
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import db
from graph.graph import graph
from graph.progress import register_queue, unregister_queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(lifespan=lifespan)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

_tasks: dict[str, dict] = {}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _get_user(request: Request) -> dict | None:
    token = request.cookies.get("session")
    if not token:
        return None
    return db.get_user_by_token(token)


# ── Pages ──────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.get("/player")
async def player_page():
    return FileResponse(static_dir / "player.html")


# ── Auth API ───────────────────────────────────────────

@app.post("/api/register")
async def register(request: Request):
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")
    if not username or not password:
        return JSONResponse({"error": "用户名和密码不能为空"}, status_code=400)
    if len(username) > 32:
        return JSONResponse({"error": "用户名过长"}, status_code=400)
    if len(password) < 4:
        return JSONResponse({"error": "密码至少4位"}, status_code=400)

    user_id = db.create_user(username, password)
    if user_id is None:
        return JSONResponse({"error": "用户名已存在"}, status_code=409)

    token = db.create_session(user_id)
    resp = JSONResponse({"username": username})
    resp.set_cookie("session", token, httponly=True, samesite="lax", max_age=86400 * 30)
    return resp


@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")

    user_id = db.verify_user(username, password)
    if user_id is None:
        return JSONResponse({"error": "用户名或密码错误"}, status_code=401)

    token = db.create_session(user_id)
    resp = JSONResponse({"username": username})
    resp.set_cookie("session", token, httponly=True, samesite="lax", max_age=86400 * 30)
    return resp


@app.post("/api/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token:
        db.delete_session(token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("session")
    return resp


@app.get("/api/me")
async def me(request: Request):
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "未登录"}, status_code=401)
    last = db.get_last_played(user["id"])
    return {"username": user["username"], "last_played": last}


# ── Podcast API ────────────────────────────────────────

@app.get("/api/podcasts")
async def list_podcasts(request: Request):
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "未登录"}, status_code=401)
    return db.get_user_podcasts(user["id"])


@app.post("/api/progress/{podcast_id}")
async def save_progress(podcast_id: str, request: Request):
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "未登录"}, status_code=401)
    body = await request.json()
    position = body.get("position", 0)
    db.save_progress(user["id"], podcast_id, position)
    return {"ok": True}


@app.get("/api/progress/{podcast_id}")
async def get_progress(podcast_id: str, request: Request):
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "未登录"}, status_code=401)
    position = db.get_progress(user["id"], podcast_id)
    return {"position": position}


# ── Podcast files ──────────────────────────────────────

@app.get("/podcast/{podcast_id}/audio")
async def podcast_audio(podcast_id: str):
    audio_path = db.PODCASTS_DIR / podcast_id / "audio.mp3"
    if not audio_path.exists():
        return JSONResponse({"error": "音频不存在"}, status_code=404)
    return FileResponse(audio_path, media_type="audio/mpeg")


@app.get("/podcast/{podcast_id}/subtitles")
async def podcast_subtitles(podcast_id: str):
    sub_path = db.PODCASTS_DIR / podcast_id / "subtitles.json"
    if not sub_path.exists():
        return JSONResponse({"error": "字幕不存在"}, status_code=404)
    return FileResponse(sub_path, media_type="application/json")


# ── Upload & Process ───────────────────────────────────

@app.post("/upload")
async def upload(file: UploadFile, request: Request):
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "未登录"}, status_code=401)

    task_id = uuid.uuid4().hex[:8]
    output_dir = tempfile.mkdtemp(prefix=f"podcast-{task_id}-")
    audio_path = f"{output_dir}/input_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    title = Path(file.filename).stem if file.filename else task_id

    _tasks[task_id] = {
        "audio_path": audio_path,
        "output_dir": output_dir,
        "result": None,
        "subtitle_file": None,
        "user_id": user["id"],
        "title": title,
    }
    return {"task_id": task_id}


@app.get("/progress/{task_id}")
async def progress(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        return StreamingResponse(
            iter([_sse("task_error", {"message": "任务不存在"})]),
            media_type="text/event-stream",
        )

    queue: asyncio.Queue = asyncio.Queue()
    register_queue(task["output_dir"], queue)

    async def run_graph():
        try:
            total_groups = 0
            done_groups = 0

            await queue.put(("step_start", {"step": "asr", "message": "正在进行语音识别..."}))

            async for chunk in graph.astream(
                {"audio_file": task["audio_path"], "output_dir": task["output_dir"]},
                stream_mode="updates",
            ):
                for node_name, update in chunk.items():
                    if node_name == "assemblyai_asr":
                        n = len(update.get("utterances", []))
                        await queue.put(("log", {"message": f"语音识别完成，共识别 {n} 条发言"}))
                        await queue.put(("step_done", {"step": "asr"}))
                        await queue.put(("step_start", {"step": "translate", "message": "正在翻译为中文..."}))

                    elif node_name == "group_utterances":
                        total_groups = len(update.get("utterance_groups", []))
                        await queue.put(("log", {"message": f"已分为 {total_groups} 组，开始并行翻译"}))

                    elif node_name == "translate_group":
                        done_groups += 1
                        await queue.put(("log", {"message": f"翻译进度: {done_groups}/{total_groups} 组完成"}))

                    elif node_name == "prepare_dialogue":
                        n = len(update.get("dialogue_inputs", []))
                        await queue.put(("log", {"message": f"对话准备完成，共 {n} 批语音待合成"}))
                        await queue.put(("step_done", {"step": "translate"}))
                        await queue.put(("step_start", {"step": "tts", "message": "正在合成语音..."}))

                    elif node_name == "elevenlabs_tts":
                        await queue.put(("step_done", {"step": "tts"}))
                        task["result"] = update.get("tts_audio_file")
                        task["subtitle_file"] = update.get("subtitle_file")

            # Persist results to data directory
            if task.get("result") and task.get("user_id"):
                podcast_dir = db.PODCASTS_DIR / task_id
                podcast_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(task["result"], podcast_dir / "audio.mp3")
                if task.get("subtitle_file"):
                    shutil.copy2(task["subtitle_file"], podcast_dir / "subtitles.json")
                db.save_podcast(task_id, task["user_id"], task["title"])

            await queue.put(("done", {"task_id": task_id}))
        except Exception as e:
            await queue.put(("task_error", {"message": str(e)}))

    async def event_stream():
        graph_task = asyncio.create_task(run_graph())
        try:
            while True:
                event_type, data = await queue.get()
                yield _sse(event_type, data)
                if event_type in ("done", "task_error"):
                    break
        finally:
            unregister_queue(task["output_dir"])
            if not graph_task.done():
                graph_task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Legacy endpoints (kept for in-session download) ───

@app.get("/download/{task_id}")
async def download(task_id: str):
    task = _tasks.get(task_id)
    if not task or not task.get("result"):
        return {"error": "结果未就绪"}
    return FileResponse(
        task["result"],
        media_type="audio/mpeg",
        filename="podcast_chinese.mp3",
    )


@app.get("/subtitles/{task_id}")
async def subtitles(task_id: str):
    task = _tasks.get(task_id)
    if not task or not task.get("subtitle_file"):
        return {"error": "字幕数据未就绪"}
    return FileResponse(
        task["subtitle_file"],
        media_type="application/json",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
