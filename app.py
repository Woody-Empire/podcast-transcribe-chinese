import asyncio
import json
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from graph.graph import graph
from graph.progress import register_queue, unregister_queue

app = FastAPI()

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

_tasks: dict[str, dict] = {}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.post("/upload")
async def upload(file: UploadFile):
    """接收音频文件，保存到临时目录，返回 task_id。"""
    task_id = uuid.uuid4().hex[:8]
    output_dir = tempfile.mkdtemp(prefix=f"podcast-{task_id}-")
    audio_path = f"{output_dir}/input_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    _tasks[task_id] = {
        "audio_path": audio_path,
        "output_dir": output_dir,
        "result": None,
    }
    return {"task_id": task_id}


@app.get("/progress/{task_id}")
async def progress(task_id: str):
    """SSE 端点：启动处理流水线并实时推送进度事件。"""
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


@app.get("/download/{task_id}")
async def download(task_id: str):
    """下载处理完成的音频文件。"""
    task = _tasks.get(task_id)
    if not task or not task.get("result"):
        return {"error": "结果未就绪"}
    return FileResponse(
        task["result"],
        media_type="audio/mpeg",
        filename="podcast_chinese.mp3",
    )
