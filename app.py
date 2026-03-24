import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from graph.graph import graph

app = FastAPI()

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    """接收音频文件，执行转录-翻译-TTS 流水线，返回中文播客音频。"""
    output_dir = tempfile.mkdtemp(prefix=f"podcast-{uuid.uuid4().hex[:8]}-")

    # 将上传文件写入临时目录
    audio_path = f"{output_dir}/input_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    result = await graph.ainvoke({
        "audio_file": audio_path,
        "output_dir": output_dir,
    })

    return FileResponse(
        result["tts_audio_file"],
        media_type="audio/mpeg",
        filename="podcast_chinese.mp3",
    )
