import asyncio
import json
import os
import subprocess
import tempfile

from dotenv import load_dotenv
from elevenlabs import AsyncElevenLabs, DialogueInput

from graph.progress import send_log
from graph.state import GraphState

load_dotenv()


async def elevenlabs_tts(state: GraphState) -> GraphState:
    """使用 ElevenLabs text-to-dialogue 接口将多角色对话分批并发合成为语音，合并输出。"""
    dialogue_inputs = state["dialogue_inputs"]
    subtitle_items = state.get("subtitle_items", [])

    client = AsyncElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    model_id = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_v3")
    output_format = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
    max_concurrency = int(os.environ.get("ELEVENLABS_TTS_MAX_CONCURRENCY", "5"))

    output_dir = state.get("output_dir") or os.environ.get("TTS_OUTPUT_DIR", "output")
    os.makedirs(output_dir, exist_ok=True)

    async def log(msg: str):
        print(msg)
        await send_log(output_dir, msg)

    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_batch(i: int, batch: list[dict]) -> str:
        async with semaphore:
            inputs = [
                DialogueInput(text=item["text"], voice_id=item["voice_id"])
                for item in batch
            ]
            total_chars = sum(len(item["text"]) for item in batch)

            await log(f"正在合成第 {i + 1}/{len(dialogue_inputs)} 批语音 ({len(inputs)} 条, {total_chars} 字符)...")
            audio = client.text_to_dialogue.convert(
                inputs=inputs,
                model_id=model_id,
                output_format=output_format,
            )

            segment_path = os.path.join(output_dir, f"segment_{i:03d}.mp3")
            with open(segment_path, "wb") as f:
                async for chunk in audio:
                    f.write(chunk)

            file_size = os.path.getsize(segment_path)
            await log(f"  已保存: {segment_path} ({file_size} bytes)")
            return segment_path

    audio_segments = await asyncio.gather(
        *(process_batch(i, batch) for i, batch in enumerate(dialogue_inputs))
    )

    # 生成 0.5s 静音片段，用于拼接间隔
    gap_duration = 0.5
    silence_path = os.path.join(output_dir, "silence.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo", "-t", str(gap_duration), "-c:a", "libmp3lame", "-b:a", "128k", silence_path],
        check=True,
        capture_output=True,
    )

    # 使用 ffmpeg concat 合并所有音频片段，每段之间插入静音
    merged_path = os.path.join(output_dir, "podcast_chinese.mp3")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for i, seg_path in enumerate(audio_segments):
            f.write(f"file '{os.path.abspath(seg_path)}'\n")
            if i < len(audio_segments) - 1:
                f.write(f"file '{os.path.abspath(silence_path)}'\n")
        concat_list = f.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c:a", "libmp3lame", "-b:a", "128k", merged_path],
            check=True,
            capture_output=True,
        )
    finally:
        os.unlink(concat_list)

    # 构建字幕时间戳：每个 segment 对应一条字幕，ffprobe 获取精确时长
    subtitle_file = None
    if subtitle_items:
        subtitles = []
        global_offset = 0.0

        for seg_idx, seg_path in enumerate(audio_segments):
            duration = _get_duration(seg_path)
            if seg_idx < len(subtitle_items):
                sub = subtitle_items[seg_idx]
                subtitles.append({
                    "start": round(global_offset, 3),
                    "end": round(global_offset + duration, 3),
                    "chinese": sub["chinese"],
                    "english": sub["english"],
                    "speaker": sub["speaker"],
                })
            global_offset += duration
            if seg_idx < len(audio_segments) - 1:
                global_offset += gap_duration

        subtitle_file = os.path.join(output_dir, "subtitles.json")
        with open(subtitle_file, "w", encoding="utf-8") as f:
            json.dump({"subtitles": subtitles}, f, ensure_ascii=False)

        await log(f"字幕数据已生成: {subtitle_file} ({len(subtitles)} 条)")

    await log(f"语音合成完成: {merged_path}")
    return {"tts_audio_file": merged_path, "subtitle_file": subtitle_file}


def _get_duration(audio_path: str) -> float:
    """使用 ffprobe 获取音频文件时长（秒）。"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            audio_path,
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())
