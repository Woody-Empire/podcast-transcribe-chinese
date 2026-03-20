import os

import requests
from dotenv import load_dotenv

from graph.state import GraphState

load_dotenv()

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


def elevenlabs_tts(state: GraphState) -> GraphState:
    """使用 ElevenLabs 将翻译后的中文文本合成为语音，输出合并后的音频文件路径。"""
    translated_groups = state["translated_groups"]

    api_key = os.environ["ELEVENLABS_API_KEY"]
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    model_id = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_v3")
    output_format = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")

    output_dir = os.environ.get("TTS_OUTPUT_DIR", "output")
    os.makedirs(output_dir, exist_ok=True)

    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }

    audio_segments = []
    for i, text in enumerate(translated_groups):
        print(f"正在合成第 {i + 1}/{len(translated_groups)} 段语音...")
        resp = requests.post(
            url,
            headers=headers,
            json={
                "text": text,
                "model_id": model_id,
            },
            params={"output_format": output_format},
        )
        resp.raise_for_status()

        segment_path = os.path.join(output_dir, f"segment_{i:03d}.mp3")
        with open(segment_path, "wb") as f:
            f.write(resp.content)
        print(f"  已保存: {segment_path} ({len(resp.content)} bytes)")
        audio_segments.append(segment_path)

    # 合并所有音频片段
    merged_path = os.path.join(output_dir, "podcast_chinese.mp3")
    with open(merged_path, "wb") as outfile:
        for seg_path in audio_segments:
            with open(seg_path, "rb") as infile:
                outfile.write(infile.read())

    print(f"语音合成完成: {merged_path}")
    return {"tts_audio_file": merged_path}
