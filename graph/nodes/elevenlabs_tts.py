import os

from dotenv import load_dotenv
from elevenlabs import DialogueInput, ElevenLabs

from graph.state import GraphState

load_dotenv()


def elevenlabs_tts(state: GraphState) -> GraphState:
    """使用 ElevenLabs text-to-dialogue 接口将多角色对话分批合成为语音，合并输出。"""
    dialogue_inputs = state["dialogue_inputs"]

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    model_id = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_v3")
    output_format = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")

    output_dir = os.environ.get("TTS_OUTPUT_DIR", "output")
    os.makedirs(output_dir, exist_ok=True)

    audio_segments = []
    for i, batch in enumerate(dialogue_inputs):
        inputs = [
            DialogueInput(text=item["text"], voice_id=item["voice_id"])
            for item in batch
        ]
        total_chars = sum(len(item["text"]) for item in batch)

        print(f"正在合成第 {i + 1}/{len(dialogue_inputs)} 批语音 ({len(inputs)} 条, {total_chars} 字符)...")
        audio = client.text_to_dialogue.convert(
            inputs=inputs,
            model_id=model_id,
            output_format=output_format,
        )

        segment_path = os.path.join(output_dir, f"segment_{i:03d}.mp3")
        with open(segment_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        file_size = os.path.getsize(segment_path)
        print(f"  已保存: {segment_path} ({file_size} bytes)")
        audio_segments.append(segment_path)

    # 合并所有音频片段
    merged_path = os.path.join(output_dir, "podcast_chinese.mp3")
    with open(merged_path, "wb") as outfile:
        for seg_path in audio_segments:
            with open(seg_path, "rb") as infile:
                outfile.write(infile.read())

    print(f"语音合成完成: {merged_path}")
    return {"tts_audio_file": merged_path}
