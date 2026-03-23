"""测试 ElevenLabs TTS 节点：读取 prepare_dialogue 输出，调用 text-to-dialogue 合成语音。"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes.elevenlabs_tts import elevenlabs_tts

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
INPUT_FILE = os.path.join(OUTPUT_DIR, "prepare_dialogue_output.json")


def main():
    assert os.path.exists(INPUT_FILE), (
        f"输入文件不存在: {INPUT_FILE}\n请先运行 test_prepare_dialogue.py"
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dialogue_output = json.load(f)

    dialogue_inputs = dialogue_output["dialogue_inputs"]
    total_entries = sum(len(batch) for batch in dialogue_inputs)
    print(f"读取对话输入: {len(dialogue_inputs)} 批, 共 {total_entries} 条")

    # TTS 输出到 tests/outputs/tts/
    tts_output_dir = os.path.join(OUTPUT_DIR, "tts")
    os.environ["TTS_OUTPUT_DIR"] = tts_output_dir

    result = elevenlabs_tts({"dialogue_inputs": dialogue_inputs})

    print(f"\n语音合成完成: {result['tts_audio_file']}")


if __name__ == "__main__":
    main()
