"""测试 ElevenLabs TTS 节点：读取翻译输出，合成语音并保存结果。"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes.elevenlabs_tts import elevenlabs_tts

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
INPUT_FILE = os.path.join(OUTPUT_DIR, "translate_group_output.json")


def main():
    assert os.path.exists(INPUT_FILE), (
        f"输入文件不存在: {INPUT_FILE}\n请先运行 test_translate_group.py"
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        translate_output = json.load(f)

    groups = translate_output["translated_groups"]
    print(f"读取翻译输出: {len(groups)} 组待合成")

    # TTS 输出到 tests/outputs/tts/
    tts_output_dir = os.path.join(OUTPUT_DIR, "tts")
    os.environ["TTS_OUTPUT_DIR"] = tts_output_dir

    result = elevenlabs_tts({"translated_groups": groups})

    print(f"\n语音合成完成: {result['tts_audio_file']}")


if __name__ == "__main__":
    main()
