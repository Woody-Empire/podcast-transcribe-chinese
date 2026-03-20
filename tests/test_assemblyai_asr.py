"""测试 AssemblyAI ASR 节点：转录音频并保存结果。"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes.assemblyai_asr import assemblyai_asr

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "assemblyai_asr_output.json")

# 默认测试音频路径，可通过环境变量覆盖
DEFAULT_AUDIO = os.path.join(os.path.dirname(__file__), "..", "test.mp3")
AUDIO_FILE = os.environ.get("TEST_AUDIO_FILE", os.path.abspath(DEFAULT_AUDIO))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"音频文件: {AUDIO_FILE}")
    assert os.path.exists(AUDIO_FILE), f"音频文件不存在: {AUDIO_FILE}"

    print("正在调用 AssemblyAI 转录...")
    result = assemblyai_asr({"audio_file": AUDIO_FILE})

    print(f"转录完成，共 {len(result['transcript_text'])} 字符")
    print(f"发言分段: {len(result['utterances'])} 段")

    # 打印前 5 段
    for u in result["utterances"][:5]:
        print(f"  Speaker {u['speaker']}: {u['text'][:80]}...")

    # 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
