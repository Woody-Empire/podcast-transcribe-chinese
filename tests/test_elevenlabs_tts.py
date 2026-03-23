"""测试 ElevenLabs TTS 节点：读取 prepare_dialogue 输出，并发调用 text-to-dialogue 合成语音（带计时）。"""

import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes.elevenlabs_tts import elevenlabs_tts

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
INPUT_FILE = os.path.join(OUTPUT_DIR, "prepare_dialogue_output.json")


async def main():
    assert os.path.exists(INPUT_FILE), (
        f"输入文件不存在: {INPUT_FILE}\n请先运行 test_prepare_dialogue.py"
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dialogue_output = json.load(f)

    dialogue_inputs = dialogue_output["dialogue_inputs"]
    total_entries = sum(len(batch) for batch in dialogue_inputs)
    max_concurrency = int(os.environ.get("ELEVENLABS_TTS_MAX_CONCURRENCY", "5"))

    print(f"读取对话输入: {len(dialogue_inputs)} 批, 共 {total_entries} 条")
    print(f"最大并行数: {max_concurrency}")

    # TTS 输出到 tests/outputs/tts/
    tts_output_dir = os.path.join(OUTPUT_DIR, "tts")
    os.environ["TTS_OUTPUT_DIR"] = tts_output_dir

    start = time.time()
    result = await elevenlabs_tts({"dialogue_inputs": dialogue_inputs})
    elapsed = time.time() - start

    print(f"\n语音合成完成: {result['tts_audio_file']}")
    print(f"总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
