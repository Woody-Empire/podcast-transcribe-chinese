"""测试 prepare_dialogue 节点：读取翻译输出，转换为 text-to-dialogue 接口格式并保存结果。"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes.prepare_dialogue import prepare_dialogue

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
TRANSLATE_FILE = os.path.join(OUTPUT_DIR, "translate_group_output.json")
GROUP_FILE = os.path.join(OUTPUT_DIR, "group_utterances_output.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "prepare_dialogue_output.json")


def main():
    assert os.path.exists(TRANSLATE_FILE), (
        f"输入文件不存在: {TRANSLATE_FILE}\n请先运行 test_translate_group.py"
    )
    assert os.path.exists(GROUP_FILE), (
        f"输入文件不存在: {GROUP_FILE}\n请先运行 test_group_utterances.py"
    )

    with open(TRANSLATE_FILE, "r", encoding="utf-8") as f:
        translate_output = json.load(f)
    with open(GROUP_FILE, "r", encoding="utf-8") as f:
        group_output = json.load(f)

    translated = translate_output["translated_groups"]
    speakers = translate_output["utterance_group_speakers"]
    utterance_groups = group_output["utterance_groups"]
    print(f"读取翻译输出: {len(translated)} 组待转换")

    result = prepare_dialogue({
        "translated_groups": translated,
        "utterance_group_speakers": speakers,
        "utterance_groups": utterance_groups,
    })

    dialogue_inputs = result["dialogue_inputs"]
    total_entries = sum(len(batch) for batch in dialogue_inputs)
    voice_ids = set()
    for batch in dialogue_inputs:
        for entry in batch:
            voice_ids.add(entry["voice_id"])

    print(f"\n转换完成:")
    print(f"  批次数: {len(dialogue_inputs)}")
    print(f"  总对话条目: {total_entries}")
    print(f"  不同 voice_id 数: {len(voice_ids)}")

    for i, batch in enumerate(dialogue_inputs[:3]):
        batch_chars = sum(len(e["text"]) for e in batch)
        print(f"\n--- 第 {i + 1} 批 ({len(batch)} 条, {batch_chars} 字符) ---")
        for entry in batch[:3]:
            text_preview = entry["text"][:60] + "..." if len(entry["text"]) > 60 else entry["text"]
            print(f"  [{entry['voice_id']}] {text_preview}")

    subtitle_items = result.get("subtitle_items", [])
    print(f"  字幕条目数: {len(subtitle_items)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "dialogue_inputs": dialogue_inputs,
            "subtitle_items": subtitle_items,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
