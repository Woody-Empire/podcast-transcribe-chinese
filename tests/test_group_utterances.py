"""测试 group_utterances 节点：读取 ASR 输出，分组并保存结果。"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes.group_utterances import group_utterances

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
INPUT_FILE = os.path.join(OUTPUT_DIR, "assemblyai_asr_output.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "group_utterances_output.json")


def main():
    assert os.path.exists(INPUT_FILE), (
        f"输入文件不存在: {INPUT_FILE}\n请先运行 test_assemblyai_asr.py"
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        asr_output = json.load(f)

    print(f"读取 ASR 输出: {len(asr_output['utterances'])} 段发言")

    result = group_utterances({"utterances": asr_output["utterances"]})

    groups = result["utterance_groups"]
    speakers = result["utterance_group_speakers"]
    print(f"分组完成: {len(groups)} 组")

    for i, (group, spks) in enumerate(zip(groups, speakers)):
        unique_spks = sorted(set(spks))
        print(f"\n--- 第 {i + 1} 组 ({len(group)} 条, speakers: {unique_spks}) ---")
        preview = group[0][:100] + "..." if len(group[0]) > 100 else group[0]
        print(f"  [{spks[0]}] {preview}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
