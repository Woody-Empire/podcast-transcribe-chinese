"""测试 translate_group 节点：读取分组输出，翻译并保存结果。"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes.translate_group import translate_group

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
INPUT_FILE = os.path.join(OUTPUT_DIR, "group_utterances_output.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "translate_group_output.json")


def main():
    assert os.path.exists(INPUT_FILE), (
        f"输入文件不存在: {INPUT_FILE}\n请先运行 test_group_utterances.py"
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        group_output = json.load(f)

    groups = group_output["utterance_groups"]
    print(f"读取分组输出: {len(groups)} 组待翻译")

    all_translated = []

    for i, group_text in enumerate(groups):
        print(f"\n正在翻译第 {i + 1}/{len(groups)} 组...")
        result = translate_group({"group_text": group_text})
        translated = result["translated_groups"][0]
        all_translated.append(translated)
        # 打印前 200 字符预览
        print(f"  翻译预览: {translated[:200]}...")

    output = {"translated_groups": all_translated}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n全部翻译完成，共 {len(all_translated)} 组")
    print(f"结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
