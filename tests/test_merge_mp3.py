"""测试 MP3 合并：使用 ffmpeg concat 合并 tests/outputs/tts 中的分段文件，验证合并结果。"""

import glob
import os
import subprocess
import tempfile

TESTS_DIR = os.path.dirname(__file__)
SEGMENTS_DIR = os.path.join(TESTS_DIR, "outputs", "tts")
OUTPUT_FILE = os.path.join(SEGMENTS_DIR, "merged_test.mp3")


def get_duration(filepath):
    """使用 ffprobe 获取音频时长（秒）。"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            filepath,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def merge_mp3(segment_paths, output_path):
    """使用 ffmpeg concat 合并 MP3 文件。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for seg_path in segment_paths:
            f.write(f"file '{os.path.abspath(seg_path)}'\n")
        concat_list = f.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", output_path],
            check=True,
            capture_output=True,
        )
    finally:
        os.unlink(concat_list)


def main():
    # 查找所有分段文件
    segments = sorted(glob.glob(os.path.join(SEGMENTS_DIR, "segment_*.mp3")))
    assert segments, f"未找到分段文件: {SEGMENTS_DIR}/segment_*.mp3"

    print(f"找到 {len(segments)} 个分段文件:")
    total_duration = 0.0
    for seg in segments:
        dur = get_duration(seg)
        size = os.path.getsize(seg)
        print(f"  {os.path.basename(seg)}: {dur:.2f}s, {size} bytes")
        total_duration += dur

    print(f"\n分段总时长: {total_duration:.2f}s")

    # 合并
    print(f"\n正在合并...")
    merge_mp3(segments, OUTPUT_FILE)

    # 验证合并结果
    merged_duration = get_duration(OUTPUT_FILE)
    merged_size = os.path.getsize(OUTPUT_FILE)
    print(f"合并完成: {OUTPUT_FILE}")
    print(f"  时长: {merged_duration:.2f}s, 大小: {merged_size} bytes")

    # 检查时长误差（允许 0.5s 以内）
    diff = abs(merged_duration - total_duration)
    print(f"  时长差: {diff:.2f}s")
    assert diff < 0.5, f"时长偏差过大: {diff:.2f}s"

    print("\n测试通过!")


if __name__ == "__main__":
    main()
