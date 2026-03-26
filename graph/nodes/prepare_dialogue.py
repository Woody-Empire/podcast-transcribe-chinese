"""将翻译后的文本适配为 ElevenLabs text-to-dialogue 接口的 inputs 格式。"""

import re

from graph.state import GraphState

# Speaker 与 voice_id 的映射表
# text-to-dialogue 接口最多支持 10 个不同的 voice_id
SPEAKER_VOICE_MAP: dict[str, str] = {
    "A": "D9bZgM9Er0PhIxuW9Jqa",
    "B": "MQkiCZS3mnl44caDtxkJ",
    "C": "r6qgCCGI7RWKXCagm158",
    "D": "pU9NaAwkoR3v0Mrg3uKz",
    "E": "BWN0mOtkGHghA3CYFzFK",
    "F": "agczkAUlHLowaNnL72Cc",
    "G": "bhJUNIXWQQ94l8eI2VUf",
    "H": "dn9HtxgDwCH96MVX9iAO",
    "I": "5qr5FEpvZGzmVOPBS55W",
    "J": "9DMBSOAnMDPiFAsz1ZGK",
}

_VOICE_SLOTS = list(SPEAKER_VOICE_MAP.values())


def prepare_dialogue(state: GraphState) -> GraphState:
    """将翻译结果与 ASR speaker 信息组装为 text-to-dialogue 接口所需的 inputs 格式。

    每条发言独立为一个 batch（一次 API 调用），便于后续精确获取每条字幕的音频时长。
    超长文本在 batch 内按句子边界拆分为多个 item，但仍属同一次调用。
    """
    translated_groups = state["translated_groups"]
    group_speakers = state["utterance_group_speakers"]
    utterance_groups = state["utterance_groups"]

    if len(translated_groups) != len(group_speakers):
        raise ValueError(
            f"translated_groups 长度 ({len(translated_groups)}) "
            f"与 utterance_group_speakers 长度 ({len(group_speakers)}) 不一致"
        )

    # 运行时 speaker → voice_id 映射，跨所有 group 保持一致
    runtime_map: dict[str, str] = {}
    next_slot_idx = 0

    def get_voice_id(speaker: str) -> str:
        nonlocal next_slot_idx
        if speaker in runtime_map:
            return runtime_map[speaker]

        if speaker in SPEAKER_VOICE_MAP:
            runtime_map[speaker] = SPEAKER_VOICE_MAP[speaker]
            return runtime_map[speaker]

        if next_slot_idx >= len(_VOICE_SLOTS):
            raise ValueError(
                f"Speaker 数量超过上限 {len(_VOICE_SLOTS)}：{speaker}"
            )
        runtime_map[speaker] = _VOICE_SLOTS[next_slot_idx]
        next_slot_idx += 1
        return runtime_map[speaker]

    max_chars = 3000
    batches: list[list[dict]] = []
    subtitle_items: list[dict] = []

    for translations, speakers, originals in zip(
        translated_groups, group_speakers, utterance_groups
    ):
        for chinese, speaker, english in zip(translations, speakers, originals):
            voice_id = get_voice_id(speaker)

            # 每条发言 = 一个 batch；超长文本在 batch 内拆分
            if len(chinese) <= max_chars:
                batches.append([{"text": chinese, "voice_id": voice_id}])
            else:
                chunks = _split_text(chinese, max_chars)
                batches.append(
                    [{"text": chunk, "voice_id": voice_id} for chunk in chunks]
                )

            subtitle_items.append({
                "chinese": chinese,
                "english": english,
                "speaker": speaker,
            })

    return {"dialogue_inputs": batches, "subtitle_items": subtitle_items}


# 中英文句子结束符，保留分隔符在前一段
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])")


def _split_text(text: str, max_chars: int) -> list[str]:
    """按句子边界将超长文本拆分为不超过 max_chars 的片段。"""
    sentences = _SENTENCE_SPLIT.split(text)
    # 过滤空字符串
    sentences = [s for s in sentences if s]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current)
            current = ""
        current += sentence

    if current:
        chunks.append(current)

    # 兜底：如果某个 chunk 仍然超长（单句就超限），强制按字符截断
    result: list[str] = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            result.append(chunk[:max_chars])
            chunk = chunk[max_chars:]
        if chunk:
            result.append(chunk)

    return result
