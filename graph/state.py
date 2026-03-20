from typing import TypedDict


class GraphState(TypedDict, total=False):
    audio_file: str
    transcript_text: str
    utterances: list[dict]
    utterance_groups: list[str]
