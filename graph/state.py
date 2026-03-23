import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    audio_file: str
    transcript_text: str
    utterances: list[dict]
    utterance_groups: list[list[str]]
    utterance_group_speakers: list[list[str]]
    translated_groups: Annotated[list[list[str]], operator.add]
    dialogue_inputs: list[list[dict]]
    tts_audio_file: str
