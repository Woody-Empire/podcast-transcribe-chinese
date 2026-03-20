import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    audio_file: str
    transcript_text: str
    utterances: list[dict]
    utterance_groups: list[str]
    translated_groups: Annotated[list[str], operator.add]
