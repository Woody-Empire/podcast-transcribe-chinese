from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from graph.nodes.assemblyai_asr import assemblyai_asr
from graph.nodes.elevenlabs_tts import elevenlabs_tts
from graph.nodes.group_utterances import group_utterances
from graph.nodes.prepare_dialogue import prepare_dialogue
from graph.nodes.translate_group import translate_group
from graph.state import GraphState


def fan_out_translate(state: GraphState) -> list[Send]:
    """根据 utterance_groups 数量并行发起翻译任务。"""
    return [
        Send("translate_group", {"group_texts": group})
        for group in state["utterance_groups"]
    ]


builder = StateGraph(GraphState)

builder.add_node("assemblyai_asr", assemblyai_asr)
builder.add_node("group_utterances", group_utterances)
builder.add_node("translate_group", translate_group)
builder.add_node("prepare_dialogue", prepare_dialogue)
builder.add_node("elevenlabs_tts", elevenlabs_tts)

builder.add_edge(START, "assemblyai_asr")
builder.add_edge("assemblyai_asr", "group_utterances")
builder.add_conditional_edges("group_utterances", fan_out_translate, ["translate_group"])
builder.add_edge("translate_group", "prepare_dialogue")
builder.add_edge("prepare_dialogue", "elevenlabs_tts")
builder.add_edge("elevenlabs_tts", END)

graph = builder.compile()
