from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from graph.nodes.assemblyai_asr import assemblyai_asr
from graph.nodes.group_utterances import group_utterances
from graph.nodes.translate_group import translate_group
from graph.state import GraphState


def fan_out_translate(state: GraphState) -> list[Send]:
    """根据 utterance_groups 数量并行发起翻译任务。"""
    return [
        Send("translate_group", {"group_text": group})
        for group in state["utterance_groups"]
    ]


builder = StateGraph(GraphState)

builder.add_node("assemblyai_asr", assemblyai_asr)
builder.add_node("group_utterances", group_utterances)
builder.add_node("translate_group", translate_group)

builder.add_edge(START, "assemblyai_asr")
builder.add_edge("assemblyai_asr", "group_utterances")
builder.add_conditional_edges("group_utterances", fan_out_translate, ["translate_group"])
builder.add_edge("translate_group", END)

graph = builder.compile()
