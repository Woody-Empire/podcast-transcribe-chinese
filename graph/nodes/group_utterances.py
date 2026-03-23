from graph.state import GraphState


def group_utterances(state: GraphState) -> GraphState:
    """每50条发言分为一组，文本与 speaker 分离存储。"""
    utterances = state["utterances"]
    groups: list[list[str]] = []
    speakers: list[list[str]] = []

    for i in range(0, len(utterances), 50):
        chunk = utterances[i:i + 50]
        groups.append([u["text"] for u in chunk])
        speakers.append([u["speaker"] for u in chunk])

    return {
        "utterance_groups": groups,
        "utterance_group_speakers": speakers,
    }
