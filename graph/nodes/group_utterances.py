from graph.state import GraphState


def group_utterances(state: GraphState) -> GraphState:
    """每50条发言合并成一组字符串。"""
    utterances = state["utterances"]
    groups = []

    for i in range(0, len(utterances), 50):
        chunk = utterances[i:i + 50]
        text = "\n".join(
            f"Speaker {u['speaker']}: {u['text']}" for u in chunk
        )
        groups.append(text)

    return {"utterance_groups": groups}
