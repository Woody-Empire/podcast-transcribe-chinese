import os

import assemblyai as aai

from graph.state import GraphState


def assemblyai_asr(state: GraphState) -> GraphState:
    """使用 AssemblyAI 将音频转录成文本。"""
    aai.settings.base_url = "https://api.assemblyai.com"
    aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]

    audio_file = state["audio_file"]

    config = aai.TranscriptionConfig(
        speech_models=["universal-3-pro"],
        language_detection=True,
        speaker_labels=True,
    )

    transcript = aai.Transcriber().transcribe(audio_file, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    utterances = []
    if transcript.utterances:
        for u in transcript.utterances:
            utterances.append({
                "speaker": u.speaker,
                "text": u.text,
                "start": u.start,
                "end": u.end,
            })

    return {
        "transcript_text": transcript.text,
        "utterances": utterances,
    }
