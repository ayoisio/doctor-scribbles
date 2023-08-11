from src.microphone_stream import MicrophoneStream

microphone_stream = MicrophoneStream(
    project_id="aadev-2541",
    region="us-central1",
    alternative_language_codes=["hi-IN", "es-US"],
    transcription_name="transcription-1b-llm-corrected-output-testing.txt",
)

microphone_stream.run_gradio()
