"""
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 """


import gradio as gr
import os
import pyaudio
import pydub
import time
import vertexai
import wave
from datetime import datetime
from google.cloud import speech, translate
from queue import Queue
from termcolor import cprint
from threading import Thread
from typing import List, Optional
from vertexai.language_models import TextGenerationModel
from vertexai.preview.language_models import ChatModel


class MicrophoneStream:

    def __init__(
            self,
            project_id: str,
            region: str,
            channels: int = 1,
            frame_rate: int = 32000,
            audio_format: int = pyaudio.paInt16,
            chunk_size: int = 1024,
            target_record_seconds: int = 2,
            silence_threshold: int = -61,
            transcription_name: Optional[str] = None,
            output_dir: Optional[str] = None,
            stop_phrase: str = "stop recording",
            temperature: float = 0.2,
            max_output_tokens: int = 1024,
            diarization_model: str = "text-bison@001",
            diarization_prompt: Optional[str] = None,
            chat_model: str = "chat-bison@001",
            chat_prompt: Optional[str] = None,
            alternative_language_codes: Optional[List[str]] = None,
            top_p: float = 0.8,
            top_k: int = 40,
            verbose: bool = True
    ):
        self.project_id = project_id
        self.region = region
        vertexai.init(project=project_id, location=region)

        self.channels = channels
        self.frame_rate = frame_rate
        self.audio_format = audio_format
        self.chunk_size = chunk_size
        self.target_record_seconds = target_record_seconds
        self.target_frame_count = (frame_rate * target_record_seconds) / chunk_size
        self.silence_threshold = silence_threshold # decibels
        self.transcription_name = transcription_name
        self.output_dir = output_dir if output_dir else os.getcwd()
        self.stop_phrase = stop_phrase
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.diarization_model = diarization_model
        self.diarization_prompt = diarization_prompt
        self.chat_model = chat_model
        self.chat_prompt = chat_prompt
        self.alternative_language_codes = alternative_language_codes
        self.chat = None
        self.top_p = top_p
        self.top_k = top_k
        self.verbose = verbose
        self.start_time = None
        self.start_datetime = None
        self.messages = Queue()
        self.recordings = Queue()
        self.raw_transcription_inputs = Queue()
        self.enhanced_transcription_inputs = Queue()
        self.outputs = []
        self.chat_outputs = []
        self.temp_outputs = []
        self.speech_client = speech.SpeechClient()

        # gradio specific parameters
        self.gradio_is_running = False
        self.gradio_running_audio_segment = pydub.AudioSegment.empty()
        self.gradio_active_chat = []
        self.gradio_corrected_chat = []
        self.gradio_latest_recommendations = ""

    def start_recording(self):
        """Start recording"""
        # start record microphone thread
        self.messages.put(True)
        record = Thread(target=self.record_microphone)
        record.start()
        self.start_time = time.time()
        self.start_datetime = datetime.fromtimestamp(self.start_time)

        # start speech recognition thread
        transcribe = Thread(target=self.speech_recognition)
        transcribe.start()

        # start enhanced transcription thread
        enhance_transcription = Thread(target=self.get_llm_doctor_patient_speech_diarization)
        enhance_transcription.start()

        # start latest recommendations thread
        latest_recommendations = Thread(target=self.get_latest_recommendations)
        latest_recommendations.start()

    def stop_recording(self):
        """Stop recording"""
        self.messages.get()

        if self.verbose is True:
            print("Recording stopped.")
            self.write_transcription_output()

    def write_transcription_output(self):
        """Write transcription output"""

        output_str = ("-" * 80).join(self.outputs)
        chat_output_str = ("-" * 80).join(self.chat_outputs)
        output_lines = output_str.split("\n")
        chat_output_lines = chat_output_str.split("\n")

        # open the file in write mode
        output_file_name = os.path.join(
            self.output_dir,
            self.transcription_name.split(".")[0] + "-" +
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt")
        with open(os.path.join(self.output_dir, output_file_name), 'w') as output_file:
            # write each string from the list to the file
            for output_line, chat_output_line in zip(output_lines, chat_output_lines):
                # line = line.lstrip()
                output_file.write(output_line + "\n")
                output_file.write(chat_output_line + "\n")

    def record_microphone(self):
        """Record microphone"""

        p = pyaudio.PyAudio()

        stream = p.open(format=self.audio_format,
                        channels=self.channels,
                        rate=self.frame_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size)

        sample_width = p.get_sample_size(pyaudio.paInt16)
        frames = []

        while not self.messages.empty():
            data = stream.read(self.chunk_size)

            audio_segment = pydub.AudioSegment(
                data=data,
                sample_width=sample_width,
                channels=self.channels,
                frame_rate=self.frame_rate
            )

            frames.append(data)

            if len(frames) >= self.target_frame_count and audio_segment.dBFS < self.silence_threshold:
                self.recordings.put(frames.copy())
                frames = []

        stream.stop_stream()
        stream.close()
        p.terminate()

    def get_llm_doctor_patient_speech_diarization(self):
        """Get LLM Doctor Patient Speech Diarization"""

        while not self.messages.empty():
            temp_outputs = self.raw_transcription_inputs.get()

            parameters = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k
            }
            model = TextGenerationModel.from_pretrained(self.diarization_model)
            prompt = self.diarization_prompt

            if len(self.outputs) > 0:
                input_text = ("-" * 80).join(temp_outputs)
            else:
                input_text = ("-" * 80).join(temp_outputs + self.outputs[-1:])

            import re
            pattern = r"Time: (\d{1,2}:\d{2}:\d{2}) / (\d{4}-\d{2}-\d{2} \d{2}:\d{2} [APap][Mm])"

            matches = re.findall(pattern, input_text)

            for match in matches:
                print("Time:", match[0])
                print("Timestamp:", match[1])
                print()

            prompt = prompt.format(input_text=input_text, timestamp_hint=list(matches))
            response = model.predict(prompt, **parameters)

            enhanced_transcription = response.text.split("output:")[-1]

            self.outputs.append(enhanced_transcription)
            cprint(response.text, "blue", attrs=["bold"])

            if self.gradio_is_running:
              for corrected_message in enhanced_transcription.split("Time:"):
                if corrected_message == "Time:":
                  continue
                self.gradio_corrected_chat.append(("Time:" + corrected_message, "---"))

            if not self.chat:
                self.create_chat()

            self.enhanced_transcription_inputs.put(enhanced_transcription)

    def create_chat(self):
        """Create chat"""
        vertexai.init(project=self.project_id, location=self.region)
        chat_model = ChatModel.from_pretrained(self.chat_model)

        context = self.chat_prompt

        chat = chat_model.start_chat(
            context=context,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        self.chat = chat

    def get_latest_recommendations(self):
        """Get latest recommendations"""

        while not self.messages.empty():
            enhanced_transcription = self.enhanced_transcription_inputs.get()

            chat_output = self.chat.send_message(enhanced_transcription)

            if chat_output and chat_output != "" and chat_output != '""':
                cprint(chat_output, "red", attrs=["bold"])

                if self.gradio_is_running:
                  self.gradio_latest_recommendations = chat_output

            self.chat_outputs.append(chat_output.text)

    def print_response(self, response: speech.RecognizeResponse, current_time):
        """Print response"""
        if len(response.results) > 0:
            self.print_result(response.results[-1], current_time)

    def print_result(self, result: speech.SpeechRecognitionResult, current_time):
        """Print result"""
        current_datetime = datetime.fromtimestamp(current_time)
        duration = current_datetime - self.start_datetime

        best_alternative = result.alternatives[0]
        cprint("-" * 80, "white")
        cprint("Time: {} / {}".format(str(duration).split('.')[0], current_datetime.strftime('%Y-%m-%d %I:%M %p %Z')), "white")
        cprint(f"language_code: {result.language_code}", "white")
        cprint(f"transcript:    {best_alternative.transcript}", "white")
        cprint(f"confidence:    {best_alternative.confidence:.0%}", "white")


        if self.stop_phrase in best_alternative.transcript.lower():
            self.stop_recording()

        if result.language_code != "en-us":
            client = translate.TranslationServiceClient()

            response = client.translate_text(
                parent=f"projects/{self.project_id}",
                contents=[best_alternative.transcript],
                target_language_code="en",
            )
            translation = response.translations[0]
            print(f"{result.language_code} → en : {translation.translated_text}")
            if self.gradio_is_running:
              gradio_output = f"""\
              Time: {str(duration).split('.')[0]} / {current_datetime.strftime('%Y-%m-%d %I:%M %p %Z')}
              language_code: {result.language_code}
              {best_alternative.transcript}
              {result.language_code} → en : {translation.translated_text}
              """
            self.temp_outputs.append(translation.translated_text)
            self.gradio_active_chat.append((gradio_output, f"confidence:    {best_alternative.confidence:.0%}",))
        else:
            if self.gradio_is_running:
              gradio_output = f"""\
              Time: {str(duration).split('.')[0]} / {current_datetime.strftime('%Y-%m-%d %I:%M %p %Z')}
              language_code: {result.language_code}
              {best_alternative.transcript}
              """
            self.gradio_active_chat.append((gradio_output, f"confidence:    {best_alternative.confidence:.0%}",))

            self.temp_outputs.append(best_alternative.transcript)


        if len(self.temp_outputs) > 2:
            self.raw_transcription_inputs.put(self.temp_outputs)
            self.temp_outputs = []

    def speech_recognition(self):
        """Speech recognition"""

        if self.transcription_name:
            cprint(f'Stream for {self.transcription_name}', "yellow")

        cprint('Listening, say "Stop Recording" to stop.', "yellow")
        cprint("=====================================================", "yellow")

        while not self.messages.empty():
            current_time = time.time()
            frames = self.recordings.get()

            audio = speech.RecognitionAudio(content=b''.join(frames))
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                enable_automatic_punctuation=True,
                model="latest_short",
                sample_rate_hertz=self.frame_rate,
                language_code="en-US",
                alternative_language_codes=self.alternative_language_codes,
                max_alternatives=1,
            )

            response = self.speech_client.recognize(request={"config": config, "audio": audio})
            self.print_response(response, current_time)

    def gradio_callback(self, audio):
        """Gradio Callback"""

        current_time = time.time()

        with wave.open(audio, "rb") as wave_file:
            FRAME_RATE = wave_file.getframerate()

        audio_segment = pydub.AudioSegment.from_file(file = audio, format = "wav")
        self.gradio_running_audio_segment += audio_segment

        if audio_segment.dBFS > self.silence_threshold:
          return self.gradio_active_chat
        else:
          pass

        speech_audio = speech.RecognitionAudio(content=self.gradio_running_audio_segment.raw_data)
        self.gradio_running_audio_segment = pydub.AudioSegment.empty()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            enable_automatic_punctuation=True,
            model="latest_short",
            sample_rate_hertz=FRAME_RATE,
            language_code="en-US",
            alternative_language_codes=self.alternative_language_codes,
            max_alternatives=1,
        )

        response = self.speech_client.recognize(request={"config": config, "audio": speech_audio})
        self.print_response(response, current_time)
        return self.gradio_active_chat

    def run_gradio(self):
      """Run Gradio"""

      self.gradio_is_running = True
      self.start_time = time.time()
      self.start_datetime = datetime.fromtimestamp(self.start_time)

      # start enhance transcription thread
      self.messages.put(True)
      enhance_transcription = Thread(target=self.get_llm_doctor_patient_speech_diarization)
      enhance_transcription.start()

      # start latest recommendations thread
      latest_recommendations = Thread(target=self.get_latest_recommendations)
      latest_recommendations.start()

      def update_gradio_outputs():
        return self.gradio_corrected_chat, self.gradio_latest_recommendations

      with gr.Blocks() as demo:
        gr.Markdown(
        """
        ## Doctor Scribbles Demo

       <img src="https://storage.googleapis.com/aadev-2541-public-assets/doctor-scribbles-img-1.png" alt="drawing" style="width:200px;"/>

        """)

        with gr.Row():
          microphone = gr.Audio(source="microphone", type="filepath", streaming=True)

        with gr.Row():
          with gr.Column(scale=1):
            speech_to_text_output = gr.Chatbot(elem_id="chatbot-1", label="Speech-to-Text Output").style(height=500)

          with gr.Column(scale=1):
            llm_enhanced_transcription_output = gr.Chatbot(elem_id="chatbot-2", label="LLM Enhanced Transcription").style(height=500)


        with gr.Row():
          latest_recommendations = gr.Textbox(label="Recommendations")

        microphone.stream(
            self.gradio_callback,
            inputs=[microphone],
            outputs=[speech_to_text_output]
        )

        microphone.stop_recording(self.stop_recording)

        demo.load(update_gradio_outputs,
                  inputs=None,
                  outputs=[llm_enhanced_transcription_output, latest_recommendations],
                  every=0.5)

      demo.queue()
      demo.launch(server_name="0.0.0.0", server_port=8080, debug=True)
