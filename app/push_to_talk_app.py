#!/usr/bin/env uv run
####################################################################
# Sample TUI app with a push to talk interface to the Realtime API #
# If you have `uv` installed and the `OPENAI_API_KEY`              #
# environment variable set, you can run this example with just     #
#                                                                  #
# `./app/push_to_talk_app.py`                        #
####################################################################
#
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "textual",
#     "numpy",
#     "pyaudio",
#     "pydub",
#     "sounddevice",
#     "openai[realtime]",
#     "pyautogui"
# ]
#
# [tool.uv.sources]
# openai = { path = "../", editable = true }
# ///
from __future__ import annotations

import base64
import asyncio
from typing import Any, cast, Tuple, Union, List
from typing_extensions import override
import logging
import time
import pyautogui
import subprocess
import queue
import io

from pydub import AudioSegment
from pydub.playback import play

from textual import events
from audio_util import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
from textual.app import App, ComposeResult
from textual.widgets import Button, Static, RichLog
from textual.reactive import reactive
from textual.containers import Container

from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

from random import choice
from string import ascii_uppercase

logger = logging.getLogger(__name__)
logging.basicConfig(filename='berk.log', encoding='utf-8', level=logging.DEBUG)

class SessionDisplay(Static):
    """A widget that shows the current session ID."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session ID: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)" if self.is_recording else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class RealtimeApp(App[None]):
    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
        }

        #input-container {
            height: 5;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
        }

        Input {
            width: 80%;
            height: 3;  /* Explicit height for input */
        }

        Button {
            width: 20%;
            height: 3;  /* Explicit height for button */
        }

        #bottom-pane {
            width: 100%;
            height: 82%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #status-indicator {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        #session-display {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        Static {
            color: white;
        }
    """

    client: AsyncOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event

    def __init__(self) -> None:
        super().__init__()
        self.connection = None
        self.session = None
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.a0 = None # accumulator, stores return value
        self.t0 = None # task instruction
        self.GPT_MODEL = "gpt-4o-mini"
        self.REALTIME_MODEL = "gpt-4o-realtime-preview"
        self.screenshot_name_history = []
        self.screenshot_name_length = 20
        self.KEYWORDS = ["<enter>"]
        self.processor_queue = queue.Queue()

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield SessionDisplay(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        self.run_worker(self.handle_realtime_connection())
        self.run_worker(self.send_mic_audio())
        self.run_worker(self.processor())

    async def assemble_and_run(self, transcript):
        instructions = await self.codegen(transcript)
        await self.execute(instructions, transcript)

    async def codegen(self, transcript) -> list[str]:
        """
        Acts as an intermediate language.
        """
        response = await self.client.chat.completions.create(
            model=self.GPT_MODEL,
            messages=[
                {
                    "role": "developer", 
                    "content": """
                        You are a compiler. Given the user's task, your response should be a sequence of the following instructions. If the user is not giving a task, then only output NO-OP. Do not insert numbering or ordered list. Just separate each command by enter.:
                        1. An executable UNIX command that is not dangerous. For example, rm -rf is dangerous.
                        2. SCREENSHOT
                        3. LEFT_CLICK x y (x, y are integers)
                        4. LEFT_DOUBLE_CLICK x y (x, y are integers)
                        5. RIGHT_CLICK
                        6. OPEN_TERMINAL
                        7. ANALYSIS
                        8. RET
                        9. NO-OP
                    """
                }, {
                    "role": "developer", 
                    "content": """
                        For example, if the user asks for the score of football team X versus football team Y, output the following (do not change the integers):
                        "open https://www.google.com/"
                        "LEFT_CLICK 605 561"
                        "KEYBOARD 'X vs Y'"
                        "KEYBOARD <enter>"
                        "SCREENSHOT"
                        "ANALYSIS"
                        "RET"
                    """
                }, {
                    "role": "developer",
                    "content": """
                        For example, if the user asks to open YouTube and search X, output the following (do not change the integers):
                        "open https://www.youtube.com/"
                        "LEFT_CLICK 598 142"
                        "KEYBOARD 'X'"
                        "KEYBOARD <enter>"
                        "LEFT_CLICK 638 645"
                        "WAIT 5.5"
                        "LEFT_CLICK 1245 797"
                        "LEFT_CLICK 1292 895"
                    """
                }, {
                    "role": "developer",
                    "content": """
                        For example, if the user just says hello, you should just output:
                        "NO-OP"
                    """
                }, {
                    "role": "user", 
                    "content": transcript
                }
            ]
        )

        logger.warning(f"!!! codegen_response = {response}")
        resp = response.choices[0].message.content

        # post processing
        resp = resp.split("\n")
        instructions = [s for s in resp if len(s) > 0] # filter out empty strings
        logger.warning(f"!!! processed response = {instructions}")
        
        return instructions

    async def execute(self, instructions: list[str], transcript: str) -> None:
        for inst in instructions:
            time.sleep(1.0)
            if "KEYBOARD" in inst:
                if inst[9:] in self.KEYWORDS: # key
                    if inst[9:] == "<enter>":
                        pyautogui.press('enter')
                else:
                    self.keyboard_write(inst[9:].strip("'"))
            elif "LEFT_CLICK" in inst:
                ls = inst.split(" ")
                x, y = int(ls[1]), int(ls[2])
                self.left_click(x, y)
            elif "WAIT" in inst:
                ls = inst.split(" ")
                t = float(ls[1])
                time.sleep(t)
            elif "OPEN_TERMINAL" in inst:
                pyautogui.keyDown('command')
                pyautogui.press('space')
                pyautogui.keyUp('command')
                time.sleep(1.0)
                self.keyboard_write("terminal")
                time.sleep(1.0)
                pyautogui.press('enter')
                time.sleep(1.0)
            elif "ANALYSIS" in inst:
                await self.analysis(transcript)
            elif "RET" in inst:
                await self.ret(transcript)
            elif "SCREENSHOT" in inst:
                self.screenshot()
            elif "NO-OP" in inst:
                pass
            else:
                self.shell(inst)

    def shell(self, command: Union[str, List[str]]) -> None:
        if isinstance(command, str):
            command = command.split(" ")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)

    def generate_hash(self, n: int) -> str:
        return "".join(choice(ascii_uppercase) for _ in range(n))

    def screenshot(self, name: str="") -> None:
        if len(name) == 0:
            name = self.generate_hash(self.screenshot_name_length)
        screenshot = pyautogui.screenshot()
        screenshot.save(f"{name}.png")
        self.screenshot_name_history.append(name)
        logger.warn(f"Screenshot name = {name}")

    async def handle_realtime_connection(self) -> None:
        async with self.client.beta.realtime.connect(
            model=self.REALTIME_MODEL,
        ) as conn:
            self.connection = conn
            self.connected.set()

            # note: this is the default and can be omitted
            # if you want to manually handle VAD yourself, then set `'turn_detection': None`
            await conn.session.update(session={
                "turn_detection": {"type": "server_vad", "threshold": 0.8, "silence_duration_ms": 1000},
                "instructions": "You are an operating system named Berk, helping a user named Jay. Be very less verbose. Do not list stuff. If Jay asks you to perform operations relating to software or web search (such as sports), answer positively that you are initiating the commands.",
                "input_audio_transcription": {
                    "model":"whisper-1",
                    "language":"en"
                },
                "voice":"sage"
            })

            acc_items: dict[str, Any] = {}

            async for event in conn:
                if event.type == "session.created":
                    self.session = event.session
                    session_display = self.query_one(SessionDisplay)
                    assert event.session.id is not None
                    session_display.session_id = event.session.id
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    continue

                if event.type == "response.audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue

                # THIS APPROACH FAILED
                # if event.type == "conversation.item.created":
                #     logger.debug(f"conversation.item.created WUT {event.item}")
                #     logger.debug(f"conversation.item.created HUT {event.item.content}")
                #     continues

                if event.type == "conversation.item.input_audio_transcription.completed":
                    logger.warning(f"!!! User transcript: {event.transcript}")
                    self.processor_queue.put(event.transcript)
                    # await self.assemble_and_run(event.transcript)
                    continue

                if event.type == "response.audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta

                    # Clear and update the entire content because RichLog otherwise treats each delta as a new line
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.clear()
                    bottom_pane.write(acc_items[event.item_id])
                    continue

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def processor(self) -> None:
        while True:
            if self.processor_queue.empty(): # no task
                await asyncio.sleep(0)
                continue
            else:
                transcript = self.processor_queue.get()
                logger.warning(f"processing transcript {transcript}")
                await self.assemble_and_run(transcript)
    
    async def send_mic_audio(self) -> None:
        import sounddevice as sd  # type: ignore

        sent_audio = False

        device_info = sd.query_devices()
        print(device_info)

        read_size = int(SAMPLE_RATE * 0.02)

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
        )
        stream.start()

        status_indicator = self.query_one(AudioStatusIndicator)

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                await self.should_send_audio.wait()
                status_indicator.is_recording = True

                data, _ = stream.read(read_size)

                connection = await self._get_connection()
                if not sent_audio:
                    asyncio.create_task(connection.send({"type": "response.cancel"}))
                    sent_audio = True

                await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))

                await asyncio.sleep(0)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "enter":
            self.query_one(Button).press()
            return

        if event.key == "q":
            self.exit()
            return

        if event.key == "k":
            status_indicator = self.query_one(AudioStatusIndicator)
            if status_indicator.is_recording:
                self.should_send_audio.clear()
                status_indicator.is_recording = False

                if self.session and self.session.turn_detection is None:
                    # The default in the API is that the model will automatically detect when the user has
                    # stopped talking and then start responding itself.
                    #
                    # However if we're in manual `turn_detection` mode then we need to
                    # manually tell the model to commit the audio buffer and start responding.
                    conn = await self._get_connection()
                    await conn.input_audio_buffer.commit()
                    await conn.response.create()
            else:
                self.should_send_audio.set()
                status_indicator.is_recording = True

    def encode_image(self, image_path):
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    async def analysis(self, transcript: str) -> None:
        image_path = f"{self.screenshot_name_history[-1]}.png"
        base64_image = self.encode_image(image_path)
        response = await self.client.chat.completions.create(
            model=self.GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": transcript + " Use the image below for guidance.", # "What is the score between Man City and Real Madrid?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        )
        self.a0 = response.choices[0].message.content
        # TODO add in Pixstral as another layer of analysis
        logging.getLogger(__name__).warning(f"analysis returns = {self.a0}")
        return

    async def ret(self, transcript: str) -> None:
        # TODO can also use whisper or tacotron or nvidia
        logging.getLogger(__name__).warning(f"in ret = {self.a0}")

        connection = await self._get_connection()
        asyncio.create_task(connection.send({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": f"""
                    You are an operating system named Berk, helping a user named Jay. Be very less verbose. Do not list stuff. Jay previously asked for "{transcript}". The search result is "{self.a0}". Please summarise for the result for Jay.
                """,
                "voice": "sage",
                "output_audio_format": "pcm16",
            }
        }))
        await asyncio.sleep(0)
        return

    def left_click(self, x=None, y=None) -> None:
        if not (x is None and y is None):
            pyautogui.moveTo(x, y)
        pyautogui.click()

    def left_double_click(self) -> None:
        pyautogui.doubleClick()

    def keyboard_write(self, text: Union[str, List[str]]) -> None:
        pyautogui.typewrite(text)

    def right_click(self) -> None:
        pyautogui.click(button="right")

if __name__ == "__main__":
    app = RealtimeApp()
    app.run()
