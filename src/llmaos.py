import pyautogui
import time
import subprocess
import base64
import asyncio
import os
from typing import Union
import requests
from random import choice
from string import ascii_uppercase
import logging

class LLMAOS():

    def __init__(self, client, network):
        self.KEYWORDS = ["<enter>"]
        self.GPT_MODEL = "gpt-4o-mini"
        self.a0 = None # accumulator, stores return value
        self.t0 = None # task instruction
        self.screenshot_name_history = []
        self.screenshot_name_length = 20
        self.client = client
        self.network = network
        pass

    async def execute(self, instructions: list[str], transcript: str) -> None:
        for inst in instructions:
            time.sleep(1.5)
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
                await self.analysis_powerful(transcript)
            elif "SCREENSHOT" in inst:
                self.screenshot()
            elif "NO-OP" in inst:
                pass
            else:
                self.shell(inst)

    def shell(self, command: Union[str, list[str]]) -> None:
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
        logging.getLogger(__name__).warn(f"Screenshot name = {name}")

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
                        8. NO-OP
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

        logging.getLogger(__name__).warning(f"!!! codegen_response = {response}")
        resp = response.choices[0].message.content

        # post processing
        resp = resp.split("\n")
        instructions = [s for s in resp if len(s) > 0] # filter out empty strings
        logging.getLogger(__name__).warning(f"!!! processed response = {instructions}")
        
        return instructions


    ### CODE GEN PORTION

    async def nvidia_neva(self, transcript) -> str:
        invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
        stream = False

        from PIL import Image
        filename = self.screenshot_name_history[-1]
        image = Image.open(f"{filename}.png")
        new_size = (image.width // 6, image.height // 6)
        resized_image = image.resize(new_size, Image.LANCZOS)
        resized_image.save(f"{filename}_rescaled.png")

        with open(f"{filename}_rescaled.png", "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        assert len(image_b64) < 180_000, \
            f"The image size is {len(image_b64)}. To upload larger images, use the assets API (see docs)"

        headers = {
            "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
            "Accept": "text/event-stream" if stream else "application/json"
        }

        payload = {
            "messages": [
                {
                "role": "user",
                "content": f'Answer "{transcript}" using specific parts of the image as evidence. <img src="data:image/jpg;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.20,
            "top_p": 0.70,
            "seed": 0,
            "stream": stream
        }

        response = requests.post(invoke_url, headers=headers, json=payload)
        answer = response.json()["choices"][0]["message"]["content"]
        logging.getLogger(__name__).warning(f"answer from nvidia = {answer}")
        return answer

    async def analysis_powerful(self, transcript: str) -> None:
        image_path = f"{self.screenshot_name_history[-1]}.png"
        base64_image = self.encode_image(image_path)

        resp1, resp2 = await asyncio.gather(
            self.client.chat.completions.create(
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
            ),
            self.nvidia_neva(transcript)
        )

        resp1 = resp1.choices[0].message.content

        self.a0 = resp1 + resp2 # simple concatenation
        logging.getLogger(__name__).warning(f"final answer = {self.a0}")
        # TODO add in Pixstral as another layer of analysis
        logging.getLogger(__name__).warning(f"analysis returns = {self.a0}")

        connection = await self.network._get_connection()
        asyncio.create_task(connection.send({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": f"""
                    You are an operating system named Berk, helping a user named Jay. Be very less verbose. Do not list stuff. Jay previously asked for "{transcript}". The search result is "{self.a0}". Please summarise for the result for Jay, exclude any negative response like "I can't access..." or "You can do this by ...".
                """,
                "voice": "sage",
                "output_audio_format": "pcm16",
            }
        }))
        await asyncio.sleep(0)
        return

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

    ### DEVICE I/O PORTION

    def left_click(self, x=None, y=None) -> None:
        if not (x is None and y is None):
            pyautogui.moveTo(x, y)
        pyautogui.click()

    def left_double_click(self) -> None:
        pyautogui.doubleClick()

    def keyboard_write(self, text: Union[str, list[str]]) -> None:
        pyautogui.typewrite(text)

    def right_click(self) -> None:
        pyautogui.click(button="right")

    ### HELPER

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")