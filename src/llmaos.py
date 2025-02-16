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
from mistralai import Mistral

class LLMAOS():

    def __init__(self, client, network, neva_flag=True, pixtral_flag=True):
        self.KEYWORDS = ["<enter>"]
        self.GPT_MODEL = "gpt-4o-mini"
        self.PIXTRAL_MODEL = "pixtral-12b-2409"
        self.a0 = None # accumulator, stores return value
        self.t0 = None # task instruction
        self.screenshot_name_history = []
        self.screenshot_name_length = 20
        self.client = client
        self.network = network
        self.neva_flag = neva_flag
        self.pixtral_flag = pixtral_flag
        pass

    async def execute(self, instructions: list[str], transcript: str) -> None:
        for _inst in instructions:
            inst = _inst.strip("'").strip('"').strip()
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
                self.screenshot()
                await self.analysis_powerful(transcript)   
            elif "NO-OP" in inst:
                pass
            else:
                self.shell(inst)

    def shell(self, command: Union[str, list[str]]) -> None:
        if isinstance(command, str):
            command = command.strip("'").strip('"').strip()
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
                        2. LEFT_CLICK x y (x from 0=left to 1792=right, y from 0=up to 1120=down are integers): left clicks on (x, y) on the 1792 x 1120 screen.
                        3. OPEN_TERMINAL: opens up a terminal
                        4. ANALYSIS: takes a screenshot and send it to API endpoints for analysis
                        5. NO-OP: does nothing, used when Jay says something non-task related
                        6. KEYBOARD x (x is some string in single quotes or a special keyword like <enter>): inputs x on the keyboard
                        7. WAIT x (x is integer): waits x seconds, for a website to load for example
                    """
                }, {
                    "role": "developer", 
                    "content": """
                        For example, if the user asks for the score of football team X versus football team Y, output the following (do not change the integers):
                        "open https://www.google.com/"
                        "LEFT_CLICK 605 561"
                        "KEYBOARD 'X vs Y'"
                        "KEYBOARD <enter>"
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
                        "WAIT 6"
                        "LEFT_CLICK 1245 797"
                        "LEFT_CLICK 1293 814"
                    """
                }, {
                    "role": "developer",
                    "content": """
                        For example, if the user says press enter, output the following:
                        "KEYBOARD <enter>"
                    """
                }, {
                    "role": "developer",
                    "content": """
                        For example, if the user says type x:
                        "KEYBOARD 'x'"
                    """
                }, {
                    "role": "developer",
                    "content": """
                        For example, if the user says click on X, output the following, with integers x, y being your best guess of where to click:
                        "ANALYSIS",
                        "LEFT_CLICK x y"
                    """
                }, {
                    "role": "developer",
                    "content": """
                        For example, if the user only says hello, you should just output:
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

    async def mistralai_pixtral(self, transcript) -> str:
        filename = self.screenshot_name_history[-1]
        image_path = f"{filename}.png"
        base64_image = self.encode_image(image_path)
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The user recently asked {transcript}. Can you answer that using the image below?"
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}" 
                    }
                ]
            }
        ]

        chat_response = client.chat.complete(
            model=self.PIXTRAL_MODEL,
            messages=messages
        )

        answer = chat_response.choices[0].message.content
        logging.getLogger(__name__).warning(f"answer from pixtral = {answer}")
        return answer

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

        args = [
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
            )
        ]
        if self.neva_flag: args.append(self.nvidia_neva(transcript))
        if self.pixtral_flag: args.append(self.mistralai_pixtral(transcript))

        resp_all = await asyncio.gather(*args)

        resp_all = list(resp_all)
        resp_all[0] = resp_all[0].choices[0].message.content # 0th argument is always gpt
        self.a0 = " ".join(resp_all) # simple concatenation

        logging.getLogger(__name__).warning(f"final answer = {self.a0}")
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

    async def analysis_simple(self, transcript: str) -> None:
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
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: The file {image_path} was not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None