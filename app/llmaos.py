import pyautogui
from random import choice
from string import ascii_uppercase
from typing import Union, List
import time

class LLMAOS():
    def __init__(self) -> "LLMAOS":
        self.screenshot_name_history = []
        self.screenshot_name_length = 20

    def generate_hash(self, n:int) -> str:
        return "".join(choice(ascii_uppercase) for _ in range(n))

    def screenshot(self, name:str="") -> None:
        if len(name) == 0:
            name = self.generate_hash(self.screenshot_name_length)
        screenshot = pyautogui.screenshot()
        screenshot.save(f"{name}.png")
        self.screenshot_name_history.append(name)

    def attend_to(self, name:str="") -> None:
        if len(name) == 0:
            name = self.screenshot_name_history[-1]
        # TODO DALLE
        self.dalle.game

    def left_click(self) -> None:
        pyautogui.click()

    def left_double_click(self) -> None:
        pyautogui.doubleClick()

    def keyboard_write(self, text:Union[str, List[str]]) -> None:
        pyautogui.typewrite(text)

    def right_click(self) -> None:
        pyautogui.click(button="right")
    


if __name__ == "__main__":
    llmaos = LLMAOS()
    print("Starting operating system...")
    llmaos.screenshot()
    llmaos.keyboard_write("hello, testing the keyboard")
