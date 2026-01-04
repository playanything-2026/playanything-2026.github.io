import time

from PIL import Image
import mss

from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle import constants
from cradle.environment import UIControl
from cradle.utils.singleton import Singleton


config = Config()
logger = Logger()
io_env = IOEnvironment()

class MiniGameUIControl(UIControl, metaclass=Singleton):
    """
    《推箱子小游戏》的 UI 控制类。
    **【实时无暂停模式】**：此模式保持游戏持续运行，适用于回合制或实时动作游戏。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def pause_game(self, env_name: str, ide_name: str) -> None:
        """
        【实时无暂停】此函数为空操作，避免游戏中断和延迟。
        """
        logger.debug("Running in real-time, no-pause mode. pause_game() is a no-op.")
        # 避免窗口切换导致的延迟和干扰
        # if ide_name:
        #     ide_window = io_env.get_windows_by_name(ide_name)[0]
        #     ide_window.activate()
        #     ide_window.show()
        # time.sleep(0.01) # 仅保留极短的等待
        pass


    def unpause_game(self, env_name: str, ide_name: str) -> None:
        """
        【实时无暂停】此函数仅确保游戏窗口处于活动状态。
        """
        self.switch_to_game(env_name, ide_name)
        logger.debug("Running in real-time, no-pause mode. unpause_game() is only switch_to_game.")


    def switch_to_game(self, env_name: str, ide_name: str) -> None:
        """
        切换到游戏窗口。
        """
        target_window = io_env.get_windows_by_name(config.env_name)[0]
        try:
            target_window.activate()
        except Exception as e:
            if "Error code from Windows: 0" in str(e):
                pass
            else:
                raise e
        time.sleep(0.1) # 极短延迟以确保窗口切换完成


    def exit_back_to_pause(self, env_name: str, ide_name: str) -> None:
        """
        【实时无暂停】此函数仅调用 pause_game()，即空操作。
        """
        self.pause_game(env_name, ide_name)


    def exit_back_to_game(self, env_name: str, ide_name: str) -> None:
        """
        【实时无暂停】此函数仅确保游戏窗口激活。
        """
        self.unpause_game(env_name, ide_name)


    def is_env_paused(self) -> bool:
        """
        【实时无暂停】强制返回 False，告知 runner 游戏始终在运行。
        """
        return False


    def take_screenshot(self,
                        tid: float,
                        screen_region: tuple[int, int, int, int] = None) -> str:
        """
        截取屏幕快照。逻辑与 Cities: Skylines 保持一致。
        """
        if screen_region is None:
            screen_region = config.env_region

        region = screen_region
        region = {
            "left": region[0],
            "top": region[1],
            "width": region[2],
            "height": region[3],
        }

        output_dir = config.work_dir

        # Save screenshots
        screen_image_filename = output_dir + "/screen_" + str(tid) + ".jpg"

        with mss.mss() as sct:
            screen_image = sct.grab(region)
            image = Image.frombytes("RGB", screen_image.size, screen_image.bgra, "raw", "BGRX")
            image.save(screen_image_filename)

        return screen_image_filename