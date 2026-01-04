from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle.environment.minigame.skill_registry import register_skill
import time

config = Config()
logger = Logger()
io_env = IOEnvironment()



# --- 核心移动技能 ---

@register_skill("move_left")
def move_left():
    """
    模拟按下并释放左方向键（或游戏中的左移键），使角色向左移动。
    使用固定的内部持续时间。
    """
    duration = 0.5
    io_env.key_press("left", duration=duration)
    logger.debug(f"Action: move_left for {duration}s")


@register_skill("move_right")
def move_right():
    """
    模拟按下并释放右方向键（或游戏中的右移键），使角色向右移动。
    使用固定的内部持续时间。
    """
    duration = 0.8
    io_env.key_press("right", duration=duration)
    logger.debug(f"Action: move_right for {duration}s")


@register_skill("move_up")
def move_up():
    """
    模拟按下并释放上方向键（或游戏中的上移键），使角色向上移动。
    使用固定的内部持续时间。
    """
    duration = 0.5
    io_env.key_press("up", duration=duration)
    logger.debug(f"Action: move_up for {duration}s")


@register_skill("move_down")
def move_down():
    """
    模拟按下并释放下方向键（或游戏中的下移键），使角色向下移动。
    使用固定的内部持续时间。
    """
    duration = 0.5
    io_env.key_press("down", duration=duration)
    logger.debug(f"Action: move_down for {duration}s")


@register_skill("jump_left")
def jump_left():
    """
    模拟同时按下 left (左) 和 up (上) 键，实现向左跳跃。
    使用 io_env.key_press 的默认持续时间。
    """
    # 假设 io_env.key_press 支持列表以实现并发按键
    io_env.key_press(["left", "up"], duration=0.01)
    logger.debug(f"Action: jump_left (left+up)")


@register_skill("jump_right")
def jump_right():
    """
    模拟同时按下 right (右) 和 up (上) 键，实现向右跳跃。
    使用 io_env.key_press 的默认持续时间。
    """
    # 假设 io_env.key_press 支持列表以实现并发按键
    io_env.key_press(["right", "up"], duration=0.5)

    logger.debug(f"Action: jump_right (right+up)")


@register_skill("press_key")
def press_key(key_name: str):
    """
    The function to press a single key on the keyboard, typically used for game controls
    like A, B, C, or special keys like 'space', 'enter', 'shift'.

    Parameters:
     - key_name: The name of the key to be pressed (e.g., 'a', 'k', 'space', 'enter').
    """
    # 检查键名是否有效，防止 LLM 输出随机字符串
    valid_keys = 'abcdefghijklmnopqrstuvwxyz0123456789'
    key_name = key_name.lower().strip()

    if key_name in valid_keys or key_name in ['space', 'enter', 'shift', 'alt', 'ctrl']:
        # 使用 io_env.key_press() 执行底层的按键操作
        io_env.key_press(key_name)
    else:
        print(f"Warning: Attempted to press invalid key: {key_name}")


# --- 界面交互技能 ---

@register_skill("mouse_click")
def mouse_click(x: int, y: int):
    """
    将鼠标移动到指定的屏幕坐标 (x, y) 并执行单次左键点击。通常用于点击游戏开始按钮或菜单选项。

    Parameters:
     - x: 鼠标点击的 x坐标.
     - y: 鼠标点击的 y坐标.
    """
    button = "left"  # 固定使用左键点击
    io_env.mouse_move(x, y)
    io_env.mouse_click_button(button, clicks=1)
    logger.debug(f"Action: mouse_click at ({x}, {y}) with {button} button")
    time.sleep(0.5)


# 定义 __all__ 以便外部模块导入
__all__ = [
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "jump_left",
    "jump_right",
    "press_key",
    "mouse_click",
]