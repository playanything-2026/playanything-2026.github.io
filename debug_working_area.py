#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'cradle'))

from cradle.memory import LocalMemory
from cradle import constants

memory = LocalMemory()

# 模拟一些可能的数据
memory.working_area.update({
    "task_description": "Test task",
    "subtask_description": "Test subtask", 
    "subtask_reasoning": "Test reasoning",
    "image_introduction": [
        {
            "introduction": "This is a test",
            "path": "/path/to/image.jpg",
            "assistant": ""
        }
    ],
    "previous_summarization": "Previous summary",
    "floor_count": 10,  # 这可能是整数
    "character_health": 3,  # 这可能是整数
    "actions": "move_left(duration=0.05)",  # 字符串
    "self_reflection_reasoning": "Test reflection",
    "error_message": None,  # 这可能是 None
    "history_summary": "Test history",
    "some_dict": {"key": "value"},  # 这可能是字典
    "some_list": [1, 2, 3],  # 这可能是数字列表
})

print("Working area contents:")
for key, value in memory.working_area.items():
    print(f"  {key}: {type(value)} = {value}")

# 检查哪些值不是字符串或列表
print("\nNon-string, non-list values:")
for key, value in memory.working_area.items():
    if not isinstance(value, (str, list)):
        print(f"  {key}: {type(value)} = {value}")



