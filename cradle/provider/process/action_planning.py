from typing import Dict, Any, List, Union
import json
import re
from copy import deepcopy

from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import BaseProvider
from cradle import constants
from cradle.utils.json_utils import parse_semi_formatted_text

# 导入 Module 层的原始类
try:
    from cradle.provider.module.actionplanning import ActionPlanningProvider as ModuleActionPlanningProvider
except ImportError:
    from cradle.provider.module.action_planning import ActionPlanningProvider as ModuleActionPlanningProvider

config = Config()
logger = Logger()
memory = LocalMemory()


class ActionPlanningPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 use_screenshot_augmented=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm
        self.use_screenshot_augmented = use_screenshot_augmented

    def __call__(self):

        prompts = [
            "This screenshot is the previous step of the game.",
            "This screenshot is the current step of the game."
        ]

        # 1. Image Processing
        screenshot_paths = memory.get_recent_history("screenshot_path", k=config.action_planning_image_num)
        screenshot_augmnented_paths = memory.get_recent_history("screenshot_augmented_path",
                                                                k=config.action_planning_image_num)

        # 简单容错处理
        if not screenshot_paths: screenshot_paths = []
        if not screenshot_augmnented_paths: screenshot_augmnented_paths = []

        if not self.use_screenshot_augmented:
            image_introduction = []
            for i in range(len(screenshot_paths), 0, -1):
                image_introduction.append({
                    "introduction": prompts[-i],
                    "path": screenshot_paths[-i],
                    "assistant": ""
                })
        else:
            image_introduction = []
            for i in range(len(screenshot_augmnented_paths), 0, -1):
                image_introduction.append({
                    "introduction": prompts[-i],
                    "path": screenshot_augmnented_paths[-i],
                    "assistant": ""
                })

        # 2. Extract Text Variables
        subtask_description = memory.get_recent_history("subtask_description", k=1)[0]
        if not subtask_description: subtask_description = "None"

        game_state = memory.get_recent_history("description", k=1)[0]
        if not game_state: game_state = memory.get_recent_history("screen_classification", k=1)[0]
        if not game_state: game_state = "Game start, no detailed analysis available yet."

        success = memory.get_recent_history("success", k=1)[0]
        if success is None: success = "True"

        actions = memory.get_recent_history("actions", k=1)[0]
        if not actions: actions = "None"

        skill_library = memory.get_recent_history("skill_library", k=1)[0]
        if not skill_library: skill_library = "move_left(), move_right(), jump_left(), jump_right(), move_up(), move_down()"

        # 3. Update Working Area (Aliases included)
        processed_params = {
            "image_introduction": image_introduction,
            "subtask_description": subtask_description,
            "subtask": subtask_description,  # Alias
            "task": subtask_description,  # Alias
            "game_state_information": game_state,
            "game_state": game_state,  # Alias
            "description": game_state,  # Alias
            "success": success,
            "actions": actions,
            "skill_library": skill_library
        }

        memory.working_area.update(processed_params)
        memory.update_info_history(processed_params)

        return processed_params


class ActionPlanningProvider(ModuleActionPlanningProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = memory

    def __call__(self, *args, **kwargs):

        # 1. 获取参数
        params = deepcopy(self.memory.working_area)

        # 组装 Prompt
        message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=params)

        # 手动替换占位符 (Patch)
        def manual_replace(text_content, param_dict):
            if not isinstance(text_content, str): return text_content
            result = text_content
            for k, v in param_dict.items():
                if isinstance(v, str):
                    placeholder = f"<${k}$>"
                    if placeholder in result:
                        result = result.replace(placeholder, v)
            return result

        if isinstance(message_prompts, str):
            message_prompts = manual_replace(message_prompts, params)
        elif isinstance(message_prompts, list):
            for msg in message_prompts:
                if isinstance(msg, dict) and 'content' in msg:
                    content = msg['content']
                    if isinstance(content, str):
                        msg['content'] = manual_replace(content, params)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                part['text'] = manual_replace(part['text'], params)

        # 2. 调用 LLM
        response = {}
        try:
            raw_response_text, info = self.llm_provider.create_completion(message_prompts)
            logger.write(f"Raw LLM Response Text: {raw_response_text}")

            # 3. 尝试使用系统解析
            response = parse_semi_formatted_text(raw_response_text)

            # 🔥🔥🔥 补丁：如果系统解析失败 (返回 None key)，手动解析 🔥🔥🔥
            # 你的日志显示 response 是 {None: 'Reasoning: ...'}，这说明解析器崩了
            if 'reasoning' not in response and (None in response or not response):

                # 获取原始文本
                text = raw_response_text
                if isinstance(text, dict) and None in text:
                    text = text[None]  # 提取出那段话

                if isinstance(text, str):
                    # 手动正则提取
                    # 提取 Reasoning
                    reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=Actions:|$)', text, re.DOTALL | re.IGNORECASE)
                    if reasoning_match:
                        response['reasoning'] = reasoning_match.group(1).strip()

                    # 提取 Actions
                    actions_match = re.search(r'Actions:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
                    if actions_match:
                        act_str = actions_match.group(1).strip()
                        response['actions'] = [act_str]  # 封装成列表

        except Exception as e:
            logger.error(f"Response error: {e}, retrying...")

        self._check_output_keys(response)
        return response


class ActionPlanningPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = memory

    def __call__(self, response: Dict):
        logger.write(f"LLM Response (Action Planning) Content: {response}")

        # 🔥🔥🔥 修复 NameError：先定义 error_msg 🔥🔥🔥

        # 1. 严格校验 Reasoning (必须存在)
        if 'reasoning' not in response:
            error_msg = f"❌ CRITICAL ERROR: LLM did not return 'reasoning' key. Response keys: {list(response.keys())}"
            logger.error(error_msg)
            # 这里的 error_msg 现在肯定定义了
            raise ValueError(error_msg)

        # 2. 宽容处理 Actions
        skill_steps = []
        if 'actions' in response:
            skill_steps = response['actions']
        else:
            logger.warn("⚠️ Warning: 'actions' key missing. Proceeding with NO ACTION.")
            # skill_steps 保持为空列表

        processed_response = deepcopy(response)

        # 3. 字符串解析
        if isinstance(skill_steps, str):
            clean_str = skill_steps.strip("[]'\" ")
            if clean_str:
                if ',' in clean_str:
                    skill_steps = [s.strip(" '\"") for s in clean_str.split(',')]
                else:
                    skill_steps = [clean_str]
            else:
                skill_steps = []

        # 4. 空值检查 (允许为空，仅打印日志)
        if not skill_steps or skill_steps == ['']:
            skill_steps = []

        # 截取逻辑
        skill_steps = skill_steps[:config.number_of_execute_skills]

        if config.number_of_execute_skills > 1:
            actions = "[" + ",".join(str(s) for s in skill_steps) + "]"
        else:
            actions = str(skill_steps[0]) if len(skill_steps) > 0 else ""

        decision_making_reasoning = response['reasoning']

        processed_response.update({
            "actions": actions,
            "decision_making_reasoning": decision_making_reasoning,
            "skill_steps": skill_steps,
        })

        memory.update_info_history(processed_response)

        return processed_response


# ... (下面的 RDR2 和 Stardew 类保持不变，如果有的话) ...
# 为了确保文件完整性，如果你下面还有其他类，请保留它们。
# 如果没有，这就足够了。


class RDR2ActionPlanningPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm

    def __call__(self):

        logger.write("RDR2 Action Planning Preprocess")

        prompts = [
            "Now, I will give you five screenshots for decision making.",
            "This screenshot is five steps before the current step of the game",
            "This screenshot is three steps before the current step of the game",
            "This screenshot is two steps before the current step of the game",
            "This screenshot is the previous step of the game",
            "This screenshot is the current step of the game"
        ]

        response_keys = memory.get_recent_history("response_keys", k=1)[0]
        response = memory.get_recent_history("response", k=1)[0]
        pre_action = memory.get_recent_history("pre_action", k=1)[0]
        pre_self_reflection_reasoning = memory.get_recent_history("pre_self_reflection_reasoning", k=1)[0]
        pre_screen_classification = memory.get_recent_history("pre_screen_classification", k=1)[0]
        screen_classification = memory.get_recent_history("screen_classification", k=1)[0]
        skill_library = memory.get_recent_history("skill_library", k=1)[0]
        task_description = memory.get_recent_history("task_description", k=1)[0]

        previous_action = ""
        previous_reasoning = ""
        if pre_action:
            previous_action = memory.get_recent_history("action", k=1)[0]
            previous_reasoning = memory.get_recent_history("decision_making_reasoning", k=1)[0]

        previous_self_reflection_reasoning = ""
        if pre_self_reflection_reasoning:
            previous_self_reflection_reasoning = memory.get_recent_history("self_reflection_reasoning", k=1)[0]

        info_summary = memory.get_recent_history("summarization", k=1)[0]

        # @TODO Temporary solution with fake augmented entries if no bounding box exists. Ideally it should read images, then check for possible augmentation.
        image_memory = memory.get_recent_history("screenshot_path", k=config.action_planning_image_num)
        augmented_image_memory = memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET,
                                                           k=config.action_planning_image_num)

        image_introduction = []
        for i in range(len(image_memory), 0, -1):
            if len(augmented_image_memory) >= i and augmented_image_memory[-i] != constants.NO_IMAGE:
                if i == len(image_memory):
                    image_introduction.append(
                        {
                            "introduction": prompts[-i],
                            "path": augmented_image_memory[-i],
                            "assistant": "",
                            "resolution": "high",
                        })
                else:
                    image_introduction.append(
                        {
                            "introduction": prompts[-i],
                            "path": augmented_image_memory[-i],
                            "assistant": "",
                        })
            else:
                image_introduction.append(
                    {
                        "introduction": prompts[-i],
                        "path": image_memory[-i],
                        "assistant": ""
                    })

        # Minimap info tracking
        minimap_information = ""
        if constants.MINIMAP_INFORMATION in response_keys:
            minimap_information = response[constants.MINIMAP_INFORMATION]
            logger.write(f"{constants.MINIMAP_INFORMATION}: {minimap_information}")

            minimap_info_str = ""
            for key, value in minimap_information.items():
                if value:
                    for index, item in enumerate(value):
                        minimap_info_str = minimap_info_str + key + ' ' + str(index) + ': angle ' + str(
                            int(item['theta'])) + ' degree' + '\n'
            minimap_info_str = minimap_info_str.rstrip('\n')

            logger.write(f'minimap_info_str: {minimap_info_str}')
            minimap_information = minimap_info_str

        processed_params = {
            "pre_screen_classification": pre_screen_classification,
            "screen_classification": screen_classification,
            "previous_action": previous_action,
            "previous_reasoning": previous_reasoning,
            "previous_self_reflection_reasoning": previous_self_reflection_reasoning,
            "skill_library": skill_library,
            "task_description": task_description,
            "minimap_information": minimap_information,
            "info_summary": info_summary,
            "image_introduction": image_introduction
        }

        memory.working_area.update(processed_params)

        return processed_params

class StardewActionPlanningPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 toolbar_information: str,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm
        self.toolbar_information = toolbar_information

    def __call__(self):

        logger.write("Stardew Action Planning Preprocess")

        prompts = [
            "Now, I will give you five screenshots for decision making."
            "This screenshot is five steps before the current step of the game",
            "This screenshot is three steps before the current step of the game",
            "This screenshot is two steps before the current step of the game",
            "This screenshot is the previous step of the game. The blue band represents the left side and the yellow band represents the right side.",
            "This screenshot is the current step of the game. The blue band represents the left side and the yellow band represents the right side."
        ]

        pre_action = memory.get_recent_history("pre_action", k=1)[0]
        pre_self_reflection_reasoning = memory.get_recent_history("pre_self_reflection_reasoning", k=1)[0]
        toolbar_information = memory.get_recent_history("toolbar_information", k=1)[0]
        selected_position = memory.get_recent_history("selected_position", k=1)[0]
        summarization = memory.get_recent_history("summarization", k=1)[0]
        skill_library = memory.get_recent_history("skill_library", k=1)[0]
        task_description = memory.get_recent_history("task_description", k=1)[0]
        subtask_description = memory.get_recent_history("subtask_description", k=1)[0]
        history_summary = memory.get_recent_history("summarization", k=1)[0]

        # Decision making preparation
        toolbar_information = toolbar_information if toolbar_information is not None else self.toolbar_information
        selected_position = selected_position if selected_position is not None else 1

        previous_action = ""
        previous_reasoning = ""
        if pre_action:
            previous_action = memory.get_recent_history("action", k=1)[0]
            previous_reasoning = memory.get_recent_history("decision_making_reasoning", k=1)[0]

        previous_self_reflection_reasoning = ""
        if pre_self_reflection_reasoning:
            previous_self_reflection_reasoning = memory.get_recent_history("self_reflection_reasoning", k=1)[0]

        # @TODO Temporary solution with fake augmented entries if no bounding box exists. Ideally it should read images, then check for possible augmentation.
        image_memory = memory.get_recent_history("augmented_image", k=config.action_planning_image_num)

        image_introduction = []
        for i in range(len(image_memory), 0, -1):
            image_introduction.append(
                {
                    "introduction": prompts[-i],
                    "path": image_memory[-i],
                    "assistant": ""
                })

        processed_params = {
            "pre_self_reflection_reasoning": pre_self_reflection_reasoning,
            "toolbar_information": toolbar_information,
            "selected_position": selected_position,
            "summarization": summarization,
            "skill_library": skill_library,
            "task_description": task_description,
            "subtask_description": subtask_description,
            "history_summary": history_summary,
            "previous_action": previous_action,
            "previous_reasoning": previous_reasoning,
            "previous_self_reflection_reasoning": previous_self_reflection_reasoning,
            "image_introduction": image_introduction
        }

        memory.working_area.update(processed_params)

        return processed_params

class RDR2ActionPlanningPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

        logger.write("RDR2 Action Planning Postprocess")

        processed_response = deepcopy(response)

        skill_steps = []
        if 'actions' in response:
            skill_steps = response['actions']

        if skill_steps:
            skill_steps = [i for i in skill_steps if i != '']
        else:
            skill_steps = ['']

        skill_steps = skill_steps[:config.number_of_execute_skills]

        if config.number_of_execute_skills > 1:
            actions = "[" + ",".join(skill_steps) + "]"
        else:
            actions = str(skill_steps[0])

        decision_making_reasoning = response['reasoning']
        pre_decision_making_reasoning = decision_making_reasoning

        processed_response.update({
            "action": actions,
            "pre_decision_making_reasoning": pre_decision_making_reasoning,
            "decision_making_reasoning": decision_making_reasoning,
            "skill_steps": skill_steps,
        })
        memory.update_info_history(processed_response)

        return processed_response


class StardewActionPlanningPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

        logger.write("Stardew Action Planning Postprocess")

        processed_response = deepcopy(response)

        skill_steps = []
        if 'actions' in response:
            skill_steps = response['actions']

        if skill_steps:
            skill_steps = [i for i in skill_steps if i != '']
        else:
            skill_steps = ['']

        skill_steps = skill_steps[:config.number_of_execute_skills]
        pre_action = "[" + ",".join(skill_steps) + "]"

        if config.number_of_execute_skills > 1:
            actions = "[" + ",".join(skill_steps) + "]"
        else:
            actions = str(skill_steps[0])

        decision_making_reasoning = response['reasoning']
        pre_decision_making_reasoning = decision_making_reasoning

        processed_response.update({
            "pre_action": pre_action,
            "action": actions,
            "pre_decision_making_reasoning": pre_decision_making_reasoning,
            "decision_making_reasoning": decision_making_reasoning,
            "skill_steps": skill_steps,
        })
        memory.update_info_history(processed_response)

        return processed_response