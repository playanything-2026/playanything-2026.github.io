from typing import Dict, Any
from copy import deepcopy
import json

from cradle import constants
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import BaseProvider
from cradle.utils.json_utils import parse_semi_formatted_text

from cradle.provider.module.task_inference import TaskInferenceProvider as ModuleTaskInferenceProvider

config = Config()
logger = Logger()
memory = LocalMemory()

class TaskInferencePreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 use_screenshot_augmented=False,
                 use_video=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm
        self.use_screenshot_augmented = use_screenshot_augmented
        self.use_video = use_video
        self.memory = LocalMemory()  # 统一：在 init 中实例化

    def __call__(self):

        # 1. 准备图片
        if not self.use_video:
            screenshot_path = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET)[-1]
            try:
                screenshot_augmnented_path = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET)[-1]
            except:
                screenshot_augmnented_path = screenshot_path

            if not self.use_screenshot_augmented:
                image_introduction = [
                    {"introduction": "This screenshot is the current step of the game.", "path": screenshot_path,
                     "assistant": ""}]
            else:
                image_introduction = [{"introduction": "This screenshot is the current step of the game.",
                                       "path": screenshot_augmnented_path, "assistant": ""}]

            processed_params = {"image_introduction": image_introduction}

        else:
            images = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, config.event_count)
            reasonings = self.memory.get_recent_history('decision_making_reasoning', config.event_count)
            image_introduction = [{"path": images[i], "assistant": "", "introduction": f"Frame {i}: {reasonings[i]}"}
                                  for i in range(config.event_count)]
            processed_params = {"image_introduction": image_introduction, "event_count": config.event_count}

        # 2. 提取文本参数 (增加 None 保护)
        task_description = self.memory.get_recent_history("task_description", k=1)[0]
        if not task_description: task_description = "Complete the level."

        subtask_description = self.memory.get_recent_history("subtask_description", k=1)[0]
        if not subtask_description: subtask_description = "None (Initial Step)"

        success = self.memory.get_recent_history("success", k=1)[0]
        if success is None: success = "True"

        self_reflection_reasoning = self.memory.get_recent_history("self_reflection_reasoning", k=1)[0]
        if not self_reflection_reasoning: self_reflection_reasoning = "None"

        game_state = self.memory.get_recent_history("description", k=1)[0]
        if not game_state:
            game_state = self.memory.get_recent_history("screen_classification", k=1)[0]
        if not game_state: game_state = "Game start, no state yet."

        previous_summarization = self.memory.get_recent_history("summarization", k=1)[0]
        if not previous_summarization: previous_summarization = "None"

        processed_params.update({
            "task_description": task_description,
            "subtask_description": subtask_description,
            "success": success,
            "self_reflection_reasoning": self_reflection_reasoning,
            "game_state_information": game_state,
            "previous_summarization": previous_summarization
        })

        self.memory.working_area.update(processed_params)
        return processed_params


class TaskInferenceProvider(ModuleTaskInferenceProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = memory

    def __call__(self, *args, **kwargs):

        params = deepcopy(self.memory.working_area)

        # 1. 组装 Prompt
        message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=params)

        response = {}
        try:
            response, info = self.llm_provider.create_completion(message_prompts)
            response = parse_semi_formatted_text(response)
        except Exception as e:
            logger.error(f"Response error: {e}")
            return {}

        # 2. 🔥 强制修复 Key：LLM 有时会返回 'Subtask' 或漏掉
        if 'subtask' not in response:
            if 'Subtask' in response:
                response['subtask'] = response['Subtask']
            elif 'subtask_description' in response:
                response['subtask'] = response['subtask_description']
            else:
                # 实在没有就给个默认值，防止程序崩掉
                response['subtask'] = "Move forward to explore."

        return response

class TaskInferencePostprocessProvider(BaseProvider):

    def __init__(self, *args, use_subtask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_subtask = use_subtask
        self.memory = LocalMemory()  # 统一：在 init 中实例化

    def __call__(self, response: Dict):
        logger.write(f"LLM Response (Task Inference) Type: {type(response)}")
        logger.write(f"LLM Response (Task Inference) Content: {response}")

        processed_response = deepcopy(response)

        # 健壮的 Key 提取
        subtask_description = "None"
        if "subtask" in response:
            subtask_description = response["subtask"]
        elif "Subtask" in response:
            subtask_description = response["Subtask"]

        # 显式写入 subtask_description
        processed_response["subtask_description"] = subtask_description

        # 清理不需要的 key
        if not self.use_subtask and "subtask" in processed_response:
            processed_response.pop("subtask")

        # 写入内存
        self.memory.update_info_history(processed_response)

        logger.write(f"TaskInferencePostprocess saved subtask_description: {subtask_description}")

        return processed_response

class RDR2TaskInferencePreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm

    def __call__(self):

        logger.write(f'RDR2 Task Inference Preprocess')

        task_description = memory.get_recent_history("task_description", k=1)[0]
        screenshot_path = memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=1)[0]

        processed_params = {
            "task_description": task_description,
            constants.IMAGES_MEM_BUCKET: screenshot_path
        }

        # Information summary preparation
        if len(memory.get_recent_history("decision_making_reasoning",
                                         memory.max_recent_steps)) == memory.max_recent_steps:
            logger.write(f'> Information summary call...')

            images = memory.get_recent_history(constants.IMAGES_MEM_BUCKET, config.event_count)
            reasonings = memory.get_recent_history('decision_making_reasoning', config.event_count)

            image_introduction = [
                {
                    "path": images[event_i], "assistant": "",
                    "introduction": 'This is the {} screenshot of recent events. The description of this image: {}'.format(
                        ['first', 'second', 'third', 'fourth', 'fifth'][event_i], reasonings[event_i])
                } for event_i in range(config.event_count)
            ]

            previous_summarization = memory.get_summarization()
            event_count = str(config.event_count)

            processed_params.update({
                "image_introduction": image_introduction,
                "previous_summarization": previous_summarization,
                "event_count": event_count
            })

        memory.working_area.update(processed_params)

        return processed_params

class RDR2TaskInferencePostprocessProvider(BaseProvider):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

        logger.write(f'RDR2 Task Inference Postprocess')

        processed_response = deepcopy(response)

        if "info_summary" not in response:
            response["info_summary"] = ""

        info_summary = response["info_summary"]

        processed_response.update({
            "summarization": info_summary
        })

        memory.update_info_history(processed_response)

        return processed_response


class StardewTaskInferencePreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm

    def __call__(self):

        logger.write(f'Stardew Task Inference Preprocess')

        prompts = [
            "This screenshot is the current step of the game. The blue band represents the left side and the yellow band represents the right side."
        ]

        task_description = memory.get_recent_history("task_description", k=1)[0]
        previous_summarization = memory.get_recent_history("summarization", 1)[0]
        substask_description = memory.get_recent_history("subtask_description", 1)[0]
        substask_reasoning = memory.get_recent_history("subtask_reasoning", 1)[0]
        toolbar_information = memory.get_recent_history("toolbar_information", 1)[0]
        images = memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, 1)
        decision_making_reasoning = memory.get_recent_history('decision_making_reasoning', 1)
        self_reflection_reasoning = memory.get_recent_history('self_reflection_reasoning', 1)

        image_introduction = []
        image_introduction.append(
            {
                "introduction": prompts[-1],
                "path": images,
                "assistant": ""
            })

        processed_params = {
            "image_introduction": image_introduction,
            "previous_summarization": previous_summarization,
            "task_description": task_description,
            "subtask_description": substask_description,
            "subtask_reasoning": substask_reasoning,
            "previous_reasoning": decision_making_reasoning,
            "self_reflection_reasoning": self_reflection_reasoning,
            "toolbar_information": toolbar_information
        }

        memory.working_area.update(processed_params)

        return processed_params


class StardewTaskInferencePostprocessProvider(BaseProvider):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

        logger.write(f'Stardew Task Inference Postprocess')

        processed_response = deepcopy(response)

        history_summary = response['history_summary']

        subtask_description = response['subtask']
        subtask_reasoning = response['subtask_reasoning']

        processed_response.update({
            'summarization': history_summary,
            'subtask_description': subtask_description,
            'subtask_reasoning': subtask_reasoning
        })

        memory.update_info_history(processed_response)

        return processed_response