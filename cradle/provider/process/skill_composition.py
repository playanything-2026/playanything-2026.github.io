"""
Skill Composition Provider for AI-driven composite skill generation.

This module provides functionality to:
1. Analyze game state and action history
2. Generate composite skills by combining keyboard inputs
3. Validate generated skills against allowed keys (auto-extracted from basic skills)
"""

from copy import deepcopy
import re
from typing import Dict, List, Tuple

from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory

config = Config()
logger = Logger()


def extract_keys_from_skill_code(skill_code: str) -> List[str]:
    """从技能代码中提取 key_press 使用的按键"""
    keys = []
    # 匹配 key_press("key") 或 key_press(["key1", "key2"])
    key_pattern = r'key_press\s*\(\s*(?:\[([^\]]+)\]|["\']([^"\']+)["\'])'
    matches = re.findall(key_pattern, skill_code)

    for match in matches:
        if match[0]:  # 列表格式 ["key1", "key2"]
            found_keys = re.findall(r'["\']([^"\']+)["\']', match[0])
            keys.extend(found_keys)
        elif match[1]:  # 单个按键 "key"
            keys.append(match[1])

    return keys


class SkillCompositionPreprocessProvider(BaseProvider):
    """准备组合技能生成的上下文"""

    def __init__(self, gm, **kwargs):
        super().__init__(**kwargs)
        self.gm = gm
        self.memory = LocalMemory()

    def _extract_allowed_keys_from_basic_skills(self) -> List[str]:
        """从 skill_names_basic 中的技能代码自动提取允许的按键"""
        allowed_keys = set()
        basic_skill_names = config.skill_configs.get("skill_names_basic", [])

        for skill_name in basic_skill_names:
            skill_code, _ = self.gm.get_skill_library_in_code(skill_name)
            if skill_code:
                keys = extract_keys_from_skill_code(skill_code)
                allowed_keys.update(keys)

        return list(allowed_keys)

    @BaseProvider.debug
    @BaseProvider.write
    def __call__(self):
        params = deepcopy(self.memory.working_area)

        # 从基础技能代码中自动提取允许的按键（无需配置 allowed_keys）
        allowed_keys = self._extract_allowed_keys_from_basic_skills()
        params["allowed_keys"] = ", ".join(allowed_keys)
        # 保存到 working_area 供 postprocess 验证使用
        params["_extracted_allowed_keys"] = allowed_keys

        # 获取动作历史
        action_history = self.memory.get_recent_history("action", k=10)
        if action_history:
            params["action_history"] = "\n".join([str(a) for a in action_history if a])
        else:
            params["action_history"] = "No action history yet"

        self.memory.working_area.update(params)
        return params


class SkillCompositionProvider(BaseProvider):
    """AI 驱动的组合技能生成"""

    def __init__(self, llm_provider, gm, template_path, **kwargs):
        super().__init__(**kwargs)
        self.llm_provider = llm_provider
        self.gm = gm
        self.template = self._load_template(template_path)
        self.memory = LocalMemory()

    def _load_template(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    @BaseProvider.debug
    @BaseProvider.write
    def __call__(self):
        params = self.memory.working_area

        # 组装 prompt
        message_prompts = self.llm_provider.assemble_prompt(
            template_str=self.template,
            params=params
        )

        # 调用 LLM
        response, _ = self.llm_provider.create_completion(message_prompts)

        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict:
        """解析 LLM 响应，提取技能代码"""
        if not response or "NO_COMPOSITION_NEEDED" in response:
            return {"composition": None, "reasoning": response}

        # 提取 Python 代码块
        code_match = re.search(
            r'```python\s*\n(def \w+\(\):.*?)```',
            response,
            re.DOTALL
        )

        if code_match:
            code = code_match.group(1).strip()
            return {"composition": code, "reasoning": response}

        # 尝试直接匹配 def 函数定义
        code_match = re.search(
            r'(def \w+\(\):.*?(?=\n\n|\Z))',
            response,
            re.DOTALL
        )

        if code_match:
            code = code_match.group(1).strip()
            return {"composition": code, "reasoning": response}

        return {"composition": None, "reasoning": response}


class SkillCompositionPostprocessProvider(BaseProvider):
    """验证并注册生成的组合技能"""

    def __init__(self, gm, **kwargs):
        super().__init__(**kwargs)
        self.gm = gm
        self.memory = LocalMemory()

    @BaseProvider.debug
    @BaseProvider.write
    def __call__(self, response: Dict):
        composition = response.get("composition")

        if not composition:
            logger.write("No composition generated this cycle")
            return {"registered": False, "reason": "No composition"}

        # 从 working_area 获取 preprocess 阶段提取的允许按键
        allowed_keys = self.memory.working_area.get("_extracted_allowed_keys", [])

        # 验证代码只使用允许的按键
        is_valid, error = self._validate_keys(composition, allowed_keys)
        if not is_valid:
            logger.warn(f"Composition rejected: {error}")
            return {"registered": False, "reason": error}

        # 注册技能
        success, info = self.gm.add_new_skill(skill_code=composition)
        logger.write(f"Skill composition result: {info}")

        return {"registered": success, "reason": info}

    def _validate_keys(self, code: str, allowed_keys: List[str]) -> Tuple[bool, str]:
        """验证代码只使用允许的按键"""
        # 提取所有 key_press 中的按键
        keys_in_code = extract_keys_from_skill_code(code)

        for key in keys_in_code:
            if key not in allowed_keys:
                return False, f"Unauthorized key: {key} (allowed: {allowed_keys})"

        return True, ""
