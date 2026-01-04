import os
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Optional,
)
import json
import re
import asyncio

import backoff
import numpy as np

# ⭐ 核心: 导入 OpenAI 兼容客户端及其异常
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

# 保持 DashScope 导入，但不再使用其客户端
import dashscope

from cradle import constants
from cradle.provider.base import LLMProvider, EmbeddingProvider
from cradle.config import Config
from cradle.log import Logger
from cradle.utils.json_utils import load_json
from cradle.utils.file_utils import assemble_project_path
from cradle.utils.encoding_utils import encode_data_to_base64_path

config = Config()
logger = Logger()

# Qwen API 相关的配置键
PROVIDER_SETTING_KEY_VAR = "key_var"
PROVIDER_SETTING_COMP_MODEL = "comp_model"
PROVIDER_SETTING_BASE_URL = "base_url"  # 必须存在，用于 OpenAI 客户端
PROVIDER_SETTING_EMB_MODEL = "emb_model"
PROVIDER_SETTING_IS_AZURE = "is_azure"

# 错误类型映射到 OpenAI 客户端的异常
APIError = APIError
InvalidInput = APIError
Unauthorized = APIError


class QwenProvider(LLMProvider, EmbeddingProvider):
    """A class that wraps the Qwen (DashScope) model using the OpenAI compatible API."""

    client: OpenAI = None
    llm_model: str = ""
    embedding_model: str = ""
    retries: int = 5

    def init_provider(self, provider_config_path: str):
        """Initializes the Qwen client using the OpenAI compatible API."""
        config_data = load_json(provider_config_path)

        key_var = config_data.get(PROVIDER_SETTING_KEY_VAR)
        key = os.environ.get(key_var)
        if key is None:
            raise EnvironmentError(f"Qwen API key not found in environment variable: {key_var}")

        # 检查并获取 base_url
        base_url = config_data.get(PROVIDER_SETTING_BASE_URL)
        if base_url is None:
            # 此处应确保配置中存在 base_url
            raise ValueError(
                "qwen_config.json must contain 'base_url' set to the DashScope compatible endpoint (e.g., https://dashscope.aliyuncs.com/compatible-mode/v1)."
            )

        # ⭐ 核心修复: 使用 OpenAI 客户端初始化
        self.client = OpenAI(api_key=key, base_url=base_url)

        # 保持 DashScope 原生 SDK 的 key 设置，以防框架其他部分需要
        dashscope.api_key = key
        os.environ["DASHSCOPE_API_KEY"] = key

        # 设置 LLM 和 Embedding 模型名称
        self.llm_model = config_data.get(PROVIDER_SETTING_COMP_MODEL)
        self.embedding_model = config_data.get(PROVIDER_SETTING_EMB_MODEL)

    # ⭐ 修复: 重写 invoke_model 使用 OpenAI 客户端
    @backoff.on_exception(backoff.expo, (APIError, RateLimitError, APITimeoutError), max_tries=5, jitter=None)
    def invoke_model(self, messages: List[Dict[str, Any]], temperature: float, seed: int, max_tokens: int) -> str:
        """Invokes the Qwen LLM using the OpenAI compatible client."""

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response or not response.choices:
            logger.error("Qwen API error: No response or choices in response.")
            raise APIError("No response or choices in response.")

        return response.choices[0].message.content

    # ⭐ 实现抽象方法 create_completion (调用 invoke_model)
    def create_completion(
            self,
            messages: List[Dict[str, Any]],
            model: str | None = None,
            temperature: float = config.temperature,
            seed: int = config.seed,
            max_tokens: int = config.max_tokens,
    ) -> Tuple[str, Dict[str, int]]:
        """Synchronous completion."""

        content = self.invoke_model(
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens
        )

        # 返回占位符 token 信息
        info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "system_fingerprint": None,
        }

        return content, info

    # ⭐ 实现抽象方法 create_completion_async
    async def create_completion_async(
            self,
            messages: List[Dict[str, Any]],
            model: str | None = None,
            temperature: float = config.temperature,
            seed: int = config.seed,
            max_tokens: int = config.max_tokens,
    ) -> Tuple[str, Dict[str, int]]:
        """Asynchronous completion, run in a separate thread."""
        result = await asyncio.to_thread(
            self.create_completion,
            messages=messages,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens
        )
        return result

    # ⭐ 修复: 重写 embed_documents 使用 OpenAI 兼容客户端
    @backoff.on_exception(backoff.expo, (APIError, RateLimitError, APITimeoutError), max_tries=5, jitter=None)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute the embeddings for a list of texts using OpenAI compatible client."""
        logger.debug(f"Calling Qwen embedding model: {self.embedding_model}")

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )

        if not response or not response.data:
            logger.error("Qwen Embedding API error: No response or data in response.")
            raise APIError("No response or data in response.")

        # OpenAI 兼容模式下的响应是 response.data[i].embedding
        embeddings = [record.embedding for record in response.data]

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute the embedding for a single text query."""
        return self.embed_documents([text])[0]

    # ⭐ 保持 get_embedding_dim 方法不变
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension based on the model name."""
        if "text-embedding-v1" in self.embedding_model:
            return 1024
        elif "text-embedding-v2" in self.embedding_model:
            return 1024
        elif "text-embedding-v4" in self.embedding_model:
            return 1536
        else:
            raise ValueError(f"Unknown Qwen embedding model: {self.embedding_model}. Please specify dimension.")

    # ⭐ 保持 _merge_messages 方法不变
    def _merge_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not messages:
            return []

        merged_messages = []
        current_message = messages[0].copy()

        for next_message in messages[1:]:
            if next_message["role"] == current_message["role"] and next_message["role"] == "user":
                if isinstance(current_message["content"], list) and isinstance(next_message["content"], list):
                    current_message["content"].extend(next_message["content"])
                else:
                    current_message["content"] += "\n\n" + next_message["content"]
            else:
                merged_messages.append(current_message)
                current_message = next_message.copy()

        merged_messages.append(current_message)
        return merged_messages

    # ⭐ 核心修复: 图像消息结构适配
    def assemble_prompt_tripartite(self, template_str: str = None, params: Dict[str, Any] = None) -> List[
        Dict[str, Any]]:

        # 1. 调用原始的 Claude 组装逻辑
        from .claude import ClaudeProvider as BasePromptAssembler
        messages = BasePromptAssembler.assemble_prompt_tripartite(self, template_str=template_str, params=params)

        # 2. ⭐ 核心修复: 迭代消息结构，将所有不规范的图像数据统一为 OpenAI 嵌套格式
        for message in messages:
            if isinstance(message.get('content'), list):
                new_content_list = []
                for item in message['content']:
                    current_type = item.get('type')
                    image_data_found = None

                    # 尝试从所有可能的字段中提取 Base64 URL 字符串
                    if current_type == 'image':
                        # 可能性 A: Claude 遗留的结构 {'type': 'image', 'image': 'base64'}
                        image_data_found = item.pop('image', None)
                    elif current_type == 'image_url':
                        # 可能性 B, C: 数据可能在 image_url 字典或扁平字符串中
                        image_url_dict_or_str = item.get('image_url')

                        if isinstance(image_url_dict_or_str, dict) and 'url' in image_url_dict_or_str:
                            image_data_found = image_url_dict_or_str['url']
                        elif isinstance(image_url_dict_or_str, str):
                            image_data_found = image_url_dict_or_str

                    # --- 执行统一和修复 ---

                    if image_data_found and isinstance(image_data_found, str) and image_data_found.startswith(
                            "data:image"):
                        # 如果找到了有效的 Base64 数据，则构建正确的 OpenAI 嵌套结构
                        new_content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_found
                            }
                        })

                    # 如果当前 item 不是图像 (即 'type':'text')，则保留
                    elif current_type != 'image' and current_type != 'image_url':
                        new_content_list.append(item)

                    # NOTE: 任何无法成功提取 Base64 数据的图像部分都将被忽略，从而避免 400 错误的请求。

                message['content'] = new_content_list

        return messages

    def assemble_prompt_paragraph(self, template_str: str = None, params: Dict[str, Any] = None) -> List[
        Dict[str, Any]]:
        raise NotImplementedError("This method is not implemented yet.")

    def assemble_prompt(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_TRIPART:
            return self.assemble_prompt_tripartite(template_str=template_str, params=params)
        elif config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_PARAGRAPH:
            return self.assemble_prompt_paragraph(template_str=template_str, params=params)