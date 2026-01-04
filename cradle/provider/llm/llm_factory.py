from cradle.provider.llm.openai import OpenAIProvider
from cradle.provider.llm.restful_claude import RestfulClaudeProvider
from cradle.utils import Singleton

# ⭐ 导入 QwenProvider
from cradle.provider.llm.qwen import QwenProvider
from cradle.log import Logger

logger = Logger()


class LLMFactory(metaclass=Singleton):

    def __init__(self):
        self._builders = {}

    def create(self, llm_provider_config_path, embed_provider_config_path, **kwargs):

        llm_provider = None
        embed_provider = None

        key = llm_provider_config_path

        if "openai" in key:
            llm_provider = OpenAIProvider()
            llm_provider.init_provider(llm_provider_config_path)
            embed_provider = llm_provider
        elif "claude" in key:
            llm_provider = RestfulClaudeProvider()
            llm_provider.init_provider(llm_provider_config_path)
            # Claude 不支持原生嵌入，所以需要从另一个配置文件加载嵌入提供商
            # 这里默认使用 embed_provider_config_path 来加载 OpenAI 配置
            embed_provider = OpenAIProvider()
            embed_provider.init_provider(embed_provider_config_path)

        # ⭐ 新增 Qwen Provider 逻辑 (使用自身作为嵌入提供商)
        elif "qwen" in key:
            llm_provider = QwenProvider()
            llm_provider.init_provider(llm_provider_config_path)
            # Qwen 提供自己的嵌入模型
            embed_provider = llm_provider

        if not llm_provider or not embed_provider:
            raise ValueError(key)

        return llm_provider, embed_provider