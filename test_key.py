import os
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError  # 导入 OpenAI 异常
import numpy as np

# ----------------------------------------------------
# 客户端初始化 (统一使用 OpenAI 兼容模式)
# ----------------------------------------------------

# 确保在运行脚本前，DASHSCOPE_API_KEY 环境变量已设置！
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-vl-plus-latest"
EMB_MODEL = "text-embedding-v2"

if not API_KEY:
    print("致命错误：环境变量 DASHSCOPE_API_KEY 未设置！请先设置。")
    exit()

try:
    # 使用 OpenAI 客户端，Base URL 指向 DashScope 兼容模式
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"客户端初始化成功，Base URL: {BASE_URL}")
except Exception as e:
    print(f"客户端初始化失败: {e}")
    exit()


# ----------------------------------------------------
# 步骤 1: 验证 LLM 模型权限 (qwen-vl-plus-latest)
# ----------------------------------------------------

def test_llm_model():
    """测试 LLM 模型调用，使用 OpenAI 兼容客户端"""
    print("\n--- 1. 正在测试 LLM 模型 (qwen-vl-plus-latest) ---")

    try:
        # 使用 OpenAI 客户端的 chat.completions.create
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': '请自我介绍。'}],
            temperature=0.5,
            max_tokens=256
        )

        if response and response.choices:
            content = response.choices[0].message.content
            print(f"[成功] LLM 调用通过。模型响应前缀: {content[:30]}...")
            return True
        else:
            print("[失败] LLM 调用失败：无响应或无内容。")
            return False

    except (APIError, RateLimitError, APITimeoutError) as e:
        print(f"[失败] LLM 调用 API 错误: {e}")
        return False
    except Exception as e:
        print(f"[致命失败] LLM 调用发生异常: {type(e).__name__}: {e}")
        return False


# ----------------------------------------------------
# 步骤 2: 验证 Embedding 模型权限 (text-embedding-v2)
# ----------------------------------------------------

def test_embedding_model():
    """测试 Embedding 模型调用，使用 OpenAI 兼容客户端"""
    print("\n--- 2. 正在测试 Embedding 模型 (text-embedding-v2) ---")

    try:
        # 使用 OpenAI 客户端的 embeddings.create
        texts = ["测试文本，用于生成嵌入向量。"]
        response = client.embeddings.create(
            model=EMB_MODEL,
            input=texts
        )

        # 兼容模式下返回的是标准 OpenAI 响应对象
        if response and response.data:
            embedding_dim = len(response.data[0].embedding)
            print(f"[成功] Embedding 调用通过。嵌入向量维度: {embedding_dim} (应为 1024)")
            return True
        else:
            print("[失败] Embedding 调用失败：无响应或无内容。")
            return False

    except (APIError, RateLimitError, APITimeoutError) as e:
        print(f"[失败] Embedding 调用 API 错误: {e}")
        return False
    except Exception as e:
        print(f"[致命失败] Embedding 调用发生异常: {type(e).__name__}: {e}")
        return False


# ----------------------------------------------------
# 主程序运行
# ----------------------------------------------------

if __name__ == "__main__":
    print("\n--- 正在使用 OpenAI 兼容客户端测试 DashScope API ---")

    llm_ok = test_llm_model()
    emb_ok = test_embedding_model()

    if llm_ok and emb_ok:
        print("\n--- 总结: 密钥和模型权限验证成功。解决方案已确定。---")
    else:
        print("\n--- 总结: 验证失败。请检查密钥权限。---")