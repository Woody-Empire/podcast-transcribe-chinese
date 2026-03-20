import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from graph.state import GraphState

load_dotenv()

SYSTEM_PROMPT = """\
你是一位专业的播客内容翻译专家，擅长将英文播客转录文本翻译为流畅、自然的中文。请按照以下规则完成翻译：

## 核心要求

1. **保留说话人标识**：保持原文的 Speaker A / Speaker B / Speaker C 等标签不变，或根据上下文中出现的真实姓名替换为中文标注（如"Guy：""Miguel："）。

2. **口语化翻译**：播客是对话体，翻译应体现口语风格，避免书面化、生硬的直译。允许使用语气词（"嗯""对""哈"）来还原说话者的语气和节奏。

3. **专有名词处理**：
   - 人名：首次出现时用"中文音译（英文原名）"格式，之后仅用中文音译。
   - 品牌/公司名：知名品牌使用通用中文译名（如 WeWork 保留原名），新品牌首次出现时注明原名。
   - 地名：使用标准中文译名。

4. **文化语境适配**：对英文语境中的俚语、幽默、文化梗进行意译，必要时添加极简括号注释以帮助中文读者理解。

5. **格式规范**：
   - 每段对话前标注说话人，冒号后换行或空格。
   - 保留原文的段落分隔。
   - 电话号码、网址等信息保留原文。

6. **省略与口误**：原文中的停顿（"uh""um"）、重复、自我纠正等口语特征，翻译时可适当简化，但应保留说话者的表达风格和个性。

## 输出格式

直接输出翻译结果，不需要额外解释或注释。\
"""


class TranslateGroupInput(TypedDict):
    """Send() 并行调用时传入的单组数据。"""
    group_text: str


def translate_group(state: TranslateGroupInput) -> GraphState:
    """使用 LLM 将一组英文播客转录文本翻译为中文。

    该节点设计为通过 Send() 并行调用，每次处理一个 utterance group。
    示例（在 graph 中组织时）：
        def fan_out_translate(state):
            return [
                Send("translate_group", {"group_text": g})
                for g in state["utterance_groups"]
            ]
    """
    group_text = state["group_text"]

    llm = ChatOpenAI(model=os.environ.get("TRANSLATE_MODEL", "gpt-4o-mini"))

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": group_text},
    ])

    return {"translated_groups": [response.content]}
