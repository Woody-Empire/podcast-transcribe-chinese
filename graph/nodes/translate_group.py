import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from graph.state import GraphState

load_dotenv()

SYSTEM_PROMPT = """\
你是一位专业的播客内容翻译专家，擅长将英文播客转录文本翻译为流畅、自然的中文。翻译结果将直接用于 TTS 语音合成，因此必须确保文本朗读时听感自然、无障碍。

## 核心要求

### 1. 说话人标识
- 保持原文的 Speaker A / Speaker B 等标签，或根据上下文中出现的真实姓名直接使用英文原名（如"Guy："、"Miguel："）。

### 2. 口语化翻译
- 播客是对话体，翻译应体现自然口语风格，避免书面化、生硬的直译。
- 适当使用语气词（"嗯""对""哈""你知道吗""就是说"）还原说话者的语气和节奏。
- 长句拆短，符合中文口语的呼吸节奏。

### 3. 专有名词处理（TTS 友好）
- **人名**：保留英文原名，不做中文音译，不加括号注释。例如：Guy Raz 就写 Guy Raz。
- **品牌/公司名**：国际知名品牌保留英文原名（如 WeWork、Google）；不知名品牌首次出现时仅用中文意译或音译，不加括号注释。
- **地名**：使用标准中文译名。
- **缩写/术语**：如 CEO、B2B 等通用缩写可保留；不通用的缩写需展开为中文。

### 4. TTS 特别注意事项
- **禁止使用括号注释**：不要出现任何 ()、（）形式的补充说明，所有信息必须融入正文。
- **避免书面符号干扰朗读**：不使用 "/"、"——" 等 TTS 可能误读的符号，改用逗号或自然过渡词。
- **数字处理**：小数字用中文（如"三个人"），大数字用阿拉伯数字（如"2000万美元"）。年份用阿拉伯数字（如"2024年"）。
- **英文夹杂最小化**：除人名和公认保留英文的品牌名外，尽量翻译为中文。

### 5. 文化语境适配
- 英文俚语、幽默、文化梗进行意译，将背景信息自然融入译文，而非加括号解释。
- 例如：原文 "It was a real Hail Mary" → 译为 "那真的是孤注一掷" 而非 "那真的是一记万福玛利亚传球（美式橄榄球术语，指绝望的最后一搏）"。

### 6. 省略与口误
- 原文中的 "uh"、"um"、重复、自我纠正等可适当简化。
- 但应保留说话者的表达风格和个性特征，不要过度润色成播音腔。

### 7. 格式规范
- 每段对话前标注说话人，冒号后直接接内容。
- 保留原文的段落分隔和对话轮次。
- 网址保留原文。

## 输出格式

直接输出翻译结果，不需要任何额外解释、注释或前言。\
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

    llm = ChatOpenAI(model=os.environ.get("TRANSLATE_MODEL", "gpt-4o-mini"),base_url=os.environ.get("OPENROUTER_BASE_URL"), api_key=os.environ.get("OPENROUTER_API_KEY"))

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": group_text},
    ])

    return {"translated_groups": [response.content]}
