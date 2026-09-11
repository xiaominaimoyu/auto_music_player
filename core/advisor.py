"""AI 编谱建议(实验性):用文本模型生成符合 21 键协议的简谱建议。"""

from dataclasses import dataclass, field

from core.parser import parse_jianpu
from core.recognizer import READ_TIMEOUT, chat_text

ADVISOR_PROMPT = """你是一位游戏乐器编谱专家。游戏乐器只有 21 个琴键:
高音 1-7、中音 1-7、低音 1-7(对应简谱协议)。

请根据用户输入创作或补全一段简谱,并给出演奏建议。

简谱协议(必须严格遵守):
1. 音高:中音用数字 1-7;高音在数字后加英文单引号 ' (如 5');低音在数字后加英文逗号 , (如 5,);休止符用 0
2. 时值:不加符号=四分音符;下划线 _ = 八分音符,两个 __ = 十六分音符;减号 - = 二分音符,两个 -- = 全音符;附点用 ·(如 5_·)
3. 和弦:多个同时演奏的音用方括号括起、空格分隔(如 [1' 3' 5']),和弦内只写音高
4. 音符之间用空格分隔;不要小节线、调号、歌词、装饰音
5. 只使用上述记号,不要使用代码块

输出格式(严格遵守,不要输出其他内容):
JIANPU:
<一行简谱>
TIPS:
<最多 3 条演奏建议,每条一行>

用户输入:{desc}"""


@dataclass
class AdvisorResult:
    jianpu_text: str = ""
    tips: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    raw: str = ""
    extracted: bool = False

    @property
    def ok(self) -> bool:
        return self.extracted


def generate_advice(desc: str, api_base: str, api_key: str, model: str,
                    timeout: int = READ_TIMEOUT, chat_fn=None) -> AdvisorResult:
    """调用模型生成编谱建议;chat_fn 可注入,便于测试时避免触网。"""
    if not desc.strip():
        raise ValueError("请输入旋律描述或简谱片段")
    chat = chat_fn or chat_text
    prompt = ADVISOR_PROMPT.replace("{desc}", desc.strip())
    raw = chat(api_base, api_key, model, prompt, timeout)
    return parse_advice_response(raw)


def parse_advice_response(raw: str) -> AdvisorResult:
    """拆分模型输出,并用宽容解析器报告无法识别的记号。"""
    jianpu_lines, tips = _split_sections(raw)
    missing_format = not jianpu_lines
    if missing_format:
        jianpu_lines = [raw]
    text = " ".join(jianpu_lines).strip()
    notes, errors = parse_jianpu(text, collect=True)
    warnings = []
    if missing_format:
        warnings.append("模型未按 JIANPU/TIPS 格式输出,已自动提取音符")
    warnings.extend(str(e) for e in errors[:5])
    extracted = bool(notes)
    if not extracted:
        warnings.append("未能从模型输出中提取到任何音符")
    return AdvisorResult(
        jianpu_text=text,
        tips=tips[:5],
        warnings=warnings,
        raw=raw,
        extracted=extracted,
    )


def _split_sections(raw: str):
    jianpu, tips = [], []
    section = None
    for line in raw.splitlines():
        value = line.strip()
        if not value:
            continue
        heading = value.upper()
        if heading.startswith("JIANPU"):
            section = "jianpu"
            continue
        if heading.startswith("TIPS"):
            section = "tips"
            continue
        if section == "jianpu":
            jianpu.append(value)
        elif section == "tips":
            tips.append(value)
    return jianpu, tips
