"""乐谱识别:图片/文档 -> 曲名与规范化简谱文本。

设计为可插拔接口:
- StubRecognizer:仅供测试跑通全流程,正式界面未配置模型时不会使用
- OpenAIStyleRecognizer:兼容 OpenAI 协议的多模态接口,
  通义千问VL / GLM-4V / Kimi / OpenAI 等平台均可用,只需填 api_base/api_key/model

工程细节:
- 图片发送前自动等比压缩(长边 1600px、JPEG q85),避免大图上传/推理超时
- 使用流式响应(stream),长推理期间数据持续到达,不会触发读超时
- 网络错误转成可读的中文提示,HTTP 错误会透出服务端返回的原始信息
"""

import base64
import io
import json
import os
import re
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests
from PIL import Image

CONNECT_TIMEOUT = 15    # 建连超时(秒)
READ_TIMEOUT = 300      # 流式块间最长等待(秒)
IMAGE_MAX_SIDE = 1600   # 图片长边上限,超出则等比压缩
JPEG_QUALITY = 85

# 大模型输出协议:同时识别曲名与规范化简谱。标题缺失时显式返回 UNKNOWN,
# 避免模型为了满足格式而臆造名称。
JIANPU_PROMPT = """你是一位简谱识别专家。请识别输入乐谱的曲名、音符、时值与歌词，并严格按指定格式输出。

曲名规则:
1. 优先读取乐谱图片或文档中明确出现的标题、歌名或曲名
2. 没有明确曲名时输出 UNKNOWN，禁止猜测或根据旋律臆造

简谱规则:
1. 音高:中音用数字 1-7;高音在数字后加英文单引号 ' (如 5');低音在数字后加英文逗号 , (如 5,);休止符用 0
2. 时值:不加符号=四分音符;数字后跟一个下划线 _ = 八分音符,两个 __ = 十六分音符;跟一个减号 - = 二分音符,两个 -- = 全音符;附点用 · 加在时值符号后(如 5_· 是附点八分音符)
3. 和弦:多个同时演奏的音用方括号括起、空格分隔,时值符号加在右括号后(如 [1' 3' 5']- 是二分和弦);和弦内的音只写音高不写时值
4. 每个音符(或和弦)之间用空格分隔
5. 只有原谱明确标出的休止符才输出为 0;禁止仅根据横向字距、歌词语义、换气或乐句感觉添加休止
6. 保留原谱的乐句分行:JIANPU 中每个原谱行单独输出一行;换行本身不要转换成休止，后续由用户按原谱分行补入并试听校对
7. 小节线、调号(如 1=C)、拍号、歌词、装饰音符号、力度记号不写入 JIANPU
8. 如果原图未标注或看不清时值/休止,必须在 RHYTHM 中说明;保持谱面原有分行，不要自行推断休止
9. 不要输出其他解释文字,不要用代码块包裹

歌词规则:
1. 如果谱面有歌词,LYRICS 必须与 JIANPU 逐行对应,每个音符事件写一个歌词字并用空格分隔
2. 一字多音时后续音符用 _ 占位;休止符、过门或没有歌词的音符也用 _ 占位
3. 如果谱面没有歌词,输出 UNKNOWN;不要根据曲名补写歌词

输出格式(严格保留以下四个标签):
TITLE:
<图片或文档中的曲名;未识别到则写 UNKNOWN>
JIANPU:
<仅包含规范化简谱;保留原谱分行>
LYRICS:
<与 JIANPU 逐行、逐音符对应的歌词;无歌词写 UNKNOWN>
RHYTHM:
<一句话说明时值与停顿是明确识别还是推断;没有疑问写 OK>

示例输出:
TITLE:
小星星
JIANPU:
1 1 5 5 6 6 5-
4 4 3 3 2 2 1-
LYRICS:
一 闪 一 闪 亮 晶 晶
满 天 都 是 小 星 星
RHYTHM:
时值来自原谱标记
"""

SAMPLE_JIANPU = "1 1 5, 5, 6 6 5'- 4 4 3 3 2 2 1- 0 0 [1' 3' 5']- 1 2 3_ 3_ 5_· 5_"

_UNKNOWN_TITLES = {
    "unknown", "n/a", "na", "none", "null", "未知", "未识别", "未识别到",
    "无", "未命名", "无法识别", "无法确定",
}
_TITLE_RE = re.compile(r"^(?:title|曲名|歌名|标题)\s*[:：]\s*(.*)$", re.IGNORECASE)
_JIANPU_RE = re.compile(r"^(?:jianpu|简谱)\s*[:：]\s*(.*)$", re.IGNORECASE)
_LYRICS_RE = re.compile(r"^(?:lyrics|歌词)\s*[:：]\s*(.*)$", re.IGNORECASE)
_RHYTHM_RE = re.compile(r"^(?:rhythm|节奏说明|节奏)\s*[:：]\s*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class RecognitionResult:
    """模型响应中供界面消费的稳定结构。"""

    title: str
    jianpu_text: str
    raw: str = ""
    rhythm_notes: tuple[str, ...] = ()
    lyrics_lines: tuple[str, ...] = ()


def _clean_title(value: str) -> str:
    title = value.strip().strip("`#* \t\r\n\"'《》")
    if title.lower() in _UNKNOWN_TITLES:
        return ""
    return title[:80]


def parse_recognition_response(raw: str) -> RecognitionResult:
    """解析结构化识别响应，并兼容旧版仅返回简谱的模型输出。"""
    text = str(raw or "").strip()
    lines = [line.strip() for line in text.splitlines()
             if line.strip().lower() not in ("```", "```text")]
    title = ""
    jianpu_lines = []
    lyrics_lines = []
    rhythm_notes = []
    section = None
    saw_protocol = False

    for line in lines:
        title_match = _TITLE_RE.match(line)
        if title_match:
            saw_protocol = True
            section = "title"
            title = _clean_title(title_match.group(1))
            continue
        jianpu_match = _JIANPU_RE.match(line)
        if jianpu_match:
            saw_protocol = True
            section = "jianpu"
            inline = jianpu_match.group(1).strip()
            if inline:
                jianpu_lines.append(inline)
            continue
        lyrics_match = _LYRICS_RE.match(line)
        if lyrics_match:
            saw_protocol = True
            section = "lyrics"
            inline = lyrics_match.group(1).strip()
            if inline and inline.lower() not in _UNKNOWN_TITLES:
                lyrics_lines.append(inline)
            continue
        rhythm_match = _RHYTHM_RE.match(line)
        if rhythm_match:
            saw_protocol = True
            section = "rhythm"
            inline = rhythm_match.group(1).strip()
            if inline and inline.lower() != "ok":
                rhythm_notes.append(inline)
            continue
        if section == "title" and not title:
            title = _clean_title(line)
            section = None
        elif section == "jianpu":
            jianpu_lines.append(line)
        elif section == "lyrics":
            if line.lower() not in _UNKNOWN_TITLES:
                lyrics_lines.append(line)
        elif section == "rhythm" and line.lower() != "ok":
            rhythm_notes.append(line)

    # 老模型、手动粘贴仍可直接给一段纯简谱。
    jianpu = "\n".join(jianpu_lines).strip() if saw_protocol else text
    return RecognitionResult(
        title=title,
        jianpu_text=jianpu,
        raw=text,
        rhythm_notes=tuple(rhythm_notes),
        lyrics_lines=tuple(lyrics_lines),
    )


def add_line_break_rests(jianpu: str, rest_token: str = "0_") -> str:
    """在非空谱面行之间补休止，供用户确认后快速修正无节奏标记的图片谱。"""
    lines = [line.strip() for line in str(jianpu or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return str(jianpu or "").strip()
    return f" {rest_token}\n".join(lines)


def fallback_score_name(token: str | None = None) -> str:
    """识别不到曲名时生成短而可编辑的本地兜底名。"""
    suffix = (token or secrets.token_hex(2)).upper()[:4]
    return f"未命名乐谱-{suffix}"


def build_recognition_text(title: str, jianpu: str, lyrics, rhythm_notes=()) -> str:
    """构造可重新载入的识别文本，供上传页与乐谱详情共同持久化。"""
    lyric_text = " ".join(str(item).strip() or "_" for item in (lyrics or ())) or "UNKNOWN"
    rhythm_text = "；".join(str(item).strip() for item in (rhythm_notes or ()) if str(item).strip()) or "OK"
    return (
        f"TITLE:\n{str(title or '').strip()}\n"
        f"JIANPU:\n{str(jianpu or '').strip()}\n"
        f"LYRICS:\n{lyric_text}\n"
        f"RHYTHM:\n{rhythm_text}"
    )


class BaseRecognizer(ABC):
    @abstractmethod
    def recognize_image(self, image_path: str) -> str:
        """识别图片乐谱,返回规范化简谱文本。"""

    @abstractmethod
    def recognize_document(self, text: str) -> str:
        """识别/整理文档乐谱,返回规范化简谱文本。"""


class StubRecognizer(BaseRecognizer):
    """内置样例,不调用任何 API,用于端到端跑通流程。"""

    def recognize_image(self, image_path):
        return SAMPLE_JIANPU

    def recognize_document(self, text):
        return SAMPLE_JIANPU


def chat_messages(api_base: str, api_key: str, model: str, content: list, timeout: int = READ_TIMEOUT) -> str:
    """OpenAI 兼容协议的流式对话(识别与编谱建议共用)。

    网络/HTTP 错误统一转成可读的中文提示。
    """
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": [{"role": "user", "content": content}], "stream": True}
    try:
        with requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(CONNECT_TIMEOUT, timeout),
            stream=True,
        ) as resp:
            resp.raise_for_status()
            parts = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                parts.append(delta.get("content") or "")
            text = "".join(parts).strip()
            if not text:
                raise RuntimeError("模型未返回内容:请检查模型名称是否为支持图片输入的视觉模型")
            return text
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        detail = ""
        try:
            detail = (e.response.text or "")[:300]
        except Exception:
            pass
        raise RuntimeError(f"接口返回错误(HTTP {status}): {detail}") from e
    except requests.exceptions.ReadTimeout as e:
        raise RuntimeError("识别请求超时:图片过大或网络较慢,请重试一次;多次超时可换用更快的模型(如 qwen-vl-plus)") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"网络错误: {e}") from e


def chat_text(api_base: str, api_key: str, model: str, prompt: str, timeout: int = READ_TIMEOUT) -> str:
    """纯文本对话快捷入口(编谱建议等文本型任务使用)。"""
    return chat_messages(api_base, api_key, model, [{"type": "text", "text": prompt}], timeout)


class OpenAIStyleRecognizer(BaseRecognizer):
    """OpenAI 兼容协议的多模态识别接口(通义/GLM/Kimi/OpenAI 通用)。"""

    def __init__(self, api_base: str, api_key: str, model: str, timeout: int = READ_TIMEOUT):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _compress_image(image_path: str):
        """大图等比压缩为 JPEG,返回 (jpeg_bytes, mime)。"""
        ext = os.path.splitext(image_path)[1].lower()
        raw_mime = "image/png" if ext == ".png" else "image/jpeg"
        img = Image.open(image_path)
        img.load()
        small = max(img.size) <= IMAGE_MAX_SIDE
        if small and raw_mime == "image/jpeg" and img.mode == "RGB":
            with open(image_path, "rb") as f:
                return f.read(), raw_mime
        if img.mode != "RGB":
            img = img.convert("RGB")
        if max(img.size) > IMAGE_MAX_SIDE:
            img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue(), "image/jpeg"

    def _chat(self, text_prompt: str, image_path: str | None = None) -> str:
        content = [{"type": "text", "text": text_prompt}]
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片不存在: {image_path}")
            data, mime = self._compress_image(image_path)
            b64 = base64.b64encode(data).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        return chat_messages(self.api_base, self.api_key, self.model, content, self.timeout)

    def recognize_image(self, image_path: str) -> str:
        return self._chat(JIANPU_PROMPT, image_path=image_path)

    def recognize_document(self, text: str) -> str:
        return self._chat(f"{JIANPU_PROMPT}\n\n以下是文档提取出的乐谱内容:\n{text}")


def get_recognizer(cfg: dict) -> BaseRecognizer:
    """按配置实例化识别器。"""
    r = cfg.get("recognizer", {})
    provider = r.get("provider", "stub")
    if provider == "stub":
        return StubRecognizer()
    if provider == "openai_compatible":
        if not r.get("api_key"):
            raise ValueError("recognizer.api_key 未配置,请先在 config.yaml 填写 API Key")
        return OpenAIStyleRecognizer(
            api_base=r.get("api_base", ""),
            api_key=r.get("api_key", ""),
            model=r.get("model", ""),
            timeout=int(r.get("timeout", READ_TIMEOUT)),
        )
    raise ValueError(f"未知的 recognizer.provider: {provider}")


def get_recognizer_from_provider(provider: dict | None) -> BaseRecognizer | None:
    """按激活供应商创建识别器;配置不完整时返回 None，由界面明确引导。"""
    required = ("base_url", "api_key", "model")
    if not provider or not all(str(provider.get(k, "")).strip() for k in required):
        return None
    return OpenAIStyleRecognizer(
        api_base=provider.get("base_url", ""),
        api_key=provider.get("api_key", ""),
        model=provider.get("model", ""),
    )
