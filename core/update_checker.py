"""便携版更新检查及经用户选择的独立 EXE 下载。

发布侧应把同一份 manifest 放在 GitHub/Gitee 两个镜像，客户端用内嵌的
Ed25519 公钥验证 canonical JSON。镜像只负责可用性，不能改变信任根。
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re


MAX_MANIFEST_BYTES = 1 * 1024 * 1024
MAX_UPDATE_BYTES = 2 * 1024 * 1024 * 1024
RELEASE_PUBLIC_KEY = "6QF3M1S8xGQslxyJc2RLlCm8xy0HTRYXZvXk6RgMInY="
_VERSION_RE = re.compile(r"^\d+(?:\.\d+){1,3}(?:-[0-9A-Za-z.-]+)?$")


class UpdateCheckError(RuntimeError):
    pass


class UpdateSecurityError(UpdateCheckError):
    pass


class UpdateCancelled(UpdateCheckError):
    pass


def _check_cancelled(cancelled):
    if cancelled and cancelled():
        raise UpdateCancelled("已取消更新任务")


class _HttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise UpdateSecurityError("拒绝更新请求重定向到非 HTTPS 地址")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_https(url, *, timeout, accept):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise UpdateSecurityError("更新地址必须使用 HTTPS")
    request = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": "AutoMusicPlayer/UpdateCheck"}
    )
    return urllib.request.build_opener(_HttpsRedirectHandler()).open(request, timeout=timeout)


def _version_key(value: str):
    raw = str(value or "0").strip().lstrip("vV")
    main, _, suffix = raw.partition("-")
    numbers = []
    for item in main.split("."):
        try:
            numbers.append(int(item))
        except ValueError:
            numbers.append(0)
    while len(numbers) < 4:
        numbers.append(0)
    # 正式版本高于 prerelease；同为 prerelease 按字符串保持稳定顺序。
    return tuple(numbers[:4]) + (1 if not suffix else 0, suffix)


def _canonical_json(payload: dict) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class UpdateManifest:
    schema_version: int
    version: str
    sequence: int
    min_supported: str
    urls: tuple[str, ...]
    size: int
    sha256: str
    signature: str
    issued_at: str = ""
    expires_at: str = ""
    db_schema_version: int = 0
    release_notes: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, value: dict) -> "UpdateManifest":
        if not isinstance(value, dict):
            raise UpdateCheckError("更新 manifest 必须是 JSON 对象")
        required = ("version", "sequence", "urls", "size", "sha256", "signature")
        missing = [key for key in required if key not in value]
        if missing:
            raise UpdateCheckError(f"更新 manifest 缺少字段：{', '.join(missing)}")
        urls = value.get("urls")
        if not isinstance(urls, list) or not urls:
            raise UpdateCheckError("更新 manifest 必须包含 urls 列表")
        normalized_urls = []
        for url in urls:
            parsed = urllib.parse.urlparse(str(url))
            if parsed.scheme != "https" or not parsed.netloc:
                raise UpdateSecurityError("更新地址必须使用 HTTPS")
            normalized = str(url).strip()
            if normalized in normalized_urls:
                continue
            normalized_urls.append(normalized)
        try:
            size = int(value["size"])
            sequence = int(value["sequence"])
        except (TypeError, ValueError) as exc:
            raise UpdateCheckError("更新 manifest 的 sequence/size 无效") from exc
        if sequence <= 0 or size <= 0:
            raise UpdateCheckError("更新 manifest 的 sequence/size 必须为正数")
        if size > MAX_UPDATE_BYTES:
            raise UpdateCheckError("更新文件不能超过 2 GB")
        try:
            schema_version = int(value.get("schema_version", 1))
            db_schema_version = int(value.get("db_schema_version", 0))
        except (TypeError, ValueError) as exc:
            raise UpdateCheckError("更新 manifest 的 schema 版本无效") from exc
        if schema_version != 1:
            raise UpdateCheckError(f"不支持的更新 manifest 版本：{schema_version}")
        version = str(value["version"]).strip().lstrip("vV")
        if not _VERSION_RE.fullmatch(version):
            raise UpdateCheckError("更新 manifest 的 version 无效")
        sha256 = str(value["sha256"]).lower()
        if len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256):
            raise UpdateCheckError("更新 manifest 的 sha256 无效")
        return cls(
            schema_version=schema_version,
            version=version,
            sequence=sequence,
            min_supported=str(value.get("min_supported", "0")),
            urls=tuple(normalized_urls),
            size=size,
            sha256=sha256,
            signature=str(value["signature"]),
            issued_at=str(value.get("issued_at", "")),
            expires_at=str(value.get("expires_at", "")),
            db_schema_version=db_schema_version,
            release_notes=str(value.get("release_notes", "")),
            raw=dict(value),
        )

    def unsigned_payload(self) -> dict:
        payload = dict(self.raw)
        payload.pop("signature", None)
        return payload


def verify_manifest(manifest: UpdateManifest, public_key_b64: str) -> None:
    """验证 Ed25519 签名；未配置公钥时明确失败而不是降级为信任网络。"""

    if not public_key_b64:
        raise UpdateSecurityError("尚未配置更新 manifest 公钥")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_der_public_key

        raw_key = base64.b64decode(public_key_b64, validate=True)
        try:
            public_key = Ed25519PublicKey.from_public_bytes(raw_key)
        except ValueError:
            public_key = load_der_public_key(raw_key)
        if not isinstance(public_key, Ed25519PublicKey):
            raise UpdateSecurityError("更新 manifest 公钥不是 Ed25519 公钥")
        signature = base64.b64decode(manifest.signature, validate=True)
        public_key.verify(signature, _canonical_json(manifest.unsigned_payload()))
    except UpdateSecurityError:
        raise
    except Exception as exc:
        raise UpdateSecurityError("更新 manifest 签名校验失败") from exc


def _check_expiry(manifest: UpdateManifest, now=None):
    now = now or datetime.now(timezone.utc)
    if manifest.issued_at:
        try:
            issued = datetime.fromisoformat(manifest.issued_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise UpdateSecurityError("更新 manifest issued_at 无效") from exc
        if issued.tzinfo is None:
            issued = issued.replace(tzinfo=timezone.utc)
        if issued > now:
            raise UpdateSecurityError("更新 manifest 的签发时间晚于当前时间")
    if not manifest.expires_at:
        return
    try:
        expiry = datetime.fromisoformat(manifest.expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UpdateSecurityError("更新 manifest expires_at 无效") from exc
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    if expiry <= now:
        raise UpdateSecurityError("更新 manifest 已过期")


@dataclass(frozen=True)
class UpdateCheckResult:
    update_available: bool
    manifest: UpdateManifest | None = None
    source_url: str = ""
    error: str = ""


def fetch_manifest(url: str, *, timeout_s: float = 8.0) -> dict:
    try:
        with _open_https(str(url), timeout=max(1.0, float(timeout_s)), accept="application/json") as response:
            payload = response.read(MAX_MANIFEST_BYTES + 1)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise UpdateCheckError(f"获取更新 manifest 失败：{exc}") from exc
    if len(payload) > MAX_MANIFEST_BYTES:
        raise UpdateCheckError("更新 manifest 超过 1 MB")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateCheckError("更新 manifest 不是有效 UTF-8 JSON") from exc
    return value


def check_for_update(
    current_version: str,
    manifest_urls,
    public_key_b64: str,
    *,
    current_sequence: int = 0,
    timeout_s: float = 8.0,
    cancelled=None,
) -> UpdateCheckResult:
    """验证各镜像的清单并选择最高序列。"""

    errors = []
    candidates = []
    for url in tuple(manifest_urls or ()):
        _check_cancelled(cancelled)
        try:
            manifest = UpdateManifest.from_dict(fetch_manifest(url, timeout_s=timeout_s))
            verify_manifest(manifest, public_key_b64)
            _check_expiry(manifest)
            if manifest.sequence < int(current_sequence):
                raise UpdateSecurityError("拒绝低于当前序列号的更新")
            if _version_key(manifest.version) < _version_key(current_version):
                raise UpdateSecurityError("拒绝更低版本的更新")
            candidates.append((manifest, str(url)))
        except UpdateCheckError as exc:
            errors.append(f"{url}: {exc}")
    _check_cancelled(cancelled)
    if candidates:
        manifest, source_url = max(
            candidates,
            key=lambda item: (item[0].sequence, _version_key(item[0].version)),
        )
        available = (
            manifest.sequence > int(current_sequence)
            or _version_key(manifest.version) > _version_key(current_version)
        )
        return UpdateCheckResult(available, manifest, source_url)
    return UpdateCheckResult(False, error="；".join(errors) or "未配置更新 manifest")


def download_update(manifest, directory, *, current_version, current_sequence, cancelled=None, progress=None):
    """下载到所选目录，校验后原子发布为新文件，绝不覆盖已有文件。"""
    manifest = UpdateManifest.from_dict(manifest.raw)
    verify_manifest(manifest, RELEASE_PUBLIC_KEY)
    _check_expiry(manifest)
    version_key = _version_key(manifest.version)
    current_key = _version_key(current_version)
    if (manifest.sequence < current_sequence or version_key < current_key
            or (manifest.sequence == current_sequence and version_key == current_key)):
        raise UpdateSecurityError("该更新不高于当前版本或序列号")

    directory = Path(directory).resolve(strict=True)
    target = directory / f"AutoMusicPlayer-v{manifest.version}.exe"
    if target.exists():
        raise FileExistsError(f"文件已存在，请选择另一个保存目录：{target}")
    errors = []
    for url in manifest.urls:
        _check_cancelled(cancelled)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    _stream_package(url, manifest, stream, cancelled, progress)
                except (UpdateSecurityError, urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as exc:
                    errors.append(f"{urllib.parse.urlsplit(url).netloc}：{exc}")
                    continue
                stream.flush()
                os.fsync(stream.fileno())
            _check_expiry(manifest)
            _check_cancelled(cancelled)
            # Windows rename refuses an existing destination; POSIX link has the same no-clobber guarantee.
            if os.name == "nt":
                os.rename(temporary, target)
            else:
                os.link(temporary, target)
            return target
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise UpdateCheckError("所有更新镜像下载失败：" + "；".join(errors))


def _stream_package(url, manifest, stream, cancelled, progress):
    digest = hashlib.sha256()
    received = 0
    with _open_https(url, timeout=8.0, accept="application/octet-stream") as response:
        if progress:
            progress(0)
        while True:
            _check_cancelled(cancelled)
            chunk = response.read1(256 * 1024)
            if not chunk:
                break
            received += len(chunk)
            if received > manifest.size:
                raise UpdateSecurityError("更新文件超出清单大小")
            stream.write(chunk)
            digest.update(chunk)
            if progress:
                progress(received * 100 // manifest.size)
    if received != manifest.size or digest.hexdigest() != manifest.sha256:
        raise UpdateSecurityError("更新文件大小或 SHA-256 与签名清单不符")


__all__ = [
    "UpdateCheckError",
    "UpdateCheckResult",
    "UpdateManifest",
    "UpdateSecurityError",
    "check_for_update",
    "verify_manifest",
    "RELEASE_PUBLIC_KEY",
    "UpdateCancelled",
    "download_update",
]
