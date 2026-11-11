"""
SubprocessGramineSGXQuoteGenerator

FL client 在 host 运行，attestation 时启动轻量 gramine-sgx 子进程（dtrust_attest）
生成真实 DCAP quote，子进程结束后返回 SGXQuote 对象。

前置条件：
  - /dev/sgx_enclave 存在
  - gramine-sgx 可执行文件在 PATH
  - GRAMINE_ATTEST_DIR 目录下有：
      dtrust_attest.manifest.sgx   （由 05_setup_gramine_manifest.sh 生成）
      dtrust_attest.sig
"""

import asyncio
import os
import shutil
import struct
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.config import SGXConfig
from ..core.exceptions import AttestationError
from ..core.models import SGXQuote
from .sgx_quote import SGXQuoteGeneratorBase


# DCAP quote 字节偏移（sgx_quote3_t layout）
_HDR_LEN = 48
_OFF_MRENCLAVE = _HDR_LEN + 64
_OFF_MRSIGNER = _HDR_LEN + 128
_OFF_ISV_PROD_ID = _HDR_LEN + 256
_OFF_ISV_SVN = _HDR_LEN + 258
_OFF_REPORT_DATA = _HDR_LEN + 320
_OFF_SIG_DATA = _HDR_LEN + 384 + 4
_MIN_QUOTE_LEN = _HDR_LEN + 384

# 默认搜索路径
_DEFAULT_SEARCH = [
    os.path.expanduser("~/DTRUST/deploy"),
    "/root/DTRUST/deploy",
    os.path.join(os.path.dirname(__file__), "../../deploy"),
]


def _find_attest_dir(config: SGXConfig) -> Optional[str]:
    candidate = getattr(config, "gramine_attest_dir", "") or ""
    if candidate and Path(candidate, "dtrust_attest.manifest.sgx").exists():
        return candidate
    for d in _DEFAULT_SEARCH:
        if Path(d, "dtrust_attest.manifest.sgx").exists():
            return str(Path(d).resolve())
    return None


class SubprocessGramineSGXQuoteGenerator(SGXQuoteGeneratorBase):
    """生成真实 DCAP quote：在 host 进程中 spawn gramine-sgx 子进程。

    子进程 (quote_helper.py) 运行在 SGX enclave 内：
      - stdin  接收 64 字节 challenge（binary）
      - stdout 返回 raw DCAP quote bytes（binary）
    """

    def __init__(self, config: SGXConfig):
        self.config = config
        self._attest_dir: Optional[str] = None

    def _get_attest_dir(self) -> str:
        if self._attest_dir is None:
            self._attest_dir = _find_attest_dir(self.config)
        if self._attest_dir is None:
            raise AttestationError(
                "dtrust_attest.manifest.sgx 未找到。"
                "请先运行 05_setup_gramine_manifest.sh 生成 attestation manifest。"
            )
        return self._attest_dir

    def is_available(self) -> bool:
        if not self.config.enabled:
            return False
        if not (
            os.path.exists("/dev/sgx_enclave") or os.path.exists("/dev/sgx/enclave")
        ):
            return False
        if shutil.which("gramine-sgx") is None:
            return False
        try:
            self._get_attest_dir()
            return True
        except AttestationError:
            return False

    async def generate_quote(
        self,
        enclave_id: str,
        challenge_hash: bytes,
        user_data: Optional[bytes] = None,
    ) -> SGXQuote:
        attest_dir = self._get_attest_dir()
        report_data = challenge_hash[:64].ljust(64, b"\x00")

        proc = await asyncio.create_subprocess_exec(
            "gramine-sgx", "dtrust_attest",
            cwd=attest_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=report_data),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise AttestationError("gramine-sgx dtrust_attest 超时（60s）")

        if proc.returncode != 0:
            diag = stderr.decode(errors="replace").strip()
            raise AttestationError(
                f"gramine-sgx dtrust_attest 失败 (exit {proc.returncode}): {diag}"
            )

        quote = stdout
        if len(quote) < _MIN_QUOTE_LEN:
            diag = stderr.decode(errors="replace").strip()
            raise AttestationError(
                f"quote 太短 ({len(quote)} bytes < {_MIN_QUOTE_LEN}). stderr: {diag}"
            )

        return SGXQuote(
            quote=quote,
            signature=quote[_OFF_SIG_DATA:],
            report_data=report_data,
            enclave_measurement=quote[_OFF_MRENCLAVE : _OFF_MRENCLAVE + 32].hex(),
            signer_measurement=quote[_OFF_MRSIGNER : _OFF_MRSIGNER + 32].hex(),
            product_id=struct.unpack_from("<H", quote, _OFF_ISV_PROD_ID)[0],
            security_version=struct.unpack_from("<H", quote, _OFF_ISV_SVN)[0],
            timestamp=datetime.utcnow(),
        )
