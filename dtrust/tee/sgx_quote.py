"""
SGX Quote Generation components.
Supports both real SGX hardware and mock implementation for testing.
"""

import hashlib
import secrets
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any
import struct

from ..core.models import SGXQuote
from ..core.exceptions import AttestationError
from ..core.config import SGXConfig


class SGXQuoteGeneratorBase(ABC):
    """Abstract base class for SGX quote generators."""
    
    @abstractmethod
    async def generate_quote(
        self, 
        enclave_id: str, 
        challenge_hash: bytes,
        user_data: Optional[bytes] = None
    ) -> SGXQuote:
        """Generate SGX quote for attestation."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if SGX is available."""
        pass


class SGXQuoteGenerator(SGXQuoteGeneratorBase):
    """Real SGX quote generator using Intel SGX SDK."""
    
    def __init__(self, config: SGXConfig):
        self.config = config
        self._sgx_available = self._check_sgx_availability()
    
    def _check_sgx_availability(self) -> bool:
        """Check if SGX hardware and drivers are available."""
        try:
            # Try to import SGX SDK components
            # This would normally import from sgx_sdk or similar
            # For now, we'll simulate the check
            import os
            return os.path.exists("/dev/sgx_enclave") or os.path.exists("/dev/sgx/enclave")
        except ImportError:
            return False
    
    async def generate_quote(
        self, 
        enclave_id: str, 
        challenge_hash: bytes,
        user_data: Optional[bytes] = None
    ) -> SGXQuote:
        """Generate real SGX quote."""
        if not self.is_available():
            raise AttestationError("SGX hardware not available")
        
        try:
            # In a real implementation, this would:
            # 1. Load the enclave
            # 2. Create report with challenge_hash as report_data
            # 3. Get quote from Quoting Enclave
            # 4. Return structured quote data
            
            # For now, we'll create a realistic mock structure
            # that would be replaced with actual SGX SDK calls
            
            report_data = challenge_hash[:64].ljust(64, b'\x00')
            
            # Simulate SGX quote structure
            quote_body = self._create_quote_body(enclave_id, report_data)
            signature = self._create_quote_signature(quote_body)
            
            return SGXQuote(
                quote=quote_body,
                signature=signature,
                report_data=report_data,
                enclave_measurement=self._get_enclave_measurement(enclave_id),
                signer_measurement=self._get_signer_measurement(enclave_id),
                product_id=1,
                security_version=1,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            raise AttestationError(f"Failed to generate SGX quote: {e}")
    
    def is_available(self) -> bool:
        """Check if SGX is available."""
        return self._sgx_available and self.config.enabled
    
    def _create_quote_body(self, enclave_id: str, report_data: bytes) -> bytes:
        """Create SGX quote body structure."""
        # This would normally be handled by SGX SDK
        # Simplified structure for demonstration
        version = struct.pack('<H', 3)  # Quote version
        sign_type = struct.pack('<H', 1)  # EPID signature
        
        # EPID Group ID (4 bytes)
        epid_group_id = struct.pack('<I', 0x12345678)
        
        # QE Security Version (2 bytes)
        qe_svn = struct.pack('<H', 1)
        
        # PCE Security Version (2 bytes) 
        pce_svn = struct.pack('<H', 1)
        
        # Reserved (4 bytes)
        reserved = b'\x00' * 4
        
        # Basename (32 bytes)
        basename = hashlib.sha256(enclave_id.encode()).digest()
        
        # Report body would be created by SGX hardware
        report_body = self._create_report_body(enclave_id, report_data)
        
        return version + sign_type + epid_group_id + qe_svn + pce_svn + reserved + basename + report_body
    
    def _create_report_body(self, enclave_id: str, report_data: bytes) -> bytes:
        """Create SGX report body."""
        # CPU SVN (16 bytes)
        cpu_svn = secrets.token_bytes(16)
        
        # Miscellaneous select (4 bytes)
        misc_select = struct.pack('<I', 0)
        
        # Reserved (28 bytes)
        reserved1 = b'\x00' * 28
        
        # Attributes (16 bytes)
        attributes = struct.pack('<QQ', 0x0000000000000007, 0x0000000000000000)
        
        # Enclave measurement (32 bytes)
        mr_enclave = hashlib.sha256(enclave_id.encode()).digest()
        
        # Reserved (32 bytes)
        reserved2 = b'\x00' * 32
        
        # Signer measurement (32 bytes)
        mr_signer = hashlib.sha256(f"signer_{enclave_id}".encode()).digest()
        
        # Reserved (96 bytes)
        reserved3 = b'\x00' * 96
        
        # Product ID (2 bytes)
        isv_prod_id = struct.pack('<H', 1)
        
        # Security version (2 bytes)
        isv_svn = struct.pack('<H', 1)
        
        # Reserved (60 bytes)
        reserved4 = b'\x00' * 60
        
        # Report data (64 bytes)
        report_data_padded = report_data[:64].ljust(64, b'\x00')
        
        return (cpu_svn + misc_select + reserved1 + attributes + 
                mr_enclave + reserved2 + mr_signer + reserved3 + 
                isv_prod_id + isv_svn + reserved4 + report_data_padded)
    
    def _create_quote_signature(self, quote_body: bytes) -> bytes:
        """Create quote signature."""
        # In real SGX, this would be EPID signature from Intel
        # For simulation, create a deterministic signature
        return hashlib.sha256(quote_body + b"sgx_signature").digest()
    
    def _get_enclave_measurement(self, enclave_id: str) -> str:
        """Get enclave measurement (MRENCLAVE)."""
        return hashlib.sha256(enclave_id.encode()).hexdigest()
    
    def _get_signer_measurement(self, enclave_id: str) -> str:
        """Get signer measurement (MRSIGNER)."""
        return hashlib.sha256(f"signer_{enclave_id}".encode()).hexdigest()


class MockSGXQuoteGenerator(SGXQuoteGeneratorBase):
    """Mock SGX quote generator for testing and development."""
    
    def __init__(self, config: SGXConfig):
        self.config = config
        self._known_enclaves = {
            "test_enclave_1": {
                "measurement": "a" * 64,
                "signer": "b" * 64,
                "product_id": 1,
                "security_version": 1
            },
            "test_enclave_2": {
                "measurement": "c" * 64, 
                "signer": "d" * 64,
                "product_id": 2,
                "security_version": 1
            }
        }
    
    async def generate_quote(
        self, 
        enclave_id: str, 
        challenge_hash: bytes,
        user_data: Optional[bytes] = None
    ) -> SGXQuote:
        """Generate mock SGX quote for testing."""
        
        # Simulate some processing time
        import asyncio
        await asyncio.sleep(0.1)
        
        # Get or create enclave info
        enclave_info = self._known_enclaves.get(enclave_id, {
            "measurement": hashlib.sha256(enclave_id.encode()).hexdigest(),
            "signer": hashlib.sha256(f"signer_{enclave_id}".encode()).hexdigest(),
            "product_id": 1,
            "security_version": 1
        })
        
        # Create mock quote data
        report_data = challenge_hash[:64].ljust(64, b'\x00')
        quote_data = self._create_mock_quote(enclave_id, report_data, enclave_info)
        signature = hashlib.sha256(quote_data + b"mock_signature").digest()
        
        return SGXQuote(
            quote=quote_data,
            signature=signature,
            report_data=report_data,
            enclave_measurement=enclave_info["measurement"],
            signer_measurement=enclave_info["signer"],
            product_id=enclave_info["product_id"],
            security_version=enclave_info["security_version"],
            timestamp=datetime.utcnow()
        )
    
    def is_available(self) -> bool:
        """Mock is always available when enabled."""
        return self.config.enabled
    
    def _create_mock_quote(self, enclave_id: str, report_data: bytes, enclave_info: Dict[str, Any]) -> bytes:
        """Create mock quote data."""
        # Create a realistic-looking quote structure
        header = b"MOCK_SGX_QUOTE_V3"
        enclave_data = enclave_id.encode().ljust(32, b'\x00')
        measurement = bytes.fromhex(enclave_info["measurement"])[:32]
        signer = bytes.fromhex(enclave_info["signer"])[:32]
        
        return header + enclave_data + measurement + signer + report_data
    
    def add_test_enclave(self, enclave_id: str, measurement: str, signer: str, 
                        product_id: int = 1, security_version: int = 1):
        """Add a test enclave for mock scenarios."""
        self._known_enclaves[enclave_id] = {
            "measurement": measurement,
            "signer": signer,
            "product_id": product_id,
            "security_version": security_version
        }


class GramineSGXQuoteGenerator(SGXQuoteGeneratorBase):
    """Real DCAP quote generator backed by Gramine's /dev/attestation pseudo-files.

    Unlike ``SGXQuoteGenerator`` (a placeholder that fabricates measurements), this
    produces a genuine hardware-signed quote. It only works when the process runs
    INSIDE a Gramine SGX enclave whose manifest sets::

        sgx.remote_attestation = "dcap"

    Gramine then exposes the SGX quoting flow as files: write the 64-byte
    report_data, read back the ECDSA/DCAP quote. The enclave is fixed (it is the
    one we are running in), so ``enclave_id`` is informational only; the real
    MRENCLAVE/MRSIGNER are parsed out of the quote the hardware returns.

    On a non-SGX host (e.g. a macOS dev box) ``is_available()`` returns False and
    the factory should fall back to the mock generator.

    NOTE: this produces the quote. Verifying it for real is a separate concern --
    pair this with a DCAP verifier (libsgx-dcap-quote-verify / Azure MAA) on the
    server side rather than the placeholder ``AttestationService``.
    """

    # --- Gramine pseudo-file interface (stable across Gramine releases) ---
    _ATTEST_DIR = "/dev/attestation"
    _TYPE_FILE = _ATTEST_DIR + "/attestation_type"
    _USER_REPORT_DATA = _ATTEST_DIR + "/user_report_data"
    _QUOTE_FILE = _ATTEST_DIR + "/quote"

    # --- Byte offsets into a DCAP quote (sgx_quote3_t = 48B header + 384B report body) ---
    _HDR_LEN = 48                       # sgx_quote_header_t
    _OFF_MRENCLAVE = _HDR_LEN + 64      # sgx_report_body_t.mr_enclave  (32B)
    _OFF_MRSIGNER = _HDR_LEN + 128      # sgx_report_body_t.mr_signer   (32B)
    _OFF_ISV_PROD_ID = _HDR_LEN + 256   # sgx_report_body_t.isv_prod_id (u16)
    _OFF_ISV_SVN = _HDR_LEN + 258       # sgx_report_body_t.isv_svn     (u16)
    _OFF_REPORT_DATA = _HDR_LEN + 320   # sgx_report_body_t.report_data (64B)
    _OFF_SIG_DATA = _HDR_LEN + 384 + 4  # after report body + 4B signature_data_len
    _MIN_QUOTE_LEN = _HDR_LEN + 384     # header + full report body

    def __init__(self, config: SGXConfig):
        self.config = config

    def is_available(self) -> bool:
        """True only when running inside a Gramine enclave with DCAP enabled."""
        if not self.config.enabled:
            return False
        try:
            with open(self._TYPE_FILE) as f:
                return f.read().strip() == "dcap"
        except OSError:
            # File absent -> not inside a Gramine SGX enclave (e.g. plain Linux / Mac).
            return False

    async def generate_quote(
        self,
        enclave_id: str,
        challenge_hash: bytes,
        user_data: Optional[bytes] = None,
    ) -> SGXQuote:
        """Produce a real DCAP quote binding ``challenge_hash`` into report_data."""
        if not self.is_available():
            raise AttestationError(
                "Gramine DCAP attestation unavailable: process is not running inside "
                'an SGX enclave with sgx.remote_attestation = "dcap" in its manifest.'
            )

        # report_data carries the freshness nonce (challenge hash); SGX fixes it at 64B.
        report_data = challenge_hash[:64].ljust(64, b"\x00")

        try:
            quote = self._read_quote(report_data)
        except OSError as e:
            raise AttestationError(f"Failed to obtain Gramine DCAP quote: {e}")

        if len(quote) < self._MIN_QUOTE_LEN:
            raise AttestationError(
                f"Malformed DCAP quote: {len(quote)} bytes (< {self._MIN_QUOTE_LEN})"
            )

        return SGXQuote(
            quote=quote,
            signature=quote[self._OFF_SIG_DATA:],          # ECDSA signature_data section
            report_data=report_data,
            enclave_measurement=quote[self._OFF_MRENCLAVE:self._OFF_MRENCLAVE + 32].hex(),
            signer_measurement=quote[self._OFF_MRSIGNER:self._OFF_MRSIGNER + 32].hex(),
            product_id=struct.unpack_from("<H", quote, self._OFF_ISV_PROD_ID)[0],
            security_version=struct.unpack_from("<H", quote, self._OFF_ISV_SVN)[0],
            timestamp=datetime.utcnow(),
        )

    def _read_quote(self, report_data: bytes) -> bytes:
        """Prime report_data then read the quote. Pseudo-file IO is local & sub-ms."""
        # Writing user_report_data is what makes the next quote bind our nonce.
        with open(self._USER_REPORT_DATA, "wb") as f:
            f.write(report_data)
        with open(self._QUOTE_FILE, "rb") as f:
            return f.read()


def create_quote_generator(config: SGXConfig) -> SGXQuoteGeneratorBase:
    """Factory: 根据 SGXConfig 选择 quote generator。

    优先级（从上到下）：
      mock_mode=True               → MockSGXQuoteGenerator        (开发/CI)
      use_gramine_subprocess=True  → SubprocessGramineSGXQuoteGenerator  (推荐生产)
      use_gramine=True             → GramineSGXQuoteGenerator     (整体在 enclave 内)
      其他                         → SGXQuoteGenerator             (占位，非真实硬件)
    """
    if config.mock_mode:
        return MockSGXQuoteGenerator(config)
    if getattr(config, "use_gramine_subprocess", False):
        from .subprocess_quote import SubprocessGramineSGXQuoteGenerator
        return SubprocessGramineSGXQuoteGenerator(config)
    if getattr(config, "use_gramine", False):
        return GramineSGXQuoteGenerator(config)
    return SGXQuoteGenerator(config)