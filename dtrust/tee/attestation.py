"""SGX attestation verification: mock, DCAP quote validation, and quote lifecycle."""

import asyncio
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from ..core.models import AttestationResult, AttestationStatus, TrustLevel, SGXQuote
from ..core.exceptions import AttestationError
from ..core.config import SGXConfig
from ..core.logging import get_logger, log_exception


logger = get_logger(__name__)


class AttestationServiceBase(ABC):
    """Abstract base class for attestation services."""
    
    @abstractmethod
    async def verify_attestation(
        self, 
        quote: SGXQuote, 
        nonce: str,
        node_id: str
    ) -> AttestationResult:
        """Verify SGX attestation and return result."""
        pass


class AttestationService(AttestationServiceBase):
    """Real Intel Attestation Service integration."""
    
    def __init__(self, config: SGXConfig):
        self.config = config
        self.ias_url = config.ias_url
        self.spid = config.spid
        self.primary_key = config.primary_key
    
    async def verify_attestation(
        self, 
        quote: SGXQuote, 
        nonce: str,
        node_id: str
    ) -> AttestationResult:
        """Verify attestation with Intel Attestation Service."""
        
        if not self.spid or not self.primary_key:
            raise AttestationError("Intel Attestation Service credentials not configured")
        
        try:
            # In a real implementation, this would:
            # 1. Submit quote to Intel IAS
            # 2. Verify the response signature
            # 3. Parse attestation verification report
            # 4. Check enclave measurements against whitelist
            
            # For now, simulate the process
            await asyncio.sleep(0.5)  # Simulate network delay
            
            # Basic quote validation
            if not self._validate_quote_structure(quote):
                return AttestationResult(
                    node_id=node_id,
                    status=AttestationStatus.FAILED,
                    confidence=0.0,
                    trust_level=TrustLevel.UNTRUSTED,
                    expiry_time=datetime.utcnow() + timedelta(hours=24),
                    error_message="Invalid quote structure",
                    nonce=nonce
                )
            
            # Simulate IAS verification
            confidence = self._calculate_confidence(quote)
            trust_level = self._determine_trust_level(quote, confidence)
            
            return AttestationResult(
                node_id=node_id,
                status=AttestationStatus.SUCCESS,
                confidence=confidence,
                trust_level=trust_level,
                quote_data={
                    "enclave_measurement": quote.enclave_measurement,
                    "signer_measurement": quote.signer_measurement,
                    "product_id": quote.product_id,
                    "security_version": quote.security_version
                },
                enclave_measurement=quote.enclave_measurement,
                signer_measurement=quote.signer_measurement,
                expiry_time=datetime.utcnow() + timedelta(hours=24),
                attestation_service="intel_ias",
                nonce=nonce
            )
            
        except Exception as e:
            logger.error(f"Attestation verification failed for {node_id}: {e}")
            return AttestationResult(
                node_id=node_id,
                status=AttestationStatus.ERROR,
                confidence=0.0,
                trust_level=TrustLevel.UNKNOWN,
                expiry_time=datetime.utcnow() + timedelta(hours=1),
                error_message=str(e),
                nonce=nonce
            )
    
    def _validate_quote_structure(self, quote: SGXQuote) -> bool:
        """Validate basic SGX quote structure."""
        try:
            # Check required fields
            if not quote.quote or not quote.signature:
                return False
            
            # Check quote length (basic validation)
            if len(quote.quote) < 432:  # Minimum SGX quote size
                return False
            
            # Check timestamp is recent
            if (datetime.utcnow() - quote.timestamp).total_seconds() > 300:  # 5 minutes
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_confidence(self, quote: SGXQuote) -> float:
        """Calculate confidence score based on quote properties."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on security version
        if quote.security_version >= 2:
            confidence += 0.1
        
        # Adjust based on quote freshness
        age_seconds = (datetime.utcnow() - quote.timestamp).total_seconds()
        if age_seconds < 60:  # Very fresh
            confidence += 0.1
        elif age_seconds > 180:  # Older than 3 minutes
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _determine_trust_level(self, quote: SGXQuote, confidence: float) -> TrustLevel:
        """Determine trust level based on quote and confidence."""
        if confidence >= 0.9:
            return TrustLevel.VERIFIED
        elif confidence >= 0.8:
            return TrustLevel.HIGH
        elif confidence >= 0.6:
            return TrustLevel.MEDIUM
        elif confidence >= 0.4:
            return TrustLevel.LOW
        else:
            return TrustLevel.UNTRUSTED


class MockAttestationService(AttestationServiceBase):
    """Mock attestation service for testing and development."""
    
    def __init__(self, config: SGXConfig):
        self.config = config
        
        # Predefined test scenarios
        self.test_scenarios = {
            "test_enclave_1": {"confidence": 0.95, "trust_level": TrustLevel.VERIFIED},
            "test_enclave_2": {"confidence": 0.85, "trust_level": TrustLevel.HIGH},
            "suspicious_enclave": {"confidence": 0.3, "trust_level": TrustLevel.UNTRUSTED},
            "failing_enclave": {"confidence": 0.0, "trust_level": TrustLevel.UNTRUSTED, "fail": True}
        }
    
    async def verify_attestation(
        self, 
        quote: SGXQuote, 
        nonce: str,
        node_id: str
    ) -> AttestationResult:
        """Mock attestation verification."""
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Extract enclave info from quote
        enclave_measurement = quote.enclave_measurement
        
        # Check for test scenarios
        scenario = None
        for test_enclave, config in self.test_scenarios.items():
            if test_enclave in enclave_measurement or test_enclave in node_id:
                scenario = config
                break
        
        # Default scenario
        if not scenario:
            scenario = {"confidence": 0.8, "trust_level": TrustLevel.HIGH}
        
        # Handle failure scenario
        if scenario.get("fail", False):
            return AttestationResult(
                node_id=node_id,
                status=AttestationStatus.FAILED,
                confidence=0.0,
                trust_level=TrustLevel.UNTRUSTED,
                expiry_time=datetime.utcnow() + timedelta(hours=1),
                error_message="Mock attestation failure",
                attestation_service="mock",
                nonce=nonce
            )
        
        # Successful attestation
        return AttestationResult(
            node_id=node_id,
            status=AttestationStatus.SUCCESS,
            confidence=scenario["confidence"],
            trust_level=scenario["trust_level"],
            quote_data={
                "enclave_measurement": quote.enclave_measurement,
                "signer_measurement": quote.signer_measurement,
                "product_id": quote.product_id,
                "security_version": quote.security_version
            },
            enclave_measurement=quote.enclave_measurement,
            signer_measurement=quote.signer_measurement,
            expiry_time=datetime.utcnow() + timedelta(hours=24),
            attestation_service="mock",
            nonce=nonce
        )
    
    def add_test_scenario(
        self, 
        enclave_id: str, 
        confidence: float, 
        trust_level: TrustLevel,
        should_fail: bool = False
    ):
        """Add a custom test scenario."""
        self.test_scenarios[enclave_id] = {
            "confidence": confidence,
            "trust_level": trust_level,
            "fail": should_fail
        }


class DCAPAttestationService(AttestationServiceBase):
    """Real DCAP quote verification via libsgx_dcap_quoteverify (sgx_qv_verify_quote).

    Relying-party counterpart to GramineSGXQuoteGenerator. Verifies a genuine
    hardware ECDSA quote: the signature chain against PCK collateral (fetched by
    the QPL from the configured PCCS, e.g. Aliyun's regional server), the platform
    TCB status, and that the quote's report_data binds the challenge nonce. Needs
    the Intel DCAP quote-verify library on the host; on a box without it (e.g. a
    macOS dev machine) the factory should not select this backend.
    """

    _LIB_NAMES = ("libsgx_dcap_quoteverify.so.1", "libsgx_dcap_quoteverify.so")

    # sgx_ql_qv_result_t -> (confidence, TrustLevel, accepted?). "Degraded-accept"
    # tiers (out-of-date / config / sw-hardening) pass with reduced trust; bad
    # signature / revoked / unspecified are hard failures.
    _POLICY = {
        0x0000: (1.0, TrustLevel.VERIFIED, True),    # OK
        0xA007: (0.9, TrustLevel.VERIFIED, True),    # SW_HARDENING_NEEDED
        0xA001: (0.8, TrustLevel.HIGH, True),        # CONFIG_NEEDED
        0xA002: (0.6, TrustLevel.MEDIUM, True),      # OUT_OF_DATE
        0xA003: (0.6, TrustLevel.MEDIUM, True),      # OUT_OF_DATE_CONFIG_NEEDED
        0xA008: (0.6, TrustLevel.MEDIUM, True),      # CONFIG_AND_SW_HARDENING_NEEDED
        0xA004: (0.0, TrustLevel.UNTRUSTED, False),  # INVALID_SIGNATURE
        0xA005: (0.0, TrustLevel.UNTRUSTED, False),  # REVOKED
        0xA006: (0.0, TrustLevel.UNTRUSTED, False),  # UNSPECIFIED
    }

    # sgx_quote3_t offsets (mirror GramineSGXQuoteGenerator)
    _OFF_MRENCLAVE = 48 + 64
    _OFF_MRSIGNER = 48 + 128
    _OFF_REPORT_DATA = 48 + 320
    _MIN_QUOTE_LEN = _OFF_REPORT_DATA + 64

    def __init__(self, config: SGXConfig):
        self.config = config
        self._lib = None

    def _load_lib(self):
        """Lazily dlopen + bind the quote-verify library (raises if absent)."""
        if self._lib is not None:
            return self._lib
        import ctypes

        lib = None
        last_err = None
        for name in self._LIB_NAMES:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError as e:
                last_err = e
        if lib is None:
            raise AttestationError(f"libsgx_dcap_quoteverify not available: {last_err}")

        lib.sgx_qv_get_quote_supplemental_data_size.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        lib.sgx_qv_get_quote_supplemental_data_size.restype = ctypes.c_uint32
        lib.sgx_qv_verify_quote.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32, ctypes.c_void_p, ctypes.c_long,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.sgx_qv_verify_quote.restype = ctypes.c_uint32
        self._lib = lib
        return lib

    def _verify_raw(self, quote: bytes):
        """Blocking ctypes verification. Returns (ret, qv_result, expiration_status).

        Collateral is passed as NULL so the QVL fetches it via the QPL/PCCS.
        Verification runs in untrusted mode (no QvE report requested).
        """
        import ctypes

        lib = self._load_lib()
        qbuf = (ctypes.c_uint8 * len(quote)).from_buffer_copy(quote)
        supp = ctypes.c_uint32(0)
        lib.sgx_qv_get_quote_supplemental_data_size(ctypes.byref(supp))
        sbuf = (ctypes.c_uint8 * supp.value)() if supp.value else None
        exp = ctypes.c_uint32(0)
        res = ctypes.c_uint32(0)
        now = int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds())
        ret = lib.sgx_qv_verify_quote(
            qbuf, len(quote), None, now,
            ctypes.byref(exp), ctypes.byref(res),
            None, supp.value, sbuf if sbuf else None,
        )
        return ret, res.value, exp.value

    @staticmethod
    def _expected_report_data(nonce: str, node_id: str) -> bytes:
        """Recompute report_data the way AttestationProtocol._hash_challenge does."""
        return hashlib.sha256(f"{nonce}:{node_id}".encode()).digest().ljust(64, b"\x00")

    async def verify_attestation(
        self,
        quote: SGXQuote,
        nonce: str,
        node_id: str,
    ) -> AttestationResult:
        """Cryptographically verify a real DCAP quote and bind it to the nonce."""
        try:
            raw = quote.quote
            if not raw or len(raw) < self._MIN_QUOTE_LEN:
                return self._failed(node_id, nonce, "Malformed quote (too short)")

            loop = asyncio.get_event_loop()
            ret, qv_result, _exp = await loop.run_in_executor(None, self._verify_raw, raw)
            if ret != 0x0000:
                return self._errored(node_id, nonce, f"sgx_qv_verify_quote ret=0x{ret:04x}")

            # Freshness: the quote's report_data must bind THIS challenge.
            embedded = raw[self._OFF_REPORT_DATA:self._OFF_REPORT_DATA + 64]
            if embedded != self._expected_report_data(nonce, node_id):
                return self._failed(node_id, nonce, "report_data does not bind challenge nonce")

            confidence, trust_level, accepted = self._POLICY.get(
                qv_result, (0.0, TrustLevel.UNTRUSTED, False)
            )
            mrenclave = raw[self._OFF_MRENCLAVE:self._OFF_MRENCLAVE + 32].hex()
            mrsigner = raw[self._OFF_MRSIGNER:self._OFF_MRSIGNER + 32].hex()

            if not accepted:
                return self._failed(
                    node_id, nonce,
                    f"Quote verification result 0x{qv_result:04x} not acceptable",
                    enclave_measurement=mrenclave, signer_measurement=mrsigner,
                )

            return AttestationResult(
                node_id=node_id,
                status=AttestationStatus.SUCCESS,
                confidence=confidence,
                trust_level=trust_level,
                quote_data={
                    "enclave_measurement": mrenclave,
                    "signer_measurement": mrsigner,
                    "qv_result": f"0x{qv_result:04x}",
                },
                enclave_measurement=mrenclave,
                signer_measurement=mrsigner,
                expiry_time=datetime.utcnow() + timedelta(hours=24),
                attestation_service="dcap",
                nonce=nonce,
            )
        except AttestationError as e:
            return self._errored(node_id, nonce, str(e))
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"DCAP verification failed for {node_id}: {e}")
            return self._errored(node_id, nonce, str(e))

    def _failed(self, node_id, nonce, msg, enclave_measurement=None, signer_measurement=None):
        return AttestationResult(
            node_id=node_id,
            status=AttestationStatus.FAILED,
            confidence=0.0,
            trust_level=TrustLevel.UNTRUSTED,
            enclave_measurement=enclave_measurement,
            signer_measurement=signer_measurement,
            expiry_time=datetime.utcnow() + timedelta(hours=1),
            error_message=msg,
            attestation_service="dcap",
            nonce=nonce,
        )

    def _errored(self, node_id, nonce, msg):
        return AttestationResult(
            node_id=node_id,
            status=AttestationStatus.ERROR,
            confidence=0.0,
            trust_level=TrustLevel.UNKNOWN,
            expiry_time=datetime.utcnow() + timedelta(hours=1),
            error_message=msg,
            attestation_service="dcap",
            nonce=nonce,
        )


def create_attestation_service(config: SGXConfig) -> AttestationServiceBase:
    """Factory: 根据 SGXConfig 选择 attestation service。

    优先级（从上到下）：
      mock_mode=True                           → MockAttestationService   (开发/CI)
      use_gramine_subprocess=True              → DCAPAttestationService   (host 验证真实 DCAP quote)
      use_gramine=True                         → DCAPAttestationService   (同上)
      其他                                     → AttestationService       (占位)
    """
    if config.mock_mode:
        return MockAttestationService(config)
    if getattr(config, "use_gramine_subprocess", False) or getattr(config, "use_gramine", False):
        return DCAPAttestationService(config)
    return AttestationService(config) 