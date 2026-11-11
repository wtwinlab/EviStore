"""
Attestation -> Trust bridge.

Thin glue that runs an attestation via an ``AttestationProtocol`` and feeds the
resulting ``AttestationResult`` into a ``TrustManager`` through its existing
``update_attestation_result()`` entry point -- so real attestation_score enters
trust scoring WITHOUT changing TrustManager's interface.

Backend selection (mock vs real DCAP) lives entirely in the protocol's
``SGXConfig`` (``mock_mode`` vs ``use_gramine``). This glue is backend-agnostic,
so the mock path used by ML experiments is untouched: pass a mock-configured
protocol and nothing here changes; pass a Gramine/DCAP-configured one and the
real result flows in the same way.

Both helpers are duck-typed (no imports from ``tee``) to avoid coupling the trust
layer to the attestation layer.
"""

from typing import Any, Optional

from ..core.models import AttestationResult


async def attest_and_record(
    protocol: Any,
    trust_manager: Any,
    node_id: str,
    challenge: Optional[str] = None,
) -> AttestationResult:
    """Co-located path: run a full attestation (generate + verify) and record it.

    Suitable for the mock backend and for any host where generation and
    verification share a process. Returns the AttestationResult that was fed into
    trust scoring.
    """
    result = await protocol.process_attestation_request(node_id, challenge)
    trust_manager.update_attestation_result(node_id, result)
    return result


async def verify_quote_and_record(
    protocol: Any,
    trust_manager: Any,
    node_id: str,
    challenge: str,
    quote: Any,
) -> AttestationResult:
    """Split path: verify a quote the client already generated (in its enclave for
    the real backend) and record the result into trust scoring.

    This is the realistic real-deployment shape -- the weak SGX host only
    generates the quote; verification + trust + ML run on a separate machine.
    """
    result = await protocol.verify_quote(node_id, challenge, quote)
    trust_manager.update_attestation_result(node_id, result)
    return result
