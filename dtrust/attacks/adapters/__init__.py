"""
Attack Adapters for DTRUST - Zero side-effect integration.

These adapters wrap DTRUST components to inject attacks without modifying
the original codebase. The adapters follow the decorator/wrapper pattern
to intercept and modify behavior at runtime.

Usage:
    from dtrust.attacks.adapters import (
        AttackConfig,
        AttackingFlowerClient,
        AttackingDataLoader,
        create_attacking_client,
    )

    # Configure attack
    config = AttackConfig(
        attack_type="label_flipping",
        enabled=True,
        num_classes=10,
    )

    # Option 1: Wrap existing client
    attacking_client = AttackingFlowerClient(original_client, config)

    # Option 2: Create new attacking client from scratch
    attacking_client = create_attacking_client(
        node_id="malicious_node_1",
        enclave_id="enclave_1",
        attack_config=config,
        ...
    )
"""

from dtrust.attacks.adapters.attack_config import AttackConfig, AttackType
from dtrust.attacks.adapters.client_adapter import (
    AttackingFlowerClient,
    AttackingDataLoader,
    create_attacking_client,
)
from dtrust.attacks.adapters.model_poisoning_adapter import (
    ModelPoisoningAdapter,
    apply_model_poisoning,
)

__all__ = [
    # Configuration
    "AttackConfig",
    "AttackType",
    # Client adapters
    "AttackingFlowerClient",
    "AttackingDataLoader",
    "create_attacking_client",
    # Model poisoning
    "ModelPoisoningAdapter",
    "apply_model_poisoning",
]
