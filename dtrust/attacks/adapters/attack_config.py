"""
Attack Configuration - Centralized configuration for all attack types.

This module provides a clean configuration interface for attacks without
requiring any modifications to the original DTRUST codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class AttackType(Enum):
    """Supported attack types."""
    NONE = "none"
    LABEL_FLIPPING = "label_flipping"
    BACKDOOR = "backdoor"
    SCALING = "scaling"
    KRUM = "krum"
    TRIM = "trim"
    ADAPTIVE = "adaptive"


@dataclass
class AttackConfig:
    """
    Configuration for attack behavior.

    This configuration is completely separate from DTRUST's main config,
    ensuring no side effects on the original system.

    Attributes:
        attack_type: Type of attack to perform
        enabled: Whether the attack is active
        num_classes: Number of classes for label flipping (default 10)
        target_label: Target label for backdoor attack
        trigger_size: Size of backdoor trigger pattern
        trigger_value: Pixel value for trigger (default 1.0)
        scale_factor: Scaling factor for scaling attack
        krum_f: Krum parameter (assumed Byzantine clients)
        trim_k: Trim parameter for trimmed mean
        adaptive_params: Parameters for adaptive attack (sigma, gamma, eta, V, Q)
        malicious_fraction: Fraction of clients that are malicious
        device: Torch device for computations
    """
    attack_type: AttackType = AttackType.NONE
    enabled: bool = False

    # Data poisoning parameters
    num_classes: int = 10
    target_label: int = 0
    trigger_size: int = 3
    trigger_value: float = 1.0

    # Model poisoning parameters
    scale_factor: float = 10.0
    krum_f: int = 1
    trim_k: int = 1

    # Adaptive attack parameters
    adaptive_params: Dict[str, float] = field(default_factory=lambda: {
        "sigma": 0.5,
        "gamma": 0.005,
        "eta": 0.01,
        "V": 10,
        "Q": 10,
    })

    # General parameters
    malicious_fraction: float = 0.2
    device: str = "cpu"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def label_flipping(cls, num_classes: int = 10, **kwargs) -> "AttackConfig":
        """Create a label flipping attack configuration."""
        return cls(
            attack_type=AttackType.LABEL_FLIPPING,
            enabled=True,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def backdoor(
        cls,
        target_label: int = 0,
        trigger_size: int = 3,
        trigger_value: float = 1.0,
        **kwargs
    ) -> "AttackConfig":
        """Create a backdoor attack configuration."""
        return cls(
            attack_type=AttackType.BACKDOOR,
            enabled=True,
            target_label=target_label,
            trigger_size=trigger_size,
            trigger_value=trigger_value,
            **kwargs
        )

    @classmethod
    def scaling(cls, scale_factor: float = 10.0, **kwargs) -> "AttackConfig":
        """Create a scaling attack configuration."""
        return cls(
            attack_type=AttackType.SCALING,
            enabled=True,
            scale_factor=scale_factor,
            **kwargs
        )

    @classmethod
    def krum(cls, f: int = 1, **kwargs) -> "AttackConfig":
        """Create a Krum attack configuration."""
        return cls(
            attack_type=AttackType.KRUM,
            enabled=True,
            krum_f=f,
            **kwargs
        )

    @classmethod
    def trim(cls, k: int = 1, **kwargs) -> "AttackConfig":
        """Create a Trim attack configuration."""
        return cls(
            attack_type=AttackType.TRIM,
            enabled=True,
            trim_k=k,
            **kwargs
        )

    @classmethod
    def adaptive(
        cls,
        sigma: float = 0.5,
        gamma: float = 0.005,
        eta: float = 0.01,
        V: int = 10,
        Q: int = 10,
        **kwargs
    ) -> "AttackConfig":
        """Create an adaptive attack configuration."""
        return cls(
            attack_type=AttackType.ADAPTIVE,
            enabled=True,
            adaptive_params={
                "sigma": sigma,
                "gamma": gamma,
                "eta": eta,
                "V": V,
                "Q": Q,
            },
            **kwargs
        )

    @property
    def is_data_poisoning(self) -> bool:
        """Check if this is a data poisoning attack."""
        return self.attack_type in [
            AttackType.LABEL_FLIPPING,
            AttackType.BACKDOOR,
        ]

    @property
    def is_model_poisoning(self) -> bool:
        """Check if this is a model poisoning attack."""
        return self.attack_type in [
            AttackType.SCALING,
            AttackType.KRUM,
            AttackType.TRIM,
            AttackType.ADAPTIVE,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attack_type": self.attack_type.value,
            "enabled": self.enabled,
            "num_classes": self.num_classes,
            "target_label": self.target_label,
            "trigger_size": self.trigger_size,
            "trigger_value": self.trigger_value,
            "scale_factor": self.scale_factor,
            "krum_f": self.krum_f,
            "trim_k": self.trim_k,
            "adaptive_params": self.adaptive_params,
            "malicious_fraction": self.malicious_fraction,
            "device": self.device,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackConfig":
        """Create from dictionary."""
        attack_type = data.get("attack_type", "none")
        if isinstance(attack_type, str):
            attack_type = AttackType(attack_type)

        return cls(
            attack_type=attack_type,
            enabled=data.get("enabled", False),
            num_classes=data.get("num_classes", 10),
            target_label=data.get("target_label", 0),
            trigger_size=data.get("trigger_size", 3),
            trigger_value=data.get("trigger_value", 1.0),
            scale_factor=data.get("scale_factor", 10.0),
            krum_f=data.get("krum_f", 1),
            trim_k=data.get("trim_k", 1),
            adaptive_params=data.get("adaptive_params", {}),
            malicious_fraction=data.get("malicious_fraction", 0.2),
            device=data.get("device", "cpu"),
            metadata=data.get("metadata", {}),
        )
