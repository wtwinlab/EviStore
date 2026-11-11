"""Attack implementations used to evaluate trust-conditioned aggregation.

Label flipping, backdoor triggers, and Byzantine attacks optimized against
Krum, trimmed mean, and adaptive defenses.
"""

from dtrust.attacks.label_flipping import (
    flip_labels_l_to_m_minus_l_minus_1,
    maybe_label_flip_batch,
)
from dtrust.attacks.scaling_backdoor import (
    TriggerSpec,
    apply_corner_square_trigger,
    make_backdoor_batch,
    scale_delta,
)
from dtrust.attacks.krum_attack import (
    KrumAttackResult,
    krum_attack,
    krum_attack_full_knowledge,
)
from dtrust.attacks.trim_attack import (
    TrimAttackResult,
    trim_attack,
    trim_attack_optimized,
    trim_attack_full_knowledge,
)
from dtrust.attacks.adaptive_attack import (
    AdaptiveAttackResult,
    adaptive_attack,
    adaptive_attack_full_knowledge,
)

__all__ = [
    # Label flipping
    "flip_labels_l_to_m_minus_l_minus_1",
    "maybe_label_flip_batch",
    # Scaling/Backdoor
    "TriggerSpec",
    "apply_corner_square_trigger",
    "make_backdoor_batch",
    "scale_delta",
    # Krum attack
    "KrumAttackResult",
    "krum_attack",
    "krum_attack_full_knowledge",
    # Trim attack
    "TrimAttackResult",
    "trim_attack",
    "trim_attack_optimized",
    "trim_attack_full_knowledge",
    # Adaptive attack
    "AdaptiveAttackResult",
    "adaptive_attack",
    "adaptive_attack_full_knowledge",
]
