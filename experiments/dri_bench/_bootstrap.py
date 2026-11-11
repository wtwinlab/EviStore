"""Import bootstrap so the audit runs without the heavy DTRUST stack.

The audit pipeline only needs torch-free leaf modules: the SPRT LLR math
(``dtrust.trust._llr``), the trust engine (``dtrust.attack_experiments.
trust_model.engine``), the synthetic simulator, and the sealed store
(``dtrust.store``). But importing any ``dtrust.*`` submodule normally runs
``dtrust/__init__.py`` and ``dtrust/trust/__init__.py``, both of which import
torch/flwr transitively. On a laptop without torch that explodes.

We pre-register lightweight package stubs (correct ``__path__``, no body) for
the two packages whose ``__init__`` is heavy, so Python resolves their leaf
submodules without executing the heavy init. This is a no-op effect on the
math (we still load the real leaf files) and behaves identically on the server
where torch is installed -- it just skips an init we never use here.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _stub_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]          # mark as a package for submodule lookup
    mod.__package__ = name
    sys.modules[name] = mod


def bootstrap() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _stub_pkg("dtrust", ROOT / "dtrust")
    _stub_pkg("dtrust.trust", ROOT / "dtrust" / "trust")
