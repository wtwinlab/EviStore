"""
Vectorized (vmap) client-parallel local training for FL simulation.

The sequential harnesses train N clients one at a time (plain SGD, fixed
local steps, no momentum) with batch sizes of 32-64 — on a modern GPU this
leaves utilization in the single digits because the wall-clock is dominated
by the Python client loop, per-step host-to-device copies, and per-round
model rebuilds. Since every client runs the *same* pure-functional SGD
recipe, the whole round is one batched computation: stack the per-client
parameters into [N, ...] tensors and drive them with
``torch.func.vmap(grad(loss))``.

Semantics preserved relative to the sequential path:
- plain SGD (functional ``p - lr*g``), CrossEntropyLoss (mean), fixed steps;
- delta = final - initial flat params in ``model.parameters()`` order;
- DataLoader-equivalent sampling: shuffled epochs with drop_last for clients
  with >= batch_size samples; one smaller batch per step for tiny clients
  (padded to B with zero loss-weight so the masked mean equals the true mean);
- data poisoning (label-flip full/partial, backdoor trigger) applied to the
  assembled batch, exactly where the sequential batch_transform ran.

Not bit-identical to the sequential path (RNG streams differ); equivalence
is established by tests/test_vectorized_equivalence.py (identical fixed
batches -> identical deltas) plus end-to-end distributional checks.

Data sampling and poisoning randomness come from per-client Philox streams
keyed (seed, cid), so WHICH samples each client trains on is invariant to
--client_chunk. (Bitwise outputs may still differ across chunk sizes: conv
kernels can pick different fp reduction orders for different batch shapes.)
Dropout uses vmap(randomness='different') and is the only chunk-sensitive
randomness (resnet20gn and the default SimpleNN have no dropout).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from dtrust.attacks.scaling_backdoor import TriggerSpec


def supports_fast_path(
    model: nn.Module,
    local_epochs: Optional[int] = None,
    use_momentum: bool = False,
) -> Tuple[bool, str]:
    """Gate for the vmap fast path; (False, reason) means fall back to the
    sequential loop with a warning, never an error."""
    if local_epochs is not None:
        return False, "--local_epochs (epoch mode) is only implemented on the sequential path"
    if use_momentum:
        return False, "--use_momentum needs optimizer state the functional fast path does not carry"
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            return False, (
                "model contains BatchNorm: running-stat buffers are per-client mutable "
                "state that vmap cannot batch (and this harness never aggregates them); "
                "use the GroupNorm variant (e.g. resnet20gn_cifar10) for the fast path"
            )
    return True, ""


class ClientDataStore:
    """GPU-resident ragged per-client dataset with reproducible sampling.

    The full training set is moved to ``device`` once; per-client index sets
    are held in CSR-style storage. Batch *indices* are drawn on the CPU from
    per-client Philox streams keyed (seed, cid) — reproducible and invariant
    to client chunking — then gathered on-device.
    """

    def __init__(
        self,
        x_all: torch.Tensor,
        y_all: torch.Tensor,
        splits: Sequence[np.ndarray],
        batch_size: int,
        seed: int,
        device: torch.device,
    ):
        self.device = device
        self.batch_size = int(batch_size)
        splits = [np.asarray(s, dtype=np.int64) for s in splits]
        # Keep only samples some client actually owns (e.g. FEMNIST natural
        # partitions select ~100 writers out of ~3400), remapping indices.
        used = np.unique(np.concatenate(splits)) if splits else np.array([], dtype=np.int64)
        if len(used) < len(x_all):
            remap = np.full(len(x_all), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            x_all, y_all = x_all[used], y_all[used]
            splits = [remap[s] for s in splits]
        self.x_all = x_all.to(device)
        self.y_all = y_all.to(device)
        self.splits = splits
        self.num_clients = len(self.splits)
        for cid, s in enumerate(self.splits):
            if len(s) == 0:
                raise ValueError(f"ClientDataStore: client {cid} has an empty partition")
        self.rngs = [np.random.default_rng([int(seed), cid]) for cid in range(self.num_clients)]
        # Per-client shuffled epoch permutation + cursor (drop_last semantics).
        self._perm = [rng.permutation(len(s)) for rng, s in zip(self.rngs, self.splits)]
        self._cursor = [0] * self.num_clients

    def client_sizes(self) -> List[int]:
        return [len(s) for s in self.splits]

    def _next_batch_indices(self, cid: int) -> Tuple[np.ndarray, int]:
        """Global dataset indices for one client's next batch, plus the number
        of real (unpadded) samples in it."""
        s = self.splits[cid]
        n = len(s)
        B = self.batch_size
        if n >= B:
            if self._cursor[cid] + B > n:  # drop_last: discard the partial tail
                self._perm[cid] = self.rngs[cid].permutation(n)
                self._cursor[cid] = 0
            local = self._perm[cid][self._cursor[cid] : self._cursor[cid] + B]
            self._cursor[cid] += B
            return s[local], B
        # Tiny client (n < B): one batch of all n samples, freshly shuffled,
        # padded to B by repeating the first sample with zero loss-weight.
        local = self.rngs[cid].permutation(n)
        idx = s[local]
        pad = np.full(B - n, idx[0], dtype=np.int64)
        return np.concatenate([idx, pad]), n

    def sample_step(self, client_ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """One SGD step's batches for a chunk of clients.

        Returns (batch_idx [C, B] int64 on device, w [C, B] float32 on device),
        where w masks padding for tiny clients (all-ones otherwise).
        """
        C, B = len(client_ids), self.batch_size
        idx = np.empty((C, B), dtype=np.int64)
        w = np.ones((C, B), dtype=np.float32)
        for row, cid in enumerate(client_ids):
            idx[row], n_real = self._next_batch_indices(cid)
            if n_real < B:
                w[row, n_real:] = 0.0
        return (
            torch.from_numpy(idx).to(self.device),
            torch.from_numpy(w).to(self.device),
        )

    def gather(self, batch_idx: torch.Tensor, w: torch.Tensor):
        """Materialize the sampled batches: x [C, B, ...], y [C, B], w [C, B]."""
        return self.x_all[batch_idx], self.y_all[batch_idx], w


@dataclass
class PoisonSpec:
    """Data-poisoning applied inside the fast path, mirroring the sequential
    per-client batch_transform semantics."""

    attack: str  # "none" | "lf" | "backdoor"
    num_classes: int
    flip_frac: float = 1.0  # lf: fraction of labels flipped (Bernoulli, per client RNG)
    target_label: int = 0  # backdoor
    trigger: Optional[TriggerSpec] = None
    trigger_frac: float = 0.1  # backdoor: fraction of the batch triggered


def poison_batches(
    x: torch.Tensor,
    y: torch.Tensor,
    mal_rows: Sequence[int],
    spec: PoisonSpec,
    rngs: Sequence[np.random.Generator],
    client_ids: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Poison the malicious rows of an assembled [C, B, ...] batch in place.

    ``x``/``y`` are fresh gather copies, so in-place edits never touch the
    backing dataset. Poison randomness comes from the *client's* RNG stream
    (rngs[client_ids[row]]), keeping results chunk-invariant.
    """
    if spec.attack == "none" or not mal_rows:
        return x, y
    B = y.shape[1]
    if spec.attack == "lf":
        rows = torch.as_tensor(list(mal_rows), device=y.device)
        flipped = (spec.num_classes - 1) - y[rows]
        if spec.flip_frac >= 1.0:
            y[rows] = flipped
        else:
            for i, row in enumerate(mal_rows):
                m = rngs[client_ids[row]].random(B) < spec.flip_frac
                mask = torch.from_numpy(m).to(y.device)
                y[row][mask] = flipped[i][mask]
        return x, y
    if spec.attack == "backdoor":
        trigger = spec.trigger or TriggerSpec()
        s = trigger.size
        k = max(1, int(B * spec.trigger_frac))
        flat = x.dim() == 3  # [C, B, D] flattened image (fashion 784 = 28x28); else spatial [C,B,ch,H,W]
        xv = x.view(x.shape[0], x.shape[1], int(round(float(x.shape[-1]) ** 0.5)),
                    int(round(float(x.shape[-1]) ** 0.5))) if flat else x
        h, w_ = xv.shape[-2], xv.shape[-1]
        for row in mal_rows:
            pick = rngs[client_ids[row]].permutation(B)[:k]
            pick_t = torch.from_numpy(pick).to(x.device)
            if flat:
                xv[row, pick_t, h - s : h, w_ - s : w_] = trigger.value
            else:
                x[row, pick_t, :, h - s : h, w_ - s : w_] = trigger.value
            y[row, pick_t] = int(spec.target_label)
        return x, y
    raise ValueError(f"Unknown poison attack: {spec.attack}")


def augment_image_batch(
    x: torch.Tensor, pad: int = 4, gen: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Standard CIFAR-style augmentation on a [C, B, ch, H, W] batch, fully on
    device and vectorized (no Python loop): per-sample random horizontal flip +
    random crop (reflect-pad by ``pad``, crop back to H×W).

    Applied to the assembled batch inside the fast path (before poisoning), so
    it needs no per-client state. Randomness is training noise (like dropout):
    reproducible for a fixed seeded ``gen`` but not guaranteed bit-identical
    across ``client_chunk`` sizes.
    """
    lead = x.shape[:-3]
    ch, H, W = x.shape[-3:]
    xf = x.reshape(-1, ch, H, W)
    M = xf.shape[0]
    dev = xf.device

    # random horizontal flip
    flip = torch.rand(M, device=dev, generator=gen) < 0.5
    xf = torch.where(flip.view(M, 1, 1, 1), xf.flip(-1), xf)

    # random crop via reflect pad + per-sample offset gather
    xp = F.pad(xf, (pad, pad, pad, pad), mode="reflect")  # [M, ch, H+2p, W+2p]
    oy = torch.randint(0, 2 * pad + 1, (M,), device=dev, generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (M,), device=dev, generator=gen)
    row_idx = (oy.view(M, 1) + torch.arange(H, device=dev).view(1, H))  # [M, H]
    row_idx = row_idx.view(M, 1, H, 1).expand(M, ch, H, W + 2 * pad)
    xp = torch.gather(xp, 2, row_idx)  # crop rows -> [M, ch, H, W+2p]
    col_idx = (ox.view(M, 1) + torch.arange(W, device=dev).view(1, W))  # [M, W]
    col_idx = col_idx.view(M, 1, 1, W).expand(M, ch, H, W)
    xp = torch.gather(xp, 3, col_idx)  # crop cols -> [M, ch, H, W]
    return xp.reshape(*lead, ch, H, W)


class VmapTrainer:
    """Trains all clients of an FL round in one vectorized computation.

    Holds a template module (used only for its structure via functional_call)
    and derives per-client flat deltas identical in layout to the sequential
    ``torch.cat([p.view(-1) for p in model.parameters()])`` convention.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 augment: bool = False, augment_pad: int = 4, momentum: float = 0.0):
        self.device = device
        self.model = model.to(device)
        self.model.train()  # dropout active, as in the sequential path
        self.names = [n for n, _ in self.model.named_parameters()]
        self.buffers = {n: b.detach() for n, b in self.model.named_buffers()}
        # Optional CIFAR-style train augmentation (crop+flip) on 4D image data.
        self.augment = augment
        self.augment_pad = augment_pad
        # Local SGD momentum for the within-round steps (velocity resets each
        # round, so no global-momentum state crosses aggregation). 0.0 = plain
        # SGD (default, bit-identical to before); 0.9 accelerates convergence
        # on deep models (ResNet) where plain SGD is slow.
        self.momentum = momentum

        def loss_fn(params, x_b, y_b, w_b):
            logits = functional_call(self.model, (params, self.buffers), (x_b,))
            per_sample = F.cross_entropy(logits, y_b, reduction="none")
            return (per_sample * w_b).sum() / w_b.sum()

        self._grad_all = vmap(
            grad(loss_fn), in_dims=(0, 0, 0, 0), randomness="different"
        )

    def _train_chunk(
        self,
        p0: Dict[str, torch.Tensor],
        client_ids: Sequence[int],
        store: ClientDataStore,
        steps: int,
        lr: float,
        mal_set: frozenset,
        poison: PoisonSpec,
    ) -> torch.Tensor:
        C = len(client_ids)
        params = {k: v.unsqueeze(0).expand(C, *v.shape) for k, v in p0.items()}
        mal_rows = [row for row, cid in enumerate(client_ids) if cid in mal_set]
        vel = None if self.momentum == 0.0 else {k: torch.zeros_like(v) for k, v in params.items()}
        for _ in range(steps):
            batch_idx, w = store.sample_step(client_ids)
            x, y, w = store.gather(batch_idx, w)
            if self.augment and x.dim() == 5 and x.shape[-1] > self.augment_pad:
                x = augment_image_batch(x, self.augment_pad)
            x, y = poison_batches(x, y, mal_rows, poison, store.rngs, client_ids)
            grads = self._grad_all(params, x, y, w)
            if vel is None:
                params = {k: params[k] - lr * grads[k] for k in params}
            else:
                vel = {k: self.momentum * vel[k] + grads[k] for k in params}
                params = {k: params[k] - lr * vel[k] for k in params}
        return torch.cat(
            [(params[k] - p0[k].unsqueeze(0)).reshape(C, -1) for k in self.names],
            dim=1,
        )

    def round_deltas(
        self,
        global_params: Dict[str, torch.Tensor],
        store: ClientDataStore,
        steps: int,
        lr: float,
        malicious_ids: Sequence[int] = (),
        poison: Optional[PoisonSpec] = None,
        client_chunk: int = 0,
    ) -> torch.Tensor:
        """One FL round of local training for every client in ``store``.

        Args:
            global_params: detached named parameters of the global model.
            client_chunk: max clients per vmap call (0 = all at once); on CUDA
                OOM the chunk is halved and retried, degrading gracefully.

        Returns:
            [num_clients, D] flat deltas on ``store.device``.
        """
        p0 = {k: v.detach().to(self.device) for k, v in global_params.items()}
        mal_set = frozenset(int(m) for m in malicious_ids)
        poison = poison or PoisonSpec("none", num_classes=1)
        N = store.num_clients
        chunk = client_chunk if client_chunk and client_chunk > 0 else N
        deltas: List[torch.Tensor] = []
        start = 0
        while start < N:
            ids = list(range(start, min(start + chunk, N)))
            try:
                deltas.append(
                    self._train_chunk(p0, ids, store, steps, lr, mal_set, poison)
                )
                start += len(ids)
            except torch.cuda.OutOfMemoryError:
                if chunk <= 1:
                    raise
                torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)
                print(f"[fast_gpu] CUDA OOM — retrying with client_chunk={chunk}")
        return torch.cat(deltas, dim=0)
