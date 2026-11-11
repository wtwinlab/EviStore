# DTRUST

This repository contains the implementation of **DTRUST**, an evidence-driven
dynamic trust-management system for federated learning, and the scripts that
drive its experiments. DTRUST maintains per-client trust as an evolving
coordinator state, seals the evidence and decisions produced under a deployed
policy, and lets an operator later revise the policy from the sealed view
under a determinacy-based evidence contract.

---

## 1. Structure

```
DTRUST/
├── README.md
├── LICENSE                        # Apache License 2.0
├── requirements.txt               # core dependencies
├── setup.py                       # installable package
├── dtrust/                        # core library
│   ├── core/                      # config, logging, exceptions, base models
│   ├── trust/                     # trust state, SPRT scorer, LLR channels
│   ├── store/                     # TrustLog, D-index, SRI, schema advisor
│   ├── tee/                       # SGX attestation, DCAP quote handling
│   ├── fl/                        # FL client, trust-aware strategy, CFR
│   ├── attacks/                   # trim, min-max, Krum, backdoor, label flip
│   ├── ml/                        # model architectures and data loaders
│   ├── api/                       # REST server and client
│   └── federation/                # coordinator, silo sealing, witnesses
└── experiments/                   # experiment scripts
    ├── trust_trajectory/          # trust-state dynamics and sealed events
    ├── policy_replay/             # policy revision from the sealed view
    ├── sgx_validation/            # sealed-store and extraction checks
    ├── durability/                # WAL and crash-recovery sweeps
    ├── dri_bench/                 # D-index and SRI microbenchmarks
    ├── advisor/                   # schema-advisor validation
    ├── real_datasets/             # non-FL workloads (EPA, FAERS, AML)
    └── scaling/                   # scaling and serving benchmarks
```

The core library (`dtrust/`) is the code that runs in the coordinator. Each
subdirectory of `experiments/` holds the scripts for one class of
measurements.

---

## 2. Installation

Requires Python 3.10 or later; tested with Python 3.12 on Ubuntu 24.04.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .                  # core library
pip install -e ".[experiments]"   # plus the packages used by experiments/
```

This installs the `dtrust` command-line interface (`dtrust --help`).

---

## 3. Components

- **Trust recurrence.** `dtrust/trust/sprt_scorer.py` implements the
  log-odds trust recurrence (paper Section IV-A). Five channels
  (attestation residual, gradient cosine, per-round success, behavioral
  events, sealed-state continuity) contribute per-channel LLRs; the
  sigmoid maps the accumulator into a `[0,1]` trust score.
- **Sealed store.** `dtrust/store/store.py` implements the TrustLog: an
  append-only, schema-typed relation backed by SQLite in WAL mode, with
  4 KiB chunk-hash bookkeeping and a Merkle root anchored every M rounds.
  The record layout is defined in `dtrust/store/schema.py`.
- **D-index.** `dtrust/store/dri_dyadic.py` is the dyadic Lindley-Spitzer
  summary that answers path-dependent CUSUM earliest-crossing queries in
  `O(log R)`, used for drift-threshold revisions.
- **SRI.** `dtrust/store/sri_authenticated.py` is a Merkle persistent
  segment tree over the quantized trust score, versioned by round. It
  answers `count_below` over a round range, with weight sums, and returns
  an `O(log U)` proof verified against the sealed root.
- **Schema advisor.** `dtrust/store/schema_advisor.py` computes the
  derivation closure of a sealed schema, reports which policy families are
  replayable from it, and computes the minimum-cost augmentation (a
  weighted set-cover instance) for the rest.
- **CFR.** `dtrust/fl/cfr.py` implements certified federated repair: it
  re-aggregates committed updates under a revised weight vector after
  verifying re-supplied updates against their sealed commitments.

---

## 4. License

Apache License 2.0; see `LICENSE`.
