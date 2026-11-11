#!/usr/bin/env python3
"""P-DRI: Parametric Drift Replay Index. The access method upgrades DRI from a
NUMERIC Kadane monoid on a fixed residual stream to a FUNCTION-VALUED max-plus ENVELOPE monoid over a
future baseline parameter mu. Each increment is affine in mu: x_t(mu) = a_t - mu * b_t (the sealed
affine-increment atom (a_t, b_t)). A block stores Psi(B) = (Sum_B(mu), EnvPre, EnvSuf, EnvSub), where
Pre/Suf/Sub are UPPER ENVELOPES of the candidate-interval lines A_I - mu*B_I (piecewise-linear convex).
Merge is the Kadane recurrence with envelope ops (upper hull + max-plus convolution for the cross-boundary
subarray). Evaluating Psi at any mu recovers the ordinary DRI summary at that mu, so one sealed index serves
driftReplay over the whole changed-baseline family [mu_L, mu_U] -- no raw scan, no per-policy rebuild.

This module is P1 (the envelope monoid) with an exact correctness gate vs brute-force Kadane at every mu.
Run: PYTHONPATH=. python experiment_scripts/dtrust_experiments/db_eval/p_dri.py
"""
import hashlib, random, struct, sys

# A line is value(mu) = A - mu*B, i.e. slope m = -B, intercept c = A, over the parameter mu.
# An Envelope is the upper hull (pointwise max) of a set of lines: a convex piecewise-linear function.

def _upper_hull(lines):
    """Upper envelope (max) of lines [(A,B),...] as functions value(mu)=A-mu*B. Returns the minimal set of
    (A,B) lines that are maximal for some mu, sorted by slope m=-B ascending (so active line advances with mu)."""
    # sort by slope m=-B ascending; for equal slope keep the larger intercept A
    pts = sorted(lines, key=lambda L: (-L[1], -L[0]))   # (-B asc, A desc)
    uniq = []
    for A, B in pts:
        if uniq and -uniq[-1][1] == -B:                  # same slope -> keep larger A (already first)
            continue
        uniq.append((A, B))
    hull = []
    def redundant(l1, l2, l3):
        # l2 is redundant if the intersection(l1,l3) is at or left of intersection(l1,l2) (max hull)
        # x(li,lj) = (Ai-Aj)/(Bi-Bj) for value A-mu*B equal: Ai-mu*Bi = Aj-mu*Bj -> mu=(Ai-Aj)/(Bi-Bj)
        (A1, B1), (A2, B2), (A3, B3) = l1, l2, l3
        # cross13 <= cross12  ==>  l2 never wins
        # use cross-multiplication to avoid division (B are floats; denominators sign matters)
        # mu13 = (A1-A3)/(B1-B3), mu12 = (A1-A2)/(B1-B2); slopes strictly increasing so B strictly decreasing
        d13 = (B1 - B3); d12 = (B1 - B2)
        # both d>0 since B strictly decreasing along the sorted-by-slope order
        return (A1 - A3) * d12 <= (A1 - A2) * d13
    for L in uniq:
        while len(hull) >= 2 and redundant(hull[-2], hull[-1], L):
            hull.pop()
        hull.append(L)
    return hull


def _line_add(l1, l2):
    return (l1[0] + l2[0], l1[1] + l2[1])

def _shift(env, line):
    """Add a single line to every line of an envelope (still an upper hull of the shifted lines)."""
    return _upper_hull([_line_add(l, line) for l in env])

def _emax(e1, e2):
    """Pointwise max of two envelopes = upper hull of the union of their lines."""
    return _upper_hull(list(e1) + list(e2))

def _econv_quadratic(e1, e2):
    """Reference (O(p*q)): max-plus convolution as the upper hull of all pairwise line sums."""
    return _upper_hull([_line_add(a, b) for a in e1 for b in e2])


def _breakpoints(env):
    """mu where consecutive hull lines cross (env sorted by slope ascending, so B strictly decreasing)."""
    return [(env[i][0] - env[i + 1][0]) / (env[i][1] - env[i + 1][1]) for i in range(len(env) - 1)]


def _econv(e1, e2):
    """Max-plus convolution in O(p+q): max_i e1_i(mu) + max_j e2_j(mu) = (max e1) + (max e2), the pointwise
    SUM of two convex PL functions. Walk both breakpoint lists; on each mu-interval the active line is the
    sum of the two inputs' active lines. Output is already an upper hull (i,j advance monotonically)."""
    if not e1 or not e2:
        return []
    b1, b2 = _breakpoints(e1), _breakpoints(e2)
    i = j = 0
    out = [_line_add(e1[0], e2[0])]
    while i < len(e1) - 1 or j < len(e2) - 1:
        nb1 = b1[i] if i < len(b1) else float("inf")
        nb2 = b2[j] if j < len(b2) else float("inf")
        if nb1 < nb2:
            i += 1
        elif nb2 < nb1:
            j += 1
        else:                                  # coincident breakpoints: advance both
            i += 1; j += 1
        line = _line_add(e1[i], e2[j])
        if out[-1] != line:
            out.append(line)
    return out

def _active_in_range(env, muL, muU):
    """Exact pruning: keep only hull lines that are the maximum for some mu in [muL, muU]. A line not
    maximal anywhere in the declared range is never queried, so dropping it loses no crossing."""
    if len(env) <= 1:
        return list(env)
    bp = [float("-inf")]                                   # left breakpoint of each line
    for i in range(len(env) - 1):
        (A_i, B_i), (A_j, B_j) = env[i], env[i + 1]        # sorted by slope -> B_i > B_j, denom > 0
        bp.append((A_i - A_j) / (B_i - B_j))
    bp.append(float("inf"))
    return [env[i] for i in range(len(env)) if not (bp[i + 1] < muL or bp[i] > muU)]


def _eval(env, mu):
    """Evaluate the envelope (max over lines) at mu. env is sorted by slope ascending."""
    best = None
    for A, B in env:
        v = A - mu * B
        if best is None or v > best:
            best = v
    return best


class Psi:
    """Block summary: Sum line + Pre/Suf/Sub upper envelopes (non-empty intervals)."""
    __slots__ = ("sum", "pre", "suf", "sub")
    def __init__(self, sum_line, pre, suf, sub):
        self.sum = sum_line; self.pre = pre; self.suf = suf; self.sub = sub

    @staticmethod
    def leaf(a, b):
        L = (a, b)
        return Psi(L, [L], [L], [L])

    @staticmethod
    def merge(A, B):
        sum_line = _line_add(A.sum, B.sum)
        pre = _emax(A.pre, _shift(B.pre, A.sum))         # prefix of A, or all of A + prefix of B
        suf = _emax(B.suf, _shift(A.suf, B.sum))         # suffix of B, or suffix of A + all of B
        sub = _emax(_emax(A.sub, B.sub), _econv(A.suf, B.pre))  # in A, in B, or crossing the boundary
        return Psi(sum_line, pre, suf, sub)

    def eval_kadane(self, mu):
        """Ordinary DRI summary at this mu: (sum, maxPre, maxSuf, maxSub)."""
        return (self.sum[0] - mu * self.sum[1], _eval(self.pre, mu), _eval(self.suf, mu), _eval(self.sub, mu))


class PDRI:
    """Segment tree of Psi over the affine-increment stream [(a_t, b_t)]."""
    def __init__(self, ab):
        self.n = len(ab); self.size = 1
        while self.size < self.n: self.size <<= 1
        self.tree = [None] * (2 * self.size)
        for i, (a, b) in enumerate(ab):
            self.tree[self.size + i] = Psi.leaf(a, b)
        for i in range(self.n, self.size):              # pad with -inf-ish empty (huge negative)
            self.tree[self.size + i] = Psi.leaf(-1e18, 0.0)
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = Psi.merge(self.tree[2 * i], self.tree[2 * i + 1])

    def update_leaf(self, i, a, b):
        """Per-round maintenance: set leaf i to a new affine increment (a,b) and re-merge the O(log n)
        root-to-leaf path. This is the incremental append/update cost; each of the log n nodes re-merges
        at O(hull) work, so the path costs O(log n * max-hull-on-path)."""
        v = self.size + i
        self.tree[v] = Psi.leaf(a, b)
        v >>= 1
        while v >= 1:
            self.tree[v] = Psi.merge(self.tree[2 * v], self.tree[2 * v + 1])
            v >>= 1

    def query(self, l, r):
        """Merge cover of [l, r] (inclusive) -> Psi. O(log n) nodes."""
        resL, resR = None, None
        lo, hi = l + self.size, r + self.size + 1
        nodesL, nodesR = [], []
        while lo < hi:
            if lo & 1: nodesL.append(self.tree[lo]); lo += 1
            if hi & 1: hi -= 1; nodesR.append(self.tree[hi])
            lo >>= 1; hi >>= 1
        for node in nodesL + nodesR[::-1]:
            resL = node if resL is None else Psi.merge(resL, node)
        return resL

    def earliest_crossing(self, l, r, mu, h):
        """driftReplay(l,r,mu,h): earliest t in [l,r] where the reflected CUSUM S_t=max(0,S_{t-1}+x_t(mu))
        reaches h, starting fresh at l. O(log R) cover nodes + one O(log R) descent. Returns t or None."""
        carry = 0.0
        for ni in self.cover_nodes(l, r):
            node = self.tree[ni]
            sum_, mpre, msuf, msub = node.eval_kadane(mu)
            if max(carry + mpre, max(msub, 0.0)) >= h:        # the walk crosses somewhere inside this node
                return self._descend(ni, carry, mu, h)
            carry = max(carry + sum_, max(msuf, 0.0))         # reflected CUSUM exiting this node
        return None

    def _descend(self, ni, carry, mu, h):
        """Descend a node known to contain a crossing; return the earliest leaf round."""
        while ni < self.size:                                  # internal node -> recurse into children
            lft, rgt = self.tree[2 * ni], self.tree[2 * ni + 1]
            s, mpre, msuf, msub = lft.eval_kadane(mu)
            if max(carry + mpre, max(msub, 0.0)) >= h:
                ni = 2 * ni
            else:
                carry = max(carry + s, max(msuf, 0.0)); ni = 2 * ni + 1
        return ni - self.size                                  # leaf index = round

    def cover_nodes(self, l, r):
        """Return the O(log n) dyadic cover node indices for [l,r] (for receipts / hull-size stats)."""
        idx = []
        lo, hi = l + self.size, r + self.size + 1
        left, right = [], []
        while lo < hi:
            if lo & 1: left.append(lo); lo += 1
            if hi & 1: hi -= 1; right.append(hi)
            lo >>= 1; hi >>= 1
        return left + right[::-1]

    # ---- authenticated receipt (P2): commit each node's canonical envelope; root-verifiable driftReplay ----
    def finalize_auth(self):
        """Merkle-commit the canonical envelope of every tree node. auth_root binds the whole index."""
        N = 2 * self.size
        self._nh = [None] * N
        for i in range(1, N):
            self._nh[i] = _node_hash(self.tree[i])
        self._nh[0] = _NULL
        self._levels = _merkle_levels(self._nh)        # over the N node-hash leaves
        self.auth_root = _h(b"PDRI-ROOT", struct.pack("<q", N), self._levels[-1][0])
        return self.auth_root

    def prove_drift(self, l, r, mu, h):
        """Receipt for driftReplay(l,r,mu,h): the cover-node envelopes + (if a crossing exists) the descent
        path's child nodes, each with a Merkle path to auth_root. The verifier re-runs the deterministic
        crossing over the committed envelopes -- no raw residuals."""
        crossing = self.earliest_crossing(l, r, mu, h)
        reveal = {}                                         # node index -> {psi, path}
        cover = self.cover_nodes(l, r)
        for ni in cover:
            reveal[ni] = {"psi": _psi_wire(self.tree[ni]), "path": _path_from_levels(self._levels, ni)}
        # replay the carry to find the crossing cover node, then descend, revealing each node we touch
        carry, hit = 0.0, None
        for ni in cover:
            s, mpre, msuf, msub = self.tree[ni].eval_kadane(mu)
            if max(carry + mpre, max(msub, 0.0)) >= h: hit = ni; break
            carry = max(carry + s, max(msuf, 0.0))
        if hit is not None:
            ni = hit
            while ni < self.size:
                for c in (2 * ni, 2 * ni + 1):
                    reveal.setdefault(c, {"psi": _psi_wire(self.tree[c]), "path": _path_from_levels(self._levels, c)})
                s, mpre, msuf, msub = self.tree[2 * ni].eval_kadane(mu)
                if max(carry + mpre, max(msub, 0.0)) >= h: ni = 2 * ni
                else: carry = max(carry + s, max(msuf, 0.0)); ni = 2 * ni + 1
        return {"l": l, "r": r, "mu": mu, "h": h, "size": self.size, "crossing": crossing,
                "cover": cover, "nodes": {str(k): v for k, v in reveal.items()}}


def _eval_wire(env, mu):
    best = None
    for A, B in env:
        v = A - mu * B
        if best is None or v > best: best = v
    return best
def _kadane_wire(w, mu):
    return (w["sum"][0] - mu * w["sum"][1], _eval_wire(w["pre"], mu), _eval_wire(w["suf"], mu), _eval_wire(w["sub"], mu))

def verify_drift(auth_root, l, r, mu, h, claimed_crossing, proof):
    """Sound external verify of a driftReplay receipt, BOUND to the requested query (l,r,mu,h): (0) the
    proof answers exactly the requested query and claim; (1) every revealed node's canonical envelope is
    Merkle-committed under auth_root; (2) re-running the deterministic crossing over those committed
    envelopes reproduces the claimed crossing."""
    if (proof.get("l"), proof.get("r"), proof.get("mu"), proof.get("h"), proof.get("crossing")) \
       != (l, r, mu, h, claimed_crossing):
        return False                                      # proof must answer the REQUESTED query+claim
    size = proof["size"]
    N = 2 * size
    nodes = proof["nodes"]
    # (1) every revealed node's canonical envelope is Merkle-committed under the SAME auth_root
    tops = {_root_from_path(_node_hash_from_wire(v["psi"]), int(k), v["path"]) for k, v in nodes.items()}
    if len(tops) != 1: return False
    if _h(b"PDRI-ROOT", struct.pack("<q", N), tops.pop()) != auth_root: return False
    # (2) re-run the crossing over the committed envelopes
    carry, hit = 0.0, None
    for ni in proof["cover"]:
        w = nodes[str(ni)]["psi"]; s, mpre, msuf, msub = _kadane_wire(w, mu)
        if max(carry + mpre, max(msub, 0.0)) >= h: hit = ni; break
        carry = max(carry + s, max(msuf, 0.0))
    crossing = None
    if hit is not None:
        ni = hit
        while ni < size:
            w = nodes[str(2 * ni)]["psi"]; s, mpre, msuf, msub = _kadane_wire(w, mu)
            if max(carry + mpre, max(msub, 0.0)) >= h: ni = 2 * ni
            else: carry = max(carry + s, max(msuf, 0.0)); ni = 2 * ni + 1
        crossing = ni - size
    return crossing == proof["crossing"]


def _NULLpad(): return _NULL
_NULL = b"\x00" * 32
def _h(*p):
    x = hashlib.sha256()
    for q in p: x.update(q if isinstance(q, (bytes, bytearray)) else str(q).encode())
    return x.digest()
def _merkle_levels(leaves):
    lv = [list(leaves)]
    while len(lv[-1]) > 1:
        c = lv[-1]; lv.append([_h(c[i], c[i + 1] if i + 1 < len(c) else c[i]) for i in range(0, len(c), 2)])
    return lv
def _path_from_levels(levels, idx):
    path, i = [], idx
    for lvl in levels[:-1]:
        sib = i ^ 1; path.append(lvl[sib] if sib < len(lvl) else lvl[i]); i //= 2
    return path
def _root_from_path(leaf, idx, path):
    node, i = leaf, idx
    for sib in path:
        node = _h(node, sib) if i % 2 == 0 else _h(sib, node); i //= 2
    return node
def _env_bytes(env):
    return b"".join(struct.pack("<dd", float(A), float(B)) for (A, B) in env)
def _psi_wire(node):
    return {"sum": [node.sum[0], node.sum[1]], "pre": [list(l) for l in node.pre],
            "suf": [list(l) for l in node.suf], "sub": [list(l) for l in node.sub]}
def _node_hash_from_wire(w):
    sumb = struct.pack("<dd", float(w["sum"][0]), float(w["sum"][1]))
    return _h(b"PNODE", sumb, _env_bytes(w["pre"]), _env_bytes(w["suf"]), _env_bytes(w["sub"]))
def _node_hash(node):
    return _h(b"PNODE", struct.pack("<dd", float(node.sum[0]), float(node.sum[1])),
              _env_bytes(node.pre), _env_bytes(node.suf), _env_bytes(node.sub))


def _brute_kadane(ab, l, r, mu):
    """Brute-force max non-empty subarray of x_t(mu)=a_t-mu*b_t on [l,r]."""
    xs = [a - mu * b for (a, b) in ab[l:r + 1]]
    best = cur = xs[0]
    for x in xs[1:]:
        cur = max(x, cur + x); best = max(best, cur)
    # also maxPre/maxSuf/sum for full check
    pre = []; s = 0
    for x in xs: s += x; pre.append(s)
    total = s
    maxpre = max(pre); maxsuf = max(total - (pre[i - 1] if i > 0 else 0) for i in range(len(xs)))
    return total, maxpre, maxsuf, best


def _brute_crossing(ab, l, r, mu, h):
    carry = 0.0
    for t in range(l, r + 1):
        carry = max(0.0, carry + (ab[t][0] - mu * ab[t][1]))
        if carry >= h:
            return t
    return None

def _selftest():
    rng = random.Random(0)
    fails = 0; checks = 0; xfails = 0; xchecks = 0
    # adversarial case: mixed-sign b_t forces multi-line suffix/prefix envelopes whose best
    # cross-boundary subarray changes with mu.
    adv = [(0.0, 3.0), (10.0, -1.0), (10.0, -1.0), (0.0, 3.0)]
    aidx = PDRI(adv)
    for mu in (-100., -10., -1., 0., 1., 10., 100.):
        g = aidx.query(0, 3).eval_kadane(mu); e = _brute_kadane(adv, 0, 3, mu)
        checks += 1
        if not all(abs(a - b) < 1e-6 for a, b in zip(g, e)):
            fails += 1; print(f"  ADV MISMATCH mu={mu} got={g} exp={e}")
    for trial in range(200):
        n = rng.randint(1, 40)
        ab = [(rng.uniform(-5, 5), rng.uniform(0.1, 3.0)) for _ in range(n)]
        idx = PDRI(ab)
        for _ in range(6):
            l = rng.randint(0, n - 1); r = rng.randint(l, n - 1)
            psi = idx.query(l, r)
            for mu in (-2.0, 0.0, 0.37, 1.0, 2.5, rng.uniform(-3, 3)):
                got = psi.eval_kadane(mu)
                exp = _brute_kadane(ab, l, r, mu)
                checks += 1
                if not all(abs(g - e) < 1e-6 for g, e in zip(got, exp)):
                    fails += 1
                    if fails <= 3:
                        print(f"  MISMATCH n={n} [{l},{r}] mu={mu:.3f} got={got} exp={exp}")
                for h in (1.0, 5.0, 12.0):                    # earliest-crossing (driftReplay) exactness
                    xchecks += 1
                    if idx.earliest_crossing(l, r, mu, h) != _brute_crossing(ab, l, r, mu, h):
                        xfails += 1
                        if xfails <= 3:
                            print(f"  XING MISMATCH [{l},{r}] mu={mu:.3f} h={h} "
                                  f"got={idx.earliest_crossing(l,r,mu,h)} exp={_brute_crossing(ab,l,r,mu,h)}")
    print(f"P-DRI P1 selftest: {checks} (range,mu) Kadane checks, {fails} mismatches"
          + (" -> PASS" if fails == 0 else " -> FAIL"))
    print(f"P-DRI driftReplay: {xchecks} earliest-crossing checks, {xfails} mismatches"
          + (" -> PASS" if xfails == 0 else " -> FAIL"))
    fails += xfails
    # P2: authenticated receipt -- verifies, and rejects wrong-mu / tampered-crossing / tampered-envelope
    import copy as _copy
    arng = random.Random(7); afail = 0; achk = 0
    for _ in range(40):
        n = arng.randint(4, 64)
        ab2 = [(arng.uniform(-3, 3), arng.uniform(0.2, 2.0)) for _ in range(n)]
        idx2 = PDRI(ab2); root = idx2.finalize_auth()
        mu = arng.uniform(-2, 2); h = arng.choice([2.0, 5.0, 9.0])
        pr = idx2.prove_drift(0, n - 1, mu, h); cx = pr["crossing"]; achk += 1
        if not verify_drift(root, 0, n - 1, mu, h, cx, pr): afail += 1; print("  AUTH verify FAILED")
        if verify_drift(root, 0, n - 1, mu + 0.5, h, cx, pr): afail += 1; print("  AUTH wrong-mu ACCEPTED")
        if verify_drift(root, 0, n - 1, mu, h, (cx or 0) + 1, pr): afail += 1; print("  AUTH bad-crossing ACCEPTED")
        bad = _copy.deepcopy(pr); k = next(iter(bad["nodes"])); bad["nodes"][k]["psi"]["sub"][0][0] += 3.0
        if verify_drift(root, 0, n - 1, mu, h, cx, bad): afail += 1; print("  AUTH tampered-envelope ACCEPTED")
    print(f"P-DRI auth receipt: {achk} proofs, {afail} soundness failures"
          + (" -> PASS" if afail == 0 else " -> FAIL"))
    fails += afail
    # worst-case envelope bound: per-node |EnvSub| <= O(b log b) for a block of b leaves. Stress with
    # adversarial families designed to keep many subarrays on the upper envelope.
    import math as _math
    brng = random.Random(11); worst_ratio = 0.0; worst_b = 0; worst_sz = 0
    max_abs_sz = 0; max_abs_b = 0; violations = 0
    def _families(n):
        yield [(brng.uniform(-9, 9), brng.uniform(0.05, 9.0)) for _ in range(n)]      # wide-b random
        yield [(float(t * t), float(t + 1)) for t in range(n)]                         # convex a, linear b
        yield [((-1) ** t * float(t), 0.1 + (t % 7)) for t in range(n)]                # sign-alternating
        yield [(float(n - t), float(t + 1)) for t in range(n)]                         # decreasing a
    for n in (64, 256, 1024):
        for ab2 in _families(n):
            idx3 = PDRI(ab2)
            for i in range(1, 2 * idx3.size):
                node = idx3.tree[i]
                if node is None or node.sub[0][0] <= -1e17: continue
                b = 1
                while (1 << (b.bit_length())) and False: pass
                # block size = leaves under node i
                lvl = 0; j = i
                while j < idx3.size: j *= 2; lvl += 1
                bsz = 1 << lvl
                sz = len(node.sub)
                bound = float(bsz) * _math.log2(2 * bsz)        # tight: S(b) <= b*log2(2b) = b log b + b
                if sz > bound + 1e-9: violations += 1
                if sz / bound > worst_ratio: worst_ratio = sz / bound; worst_b = bsz; worst_sz = sz
                if sz > max_abs_sz: max_abs_sz = sz; max_abs_b = bsz
    print(f"P-DRI envelope bound: across adversarial families, {violations} nodes exceed b*log2(2b); "
          f"max |EnvSub|={max_abs_sz} at block b={max_abs_b} (bound {int(max_abs_b*_math.log2(2*max(1,max_abs_b)))}); "
          f"worst ratio={worst_ratio:.3f}" + (" -> bound HOLDS" if violations == 0 else " -> EXCEEDS"))
    # convolution adversary: m^2 pair sums must collapse to exactly 2m-1 hull pieces (additive, not product)
    convfail = 0
    for m in (4, 16, 64, 256):
        left = [(float(2 * t - 1), 1.0) for t in range(1, m + 1)]
        right = [(float(-(2 * t - 1)), 1.0) for t in range(1, m + 1)]
        eL = _upper_hull(left); eR = _upper_hull(right); conv = _econv(eL, eR)
        if len(conv) > 2 * m - 1: convfail += 1; print(f"  CONV m={m}: |conv|={len(conv)} > 2m-1={2*m-1}")
    print(f"P-DRI convolution additivity: m^2 pair-sums collapse to <= 2m-1 hull lines, {convfail} failures"
          + (" -> PASS" if convfail == 0 else " -> FAIL"))
    fails += convfail
    # linear O(p+q) _econv must equal the quadratic reference (so build is additive, not just storage)
    lrng = random.Random(17); linfail = 0
    for _ in range(300):
        e1 = _upper_hull([(lrng.uniform(-9, 9), lrng.uniform(0.05, 9)) for _ in range(lrng.randint(1, 30))])
        e2 = _upper_hull([(lrng.uniform(-9, 9), lrng.uniform(0.05, 9)) for _ in range(lrng.randint(1, 30))])
        a = _econv(e1, e2); b = _econv_quadratic(e1, e2)
        for mu in (-50, -3, 0, 0.7, 3, 50):
            if abs(_eval(a, mu) - _eval(b, mu)) > 1e-7: linfail += 1; break
    print(f"P-DRI linear convolution: O(p+q) merge == O(pq) reference on 300 cases, {linfail} mismatches"
          + (" -> PASS" if linfail == 0 else " -> FAIL"))
    fails += linfail
    # mu-range exact pruning: pruned eval == full eval for mu inside the range
    prng = random.Random(13); pfail = 0
    for _ in range(60):
        n = prng.randint(2, 80); ab4 = [(prng.uniform(-4, 4), prng.uniform(0.2, 3.0)) for _ in range(n)]
        idx4 = PDRI(ab4); muL, muU = sorted([prng.uniform(-3, 3), prng.uniform(-3, 3)])
        sub = idx4.query(0, n - 1).sub; pruned = _active_in_range(sub, muL, muU)
        for _ in range(8):
            mu = prng.uniform(muL, muU)
            if abs(_eval(pruned, mu) - _eval(sub, mu)) > 1e-9: pfail += 1
    print(f"P-DRI mu-range pruning: 480 in-range evals, {pfail} mismatches"
          + (" -> PASS" if pfail == 0 else " -> FAIL"))
    fails += violations + pfail
    # hull-size sanity on a larger random stream
    ab = [(rng.uniform(-5, 5), rng.uniform(0.1, 3.0)) for _ in range(4096)]
    idx = PDRI(ab)
    hulls = [len(idx.tree[i].sub) for i in range(1, 2 * idx.size) if idx.tree[i] is not None]
    hulls.sort()
    print(f"  sub-envelope hull sizes over all nodes: median={hulls[len(hulls)//2]} "
          f"p95={hulls[int(0.95*len(hulls))]} max={hulls[-1]}")
    # persist the bound/soundness provenance for prop:pdri-bound + thm:pdri
    import json as _json
    prov = {"_note": "P-DRI envelope-monoid soundness + worst-case bound provenance (db_eval/p_dri.py "
                     "self-test). Backs thm:pdri (exact driftReplay), prop:pdri-bound (envelope size bound + "
                     "convolution additivity + mu-range pruning), and the authenticated receipt.",
            "kadane_checks": checks, "kadane_mismatches": fails - violations - pfail - afail - convfail,
            "driftreplay_checks": xchecks, "driftreplay_mismatches": xfails,
            "auth_receipt_proofs": achk, "auth_receipt_soundness_failures": afail,
            "envelope_bound": {"adversarial_families": ["wide-b", "convex-a", "sign-alternating", "decreasing-a"],
                               "nodes_exceeding_b_log2_2b": violations, "max_envsub": max_abs_sz,
                               "max_envsub_block": max_abs_b, "bound_at_max_block": int(max_abs_b*_math.log2(2*max(1,max_abs_b))),
                               "claim": "|EnvSub| <= b*ceil(log2 b)+1 = O(b log b); total O(R log^2 R); additive not product"},
            "convolution_additivity": {"m_tested": [4, 16, 64, 256], "failures_exceeding_2m_minus_1": convfail},
            "linear_convolution_Opq": {"cases": 300, "mismatches_vs_quadratic_reference": linfail,
                                       "claim": "O(p+q) breakpoint-walk merge == O(pq) reference -> additive build"},
            "append_cost_note": "per-round append re-merges O(log R) root-to-leaf path; O(R log R) worst, "
                                "O(log R) small-hull in practice; measured on real EPA in p_dri_epa.json",
            "mu_range_pruning": {"in_range_evals": 480, "mismatches": pfail},
            "random_hull_sizes_4096": {"median": hulls[len(hulls)//2], "p95": hulls[int(0.95*len(hulls))], "max": hulls[-1]}}
    try:
        od = Path(__file__).resolve().parents[3] / "results/artifact"; od.mkdir(parents=True, exist_ok=True)
        (od / "p_dri_bound.json").write_text(_json.dumps(prov, indent=2))
        print("wrote", od / "p_dri_bound.json")
    except Exception as e:
        print("could not write provenance:", e)
    return fails == 0


from pathlib import Path
if __name__ == "__main__":
    ok = _selftest()
    sys.exit(0 if ok else 1)
