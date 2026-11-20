"""NP-beta1 generic worker (Nudge-Polish beta 1) for step-function and point-configuration minimax problems.

Problems (all expressed as MINIMIZE `score`):
  c1a : first autocorrelation, v >= 0 on n cells:   2n max_k (v*v)_k / S^2
  c1b : Erdos minimum overlap, 0 <= h <= 1, mass n/2: (2/n) max_m sum_i h_i (1 - h_{i+m})
  c1c : Barnard-Steinerberger, w >= 0, K cells:       -(K min_{0<=k<=K} A_k / S^2)   (A = autocorrelation)
  mdr : 16 points in R^2:                             (dmax / dmin)^2
  heil: 11 points in the unit equilateral triangle:   -(min triangle area) / (sqrt3/4)
Loop: read shared best; nudge (one random block of cells, or a random subset of points, scaled by sigma);
polish (step: p-norm continuation L-BFGS-B from p0, doubling to pmax within SECS; points: SLSQP epigraph);
keep if the true score drops; atomic shared best under a file lock; every sample saved and logged.
Usage: python3 npg.py <problem> <tag> <shared_dir> <seconds> <seed> --n N [--smin --smax --p0 --secs --bmin --bmax --kmax]
shared_dir must hold best.npy (float vector; points flattened).
"""
import sys, os, time, argparse, fcntl, itertools, numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
ap = argparse.ArgumentParser()
for a_, t_ in [("prob", str), ("tag", str), ("shared", str), ("seconds", float), ("seed", int)]: ap.add_argument(a_, type=t_)
ap.add_argument("--n", type=int, default=15); ap.add_argument("--smin", type=float, default=1e-8); ap.add_argument("--smax", type=float, default=1e-6)
ap.add_argument("--p0", default="4096"); ap.add_argument("--secs", default="25"); ap.add_argument("--pmax", type=float, default=3.4e7)
ap.add_argument("--bmin", type=int, default=16); ap.add_argument("--bmax", type=int, default=4096); ap.add_argument("--kmax", type=int, default=4)
A = ap.parse_args(); rng = np.random.default_rng(A.seed); t0 = time.time()
P0S = [float(t) for t in A.p0.split(",")]; SECS = [float(t) for t in A.secs.split(",")]
SQ3 = np.sqrt(3.0)

# ---------------- problem definitions ----------------
def conv(a, b): return fftconvolve(a, b)
def r_c1a(v):
    S = v.sum(); return 2 * len(v) * conv(v, v) / S ** 2
def r_c1b(h):
    n = len(h); one = np.ones(n); return (2 / n) * (conv(h, one[::-1]) - conv(h, h[::-1]))
def r_c1c(w):
    A_ = conv(w, w[::-1])[len(w) - 1:]                        # A_k, k = 0..len-1
    return KC * A_[:KC + 1] / w.sum() ** 2
PROB = A.prob; KC = None
def score(x):
    if PROB == "c1a": return r_c1a(x).max()
    if PROB == "c1b": return r_c1b(x).max()
    if PROB == "c1c": return -r_c1c(x).min()
    if PROB == "mdr":
        p = x.reshape(-1, 2); d = ((p[:, None] - p[None]) ** 2).sum(-1)[np.triu_indices(len(p), 1)]; return d.max() / d.min()
    if PROB == "heil":
        p = x.reshape(-1, 2)
        if (p[:, 1] < 0).any() or (SQ3 * p[:, 0] + p[:, 1] > SQ3).any() or (p[:, 1] > SQ3 * p[:, 0]).any(): return np.inf
        return -tri_areas(p).min() / (SQ3 / 4)
TRI = None
def tri_areas(p):
    i, j, k = TRI.T
    return np.abs(p[i, 0] * (p[j, 1] - p[k, 1]) + p[j, 0] * (p[k, 1] - p[i, 1]) + p[k, 0] * (p[i, 1] - p[j, 1])) / 2

# ---------------- polishers ----------------
def pnorm_max(r, p):          # smooth max of nonnegative r, with weights dM/dr (zeros contribute nothing)
    m = r.max(); lr = np.log(np.maximum(r, 1e-300) / m); z = np.exp(p * lr)
    Ssum = z.sum(); M = m * Ssum ** (1 / p)
    wts = np.exp((p - 1) * (lr - np.log(M / m))) * (r > 0); return M, wts

def polish_step(x, p0, secs):
    n = len(x); tl = time.time() + secs; best, bs = x.copy(), score(x); p = p0
    if PROB == "c1b":
        lam = 1e4; mu = [0.0]; w_ = 2 / n; one = np.ones(n)
        def proj(a):
            for _ in range(60):
                d = n / 2 - a.sum()
                if abs(d) < 1e-13 * n: break
                free = (a > 0) & (a < 1); a = np.clip(a + d / max(free.sum(), 1) * free, 0, 1)
            return a
    while p <= A.pmax and time.time() < tl:
        if PROB == "c1a":
            def f(v):
                S = v.sum(); c = conv(v, v); r = 2 * n * c / S ** 2; M, wt = pnorm_max(r, p)
                g = (4 * n / S ** 2) * fftconvolve(wt, v[::-1])[n - 1:2 * n - 1] - 2 * (wt @ r) / S
                return M, g
            bnds = [(0, None)] * n
        elif PROB == "c1c":
            L = n
            def f(w):
                S = w.sum(); Ak = conv(w, w[::-1])[L - 1:L - 1 + KC + 1]; r = KC * Ak / S ** 2
                inv = 1 / r; M, wt = pnorm_max(inv, p)
                dr = -wt * inv ** 2                                   # dM/dr_m
                c = dr * KC / S ** 2                                  # dM/dA_m
                H1 = conv(w[::-1], c)[:L][::-1]                       # sum_m c_m w_{j+m}
                H2 = conv(w, c)[:L]                                   # sum_m c_m w_{j-m}
                return M, H1 + H2 - 2 * (dr @ r) / S
            bnds = [(0, None)] * n
        else:  # c1b, log-sum-exp with augmented Lagrangian mass
            beta = p
            def f(a):
                C = w_ * (conv(a, one[::-1]) - conv(a, a[::-1])); m = C.max(); e = np.exp(beta * (C - m)); Z = e.sum()
                val = m + np.log(Z) / beta; q = e / Z
                g = w_ * (conv(q, one)[n - 1:2 * n - 1] - conv(q, a)[n - 1:2 * n - 1] - conv(a, q[::-1])[n - 1:2 * n - 1])
                r_ = w_ * a.sum() - 1; g = g + (mu[0] + 2 * lam * r_) * w_
                return val + mu[0] * r_ + lam * r_ ** 2, g
            bnds = [(0, 1)] * n
        left = tl - time.time()
        if left < 1: break
        r = minimize(f, x, jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": 400, "maxcor": 20, "ftol": 1e-17, "gtol": 1e-15})
        x = r.x
        if PROB == "c1b":
            mu[0] += 2 * lam * (w_ * x.sum() - 1); y = proj(x.copy())
        else: y = x
        s = score(y)
        if s < bs: bs, best = s, y.copy()
        p *= 2 if PROB != "c1b" else 1.6
    return best, bs

def polish_points(x, secs):
    p = x.reshape(-1, 2); npts = len(p)
    if PROB == "mdr":
        iu = np.triu_indices(npts, 1)
        def d2(z):
            q = z[:-1].reshape(-1, 2); return ((q[:, None] - q[None]) ** 2).sum(-1)[iu]
        sc = 1 / np.sqrt(d2(np.append(x, 0)).min()); z0 = np.append(x * sc, d2(np.append(x * sc, 0)).max())
        cons = [{"type": "ineq", "fun": lambda z: z[-1] - d2(z)}, {"type": "ineq", "fun": lambda z: d2(z) - 1.0}]
        r = minimize(lambda z: z[-1], z0, method="SLSQP", constraints=cons, options={"maxiter": 500, "ftol": 1e-16})
        y = r.x[:-1]
    else:
        i, j, k = TRI.T
        sgn = np.sign(p[i, 0] * (p[j, 1] - p[k, 1]) + p[j, 0] * (p[k, 1] - p[i, 1]) + p[k, 0] * (p[i, 1] - p[j, 1])); sgn[sgn == 0] = 1
        def sar(z):
            q = z[:-1].reshape(-1, 2)
            return sgn * (q[i, 0] * (q[j, 1] - q[k, 1]) + q[j, 0] * (q[k, 1] - q[i, 1]) + q[k, 0] * (q[i, 1] - q[j, 1])) / 2
        def inside(z):
            q = z[:-1].reshape(-1, 2); return np.concatenate([q[:, 1], SQ3 - SQ3 * q[:, 0] - q[:, 1], SQ3 * q[:, 0] - q[:, 1]])
        z0 = np.append(x, tri_areas(p).min())
        cons = [{"type": "ineq", "fun": lambda z: sar(z) - z[-1]}, {"type": "ineq", "fun": inside}]
        r = minimize(lambda z: -z[-1], z0, method="SLSQP", constraints=cons, options={"maxiter": 500, "ftol": 1e-16})
        y = r.x[:-1]; q = y.reshape(-1, 2)
        q[:, 1] = np.maximum(q[:, 1], 0)                                   # clamp tiny violations onto the boundary
        for _ in range(3):
            over = SQ3 * q[:, 0] + q[:, 1] - SQ3; m = over > 0; q[m] -= np.outer(over[m] / 4, [SQ3, 1])
            over = q[:, 1] - SQ3 * q[:, 0]; m = over > 0; q[m] += np.outer(over[m] / 4, [SQ3, -1])
        y = q.ravel()
    s = score(y)
    return y, s

# ---------------- shared best, loop ----------------
best_f = os.path.join(A.shared, "best.npy"); lock_f = os.path.join(A.shared, "best.lock")
log = open(os.path.join(A.shared, f"{A.tag}.log"), "a")
def read_best():
    with open(lock_f, "a") as L:
        fcntl.flock(L, fcntl.LOCK_SH); x = np.load(best_f).astype(float); fcntl.flock(L, fcntl.LOCK_UN)
    return x, score(x)
def offer(x, s):
    with open(lock_f, "a") as L:
        fcntl.flock(L, fcntl.LOCK_EX); won = s < score(np.load(best_f).astype(float))
        if won:
            tmp = os.path.join(A.shared, f"tmp_{A.tag}.npy"); np.save(tmp, x); os.replace(tmp, best_f)
        fcntl.flock(L, fcntl.LOCK_UN)
    return won
if PROB == "c1c": KC = int(open(os.path.join(A.shared, "K.txt")).read())
if PROB == "heil": TRI = np.array(list(itertools.combinations(range(11), 3)))
x0, s0 = read_best()
print(f"{A.tag} {PROB} start {s0:.17g} len {len(x0)} args {vars(A)}", file=log, flush=True)
k = wins = 0
while k < A.n:
    left = A.seconds - (time.time() - t0)
    secs = min(float(rng.choice(SECS)), left - 5)
    if secs < 3: break
    k += 1; x, s_in = read_best(); v = x.copy(); sigma = 10 ** rng.uniform(np.log10(A.smin), np.log10(A.smax))
    if PROB in ("mdr", "heil"):
        q = v.reshape(-1, 2); kk = int(rng.integers(1, A.kmax + 1)); idx = rng.choice(len(q), kk, replace=False)
        q[idx] += sigma * rng.standard_normal((kk, 2)); L, b = kk, int(idx[0]); p0 = 0.0
        y, cs = polish_points(q.ravel(), secs)
    else:
        n = len(v); L = int(rng.integers(A.bmin, min(A.bmax, n) + 1)); b = int(rng.integers(0, n - L + 1))
        v[b:b + L] *= 1 + sigma * rng.standard_normal(L); v = np.maximum(v, 0)
        if PROB == "c1b": v = np.clip(v, 0, 1)
        p0 = float(rng.choice(P0S)); y, cs = polish_step(v, p0, secs)
    won = bool(cs < s_in) and offer(y, cs); wins += won
    np.save(os.path.join(A.shared, f"samp_{A.tag}_{k:02d}.npy"), y)
    print(f"{A.tag} k {k} t {time.time()-t0:.0f} sigma {sigma:.2e} block {L}@{b} p0 {p0:.0f} secs {secs:.0f} in {s_in:.17g} out {cs:.17g} d {cs-s_in:+.3e}{' WIN' if won else ''}", file=log, flush=True)
x, s = read_best()
print(f"{A.tag} final samples {k} wins {wins} shared best {s:.17g} start {s0:.17g} gain {s0-s:.3e}", file=log, flush=True)
print(f"{A.tag} final samples {k} wins {wins} shared best {s:.17g} start {s0:.17g} gain {s0-s:.3e}")
