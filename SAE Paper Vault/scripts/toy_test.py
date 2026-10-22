import numpy as np
def run(seed=0, n_feat=12, d=6, D=32, k_top=2, steps=6000, p=0.08):
    rng = np.random.default_rng(seed)
    T = rng.normal(size=(n_feat, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
    def sample(n):
        a = rng.random((n, n_feat)) < p
        return (rng.uniform(0.5, 1.5, (n, n_feat)) * a) @ T
    X = sample(50000)
    Wd = rng.normal(size=(D, d)); Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
    P = {"We": Wd.T.copy(), "be": np.zeros(D), "Wd": Wd, "bd": np.zeros(d)}
    m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}
    last_fire = np.zeros(D); losses = []
    for t in range(1, steps + 1):
        xb = X[rng.integers(0, len(X), 512)]
        pre = (xb - P["bd"]) @ P["We"] + P["be"]
        thr = np.sort(pre, 1)[:, -k_top][:, None]
        mask = (pre >= thr) & (pre > 0); z = pre * mask
        err = z @ P["Wd"] + P["bd"] - xb
        losses.append((err ** 2).sum(1).mean())
        g = 2 * err / len(xb); gz = g @ P["Wd"].T * mask
        G = {"Wd": z.T @ g, "bd": g.sum(0) - gz.sum(0) @ P["We"].T, "We": (xb - P["bd"]).T @ gz, "be": gz.sum(0)}
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * G[k]; v2[k] = 0.999 * v2[k] + 0.001 * G[k] ** 2
            P[k] -= 3e-3 * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
        P["Wd"] /= np.linalg.norm(P["Wd"], axis=1, keepdims=True) + 1e-8
        fired = mask.any(0); last_fire[fired] = 0; last_fire[~fired] += 1
        if t % 500 == 0 and t < steps * 0.7:      # resample dead latents onto badly-reconstructed inputs
            dead = np.where(last_fire > 200)[0]
            if len(dead):
                worst = xb[np.argsort(-(err ** 2).sum(1))[:len(dead)]] - P["bd"]
                worst /= np.linalg.norm(worst, axis=1, keepdims=True) + 1e-8
                P["Wd"][dead] = worst; P["We"][:, dead] = worst.T * 0.5; P["be"][dead] = 0
                for k in m: m[k][...] = 0; v2[k][...] = 0
    cos = np.abs(T @ P["Wd"].T)
    return T, P, cos, losses
if __name__ == "__main__":
    for s in range(4):
        T, P, cos, L = run(s)
        best = cos.max(1); match = cos.argmax(1)
        print(s, best.round(2), "uniq", len(set(match)), "loss", round(np.mean(L[-200:]), 4))
