"""How much does memory mode need?  Shrinks the stored-residue bank and re-measures.
Run after gates.py (uses results/cache).  python bank_size.py"""
import sys, pickle, numpy as np
import os; os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, '.')
from gates import nn_dist, psnr
from sihti_model.data import load
for ds in ['mnist', 'fashion']:
    m = pickle.load(open(f'results/cache/model_{ds}.pkl', 'rb'))
    judge = pickle.load(open(f'results/cache/judge_{ds}.pkl', 'rb'))
    Xtr, ytr = load(ds, 'train'); Xte, yte = load(ds, 'test')
    P = Xtr.reshape(len(Xtr), -1)
    te_med = np.median(np.concatenate([nn_dist(Xte[yte == c][:300].reshape(300, -1), P[ytr == c]) for c in range(10)]))
    idx = np.random.default_rng(4).choice(len(Xte), 1000, replace=False)
    Zc = m.encode(Xte[idx])
    full = m.bank
    gen_params = sum(u.size for u in m.U) + sum(x.size for x in m.mean) + sum(
        sum(np.size(v) for v in (g[b] if b != 'core' else g['core'])) for g in m.gen for b in g)
    print(f'{ds}: linear generator total numbers {gen_params/1e6:.2f}M (fixed, no images stored)')
    for k in [2, 5, 20, 100, 500, 2000]:
        m.bank = [[z[:k] for z in full[c]] for c in range(10)]
        r = np.random.default_rng(7)
        S = np.clip(np.concatenate([m.decode(m.generate(c, 100, mode='memory', rng=r)) for c in range(10)]), 0, 1)
        L = np.repeat(np.arange(10), 100)
        rec = (judge.predict(S.reshape(len(S), -1)) == L).mean()
        d = np.concatenate([nn_dist(S[L == c].reshape(100, -1), P[ytr == c]) for c in range(10)])
        fill = np.clip(m.decode(m.complete(Zc, 1, 'memory')), 0, 1)
        stored = 10 * k * sum(m.dims)
        print(f'  bank {k:5d}/class  stored numbers {stored/1e6:6.3f}M  recognised {rec:.3f}  novelty {np.median(d)/te_med:.2f}  fill {psnr(fill, Xte[idx]):.2f} dB')
