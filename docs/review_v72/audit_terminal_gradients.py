"""Read saved unweighted logistic predictions; zero fits, no weight selection.

Terminal derivative sums are diagnostics, not trajectories, split gains or a
causal estimate of the benefit of reweighting.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    entries, inputs = [], {}
    for fold in range(3):
        for arm in ['Q_old_bins', 'R_observed']:
            path = ROOT / f'artifacts/v71_resolution_20260921/fold{fold}/{arm}/fit.parquet'
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            f = pd.read_parquet(path)
            y = f.label.eq(2).to_numpy(dtype=float)
            p = f.score.to_numpy(dtype=float)
            assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
            pc = np.clip(p, 1e-15, 1-1e-15)
            loss = -(y*np.log(pc)+(1-y)*np.log1p(-pc))
            g, h = p-y, p*(1-p)
            for label in [1, 2]:
                k = f.label.eq(label).to_numpy()
                entries.append(dict(fold=fold, arm=arm, label=label, rows=int(k.sum()),
                    row_fraction=float(k.mean()), terminal_loss_fraction=float(loss[k].sum()/loss.sum()),
                    terminal_absolute_gradient_fraction=float(np.abs(g[k]).sum()/np.abs(g).sum()),
                    terminal_hessian_fraction=float(h[k].sum()/h.sum()),
                    mean_terminal_absolute_gradient=float(np.abs(g[k]).mean()),
                    signed_gradient_sum=float(g[k].sum())))
    result = dict(new_fits=0, scope='Saved fit predictions only; terminal derivatives of unweighted binary logistic loss with respect to margin.',
        limitations=['Not training trajectory or leaf-wise gain.', 'Class gradient totals do not establish how well useful features receive updates.',
                     'Does not identify optimal class weights or prove a reweighting benefit.'],
        entries=entries, inputs_sha256=inputs,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps([r for r in entries if r['label']==2], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
