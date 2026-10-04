"""Saved probabilities/checkpoints only: separate fixed-prior burden from ties."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_review import ROOT, sha, check_bindings

OUT = ROOT/'artifacts/v168_root_saved_prior_burden_review_20261002'
PRIOR = ROOT/'artifacts/v164_short_supervised_trajectory_20261002'


def weighted_quantiles(v):
    return {str(p):float(np.quantile(v,p)) for p in [0,.25,.5,.75,1]} if len(v) else None


def main():
    assert not OUT.exists()
    gold_path = ROOT/'data/official/train.parquet'
    gold = pd.read_parquet(gold_path,columns=['label_binary']).label_binary.map(
        {'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    paths = {Path(__file__).resolve(),gold_path,
             ROOT/'training/v159_current_input_boundary_v3.py'}
    for role in range(3):
        paths.update([PRIOR/f'role{role}/endpoint.pt',
                      PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet',
                      ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy'])
    bindings = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    result = []
    for role in range(3):
        rows = pd.read_parquet(PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet')
        assert rows.row_position.is_unique and np.array_equal(rows.truth,gold[rows.row_position])
        bank = np.asarray(np.load(ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy',mmap_mode='r')[:,:16],np.float64)
        means = np.mean(bank,axis=1)
        state = torch.load(PRIOR/f'role{role}/endpoint.pt',weights_only=True,map_location='cpu')['state']
        out = state['output_weight'].numpy()
        assert out.shape == (16,3)
        lp = rows[['logp0','logp1','logp2']].to_numpy()
        local,truth,rival = rows.local.to_numpy(),rows.truth.to_numpy(),rows.pred.to_numpy()
        j = np.arange(len(rows))
        prior_t = np.maximum(means[local,truth],1e-12)
        prior_r = np.maximum(means[local,rival],1e-12)
        prior_margin = np.log(prior_t)-np.log(prior_r)
        actual_margin = lp[j,truth]-lp[j,rival]
        residual_margin = actual_margin-prior_margin
        # Every mean(tanh(.)) coordinate is in [-1,1]. This bounds only this
        # saved output matrix, not a model whose output matrix is trainable.
        bounds = np.array([math.fsum(abs(float(x)) for x in out[:,t]-out[:,r])
                           for t,r in zip(truth,rival)])
        assert np.all(residual_margin <= bounds+1e-10)
        classes = {}
        for cls,name in [(1,'M'),(2,'S')]:
            error = (truth==cls)&(rival!=cls)
            pure = error&rows.pure_current_input.to_numpy()
            classes[name] = dict(original_class_mass=int((truth==cls).sum()),
                errors=int(error.sum()),pure_errors=int(pure.sum()),
                truth_prior_at_or_below_fixed_floor=int((error&(means[local,truth]<=1e-12)).sum()),
                exact_actual_ties=int((error&(actual_margin==0)).sum()),
                actual_error_margin_quantiles=weighted_quantiles(actual_margin[error]),
                frozen_prior_error_margin_quantiles=weighted_quantiles(prior_margin[error]),
                learned_residual_error_margin_quantiles=weighted_quantiles(residual_margin[error]),
                current_readout_maximum_margin_change_quantiles=weighted_quantiles(bounds[error]),
                errors_impossible_if_only_hidden_changes_and_current_readout_frozen=int((error&(prior_margin+bounds<0)).sum()),
                pure_errors_impossible_under_same_frozen_readout=int((pure&(prior_margin+bounds<0)).sum()),
                all_parameters_remain_trainable=True,
                frozen_readout_bound_not_global_model_infeasibility=True)
        result.append(dict(role=role,classes=classes))
    check_bindings(bindings)
    OUT.mkdir()
    report = dict(status='V164_saved_original_row_fixed_prior_and_current_readout_burden_independently_reviewed',
        roles=result,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,
        numpy_saved_bank_mean_is_diagnostic_not_new_model_forward=True,
        interpretation='Many remaining wrong rows have a deeply wrong fixed prior. This is a state-dependent correction burden, not proof of global model incapacity, missing information or unavoidable error.',
        next_use='Track deep margins, learned residual scale, real class repairs and accepted trajectory length separately from the two losing-tie regressions.',
        training_authority=False,source_sha256=bindings)
    (OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
