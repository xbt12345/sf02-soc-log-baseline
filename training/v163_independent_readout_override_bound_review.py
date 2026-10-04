"""Saved arrays and CPU weights: a necessary readout condition, no model call."""
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v160_independent_fixed_diagnostic_review import read, sha

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'artifacts/v163_fixed_endpoint_one_sided_restoration_20261002'
OUT = ROOT / 'artifacts/v163_independent_readout_override_bound_review_20261002'
EPS = float(np.finfo(np.float64).eps)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def review(frame, prior, output):
    # tanh and its member mean are in [-1,1]. With this fixed output matrix,
    # |delta_t-delta_r| <= sum_j |W[j,t]-W[j,r]|, irrespective of the encoder.
    bounds = np.array([[math.fsum(abs(output[:,t]-output[:,r]))
                        for r in range(3)] for t in range(3)])
    local = frame.local.to_numpy(np.int64)
    truth = frame.truth.to_numpy(np.int64)
    gaps = prior[local] - prior[local, truth, None]
    limits = bounds[truth]
    resolution = 64*EPS*np.maximum(1., np.maximum(abs(gaps), abs(limits)))
    blocked = gaps > limits + resolution
    blocked[np.arange(len(frame)), truth] = False
    impossible = blocked.any(1)
    lp = frame[['logp0','logp1','logp2']].to_numpy()
    correction_difference = lp - lp[np.arange(len(frame)),truth,None] - gaps
    assert np.all(abs(correction_difference) <= limits + resolution)
    assert not (impossible & frame.pred.eq(frame.truth).to_numpy()).any()
    wrong = frame.pred.ne(frame.truth).to_numpy()
    pure = frame.pure_current_input.to_numpy()
    result = dict(original_rows=len(frame), output_pair_absolute_bounds=bounds.tolist(), classes={})
    for c, label in [(1,'malicious'),(2,'suspicious')]:
        sel = truth == c
        result['classes'][label] = dict(
            support=int(sel.sum()), current_errors=int((sel&wrong).sum()),
            pure_current_errors=int((sel&wrong&pure).sum()),
            fixed_readout_cannot_override_prior=int((sel&impossible).sum()),
            pure_errors_fixed_readout_cannot_override=int((sel&wrong&pure&impossible).sum()),
            distinct_locals_blocked=int(frame.loc[sel&impossible,'local'].nunique()),
            distinct_roots_blocked=int(frame.loc[sel&impossible,'root'].nunique()))
    return result


def main():
    assert not OUT.exists()
    OUT.mkdir()
    paths = {Path(__file__).resolve(), ROOT/'training/v160_independent_fixed_diagnostic_review.py',
             ROOT/'data/official/train.parquet'}
    chosen = []
    for role in range(3):
        parent = TRIAL/f'role{role}'
        accepted = [p.parent for p in parent.glob('restoration*/finite_probe/probe.json') if read(p)['accepted']]
        assert len(accepted) == 1
        chosen.append(accepted[0])
        paths |= {ROOT/f'artifacts/v159_class_boundary_numeric_trial_20261002/fold{role}_B/endpoint.pt',
                  ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy',
                  parent/'baseline/OOF_original_rows.parquet', accepted[0]/'OOF_original_rows.parquet',
                  accepted[0].parent/'displacement.npy'}
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
    save(OUT/'pre_review_bindings.json', dict(source_sha256=bindings, official_calls=0))
    gold = pd.read_parquet(ROOT/'data/official/train.parquet', columns=['label_binary']).label_binary.map(
        {'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    reports=[]
    for role, proposal in enumerate(chosen):
        state = torch.load(ROOT/f'artifacts/v159_class_boundary_numeric_trial_20261002/fold{role}_B/endpoint.pt',
                           weights_only=True, map_location='cpu')['state']
        assert list(state)[-1] == 'output_weight' and tuple(state['output_weight'].shape) == (16,3)
        output = state['output_weight'].numpy()
        u = np.load(proposal.parent/'displacement.npy')
        assert u.shape == (1060832,)
        updated = output + u[-48:].reshape(16,3)
        p = np.asarray(np.load(ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy',mmap_mode='r')[:,:16],np.float64)
        prior = np.log(np.maximum(p.mean(1), 1e-12))
        frames = [pd.read_parquet(TRIAL/f'role{role}/baseline/OOF_original_rows.parquet'),
                  pd.read_parquet(proposal/'OOF_original_rows.parquet')]
        for frame in frames:
            assert np.array_equal(frame.truth.to_numpy(), gold[frame.row_position.to_numpy()])
        reports.append(dict(role=role, baseline=review(frames[0],prior,output),
                            accepted_finite_candidate=review(frames[1],prior,updated)))
    assert all(sha(ROOT/p)==value for p,value in bindings.items())
    report = dict(status='actual_saved_readout_prior_override_necessary_bounds_verified',roles=reports,
                  official_calls=0,new_fits=0,permanent_updates=0,quality_acceptance=False,
                  scope='Necessary condition at the actual fixed output weights. Not permanent model capacity, input impossibility, or independent transfer evidence.')
    save(OUT/'review.json', report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    try: main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT/'failure.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
