"""Frozen, preregistered batch-mechanism confirmation; never fit a classifier."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import math
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from v116_preflight import ROOT, DEST as TRAINED, VIEW, save, sha
from v117_gradient_batch_audit import stratified
from v104_phase_b import SparseTabM, csr_tensor, K, SEED
from v75_views import BYTE_FEATURES

DEST = ROOT / 'artifacts/v118_retrospective_and_training_contract_20260929'
CASES = [(0, 25), (1, 25), (2, 25), (1, 15)]


def main():
    assert not DEST.exists(), 'No reselecting cases or overwriting evidence.'
    prior = ROOT / 'artifacts/v117_selection_failure_review_20260929/review_receipt.json'
    for rel, digest in json.loads(prior.read_text(encoding='utf-8'))['artifact_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    sources = [Path(__file__), ROOT / 'training/v117_gradient_batch_audit.py', prior, VIEW,
               TRAINED / 'inner_split_manifest.parquet']
    sources += [TRAINED / f'outer_fold{f}/epoch{e}_model.pt' for f, e in CASES]
    DEST.mkdir()
    save(DEST / 'probe_registration.json', {'cases': CASES, 'classifier_fits': 0, 'optimizer_steps': 0,
        'purpose': 'Check whether the V117 single-checkpoint mechanism holds across all three final folds and an earlier saved state.',
        'batch_seed': '10201+fold, exactly one permutation per case',
        'no_cherry_picking': 'Report every case. Do not pick fold-specific batches, epochs or weights. Variance reduction is not classification gain.',
        'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}})
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = sparse.load_npz(VIEW)
    manifest = pd.read_parquet(TRAINED / 'inner_split_manifest.parquet')
    results = []
    for fold, epoch in CASES:
        d = manifest[manifest.outer_fold.eq(fold) & manifest.fold.ne(fold)]
        counts = np.bincount(d.local.to_numpy() * 3 + d.truth.to_numpy(), minlength=x.shape[0] * 3).reshape(-1, 3)
        used = np.flatnonzero(counts.sum(1)); batches = math.ceil(len(used) / 256)
        order = used[np.random.default_rng(SEED + fold).permutation(len(used))]
        original = [(order[i:i+256], counts[order[i:i+256]].astype(float)) for i in range(0, len(order), 256)]
        candidate = stratified(counts, batches, np.random.default_rng(SEED + fold))
        model = SparseTabM().cuda()
        state = torch.load(TRAINED / f'outer_fold{fold}/epoch{epoch}_model.pt', map_location='cpu', weights_only=True)['state_dict']
        model.load_state_dict(state); model.eval()
        parameters = list(model.parameters())
        measured, means = {}, {}
        for name, plan in [('A', original), ('B', candidate)]:
            actual = np.zeros_like(counts, dtype=float)
            for ids, mass in plan:
                actual[ids] += mass
            assert len(plan) == batches and np.allclose(actual, counts, rtol=0, atol=1e-8)
            total = [torch.zeros_like(p) for p in parameters]
            square, ce = 0., 0.
            for ids, weight in plan:
                block = x[ids]
                fact = torch.as_tensor(block[:, BYTE_FEATURES:].toarray(), device='cuda')
                mass = torch.as_tensor(weight, device='cuda', dtype=torch.float32)
                model.zero_grad(set_to_none=True)
                z = model(csr_tensor(block, 'cuda'), fact)
                loss = -(torch.log_softmax(z, -1) * mass[:, None, :]).sum() / K / (len(d) / batches)
                loss.backward(); ce += float(loss.detach()) / batches
                for acc, p in zip(total, parameters):
                    grad = p.grad.detach()
                    acc.add_(grad / batches)
                    square += float(grad.double().square().sum()) / batches
            norm = sum(float(g.double().square().sum()) for g in total)
            means[name] = total
            measured[name] = {'CE': ce, 'gradient_mean_squared_norm': norm,
                              'batch_gradient_variance': max(0., square - norm),
                              'input_evaluations': sum(len(ids) for ids, _ in plan)}
        delta = sum(float((a.double() - b.double()).square().sum()) for a, b in zip(means['A'], means['B']))
        difference = math.sqrt(delta / max(measured['A']['gradient_mean_squared_norm'], 1e-30))
        assert difference <= 1e-4
        assert all(torch.equal(state[k], v.detach().cpu()) for k, v in model.state_dict().items())
        result = {'fold': fold, 'epoch': epoch, 'train_rows': len(d), 'batches': batches,
                  'original_frequency_preserved': True, 'parameters_unchanged': True,
                  'mean_gradient_relative_difference': difference, 'measurements': measured,
                  'variance_ratio_B_over_A': measured['B']['batch_gradient_variance'] / measured['A']['batch_gradient_variance']}
        results.append(result)
        save(DEST / 'probe_progress.json', results)
        print(json.dumps({k: v for k, v in result.items() if k != 'measurements'}), flush=True)
        del model, parameters, means, total
    save(DEST / 'probe_results.json', {'status': 'frozen_confirmation_complete', 'classifier_fits': 0, 'optimizer_steps': 0,
        'all_registered_cases_executed': True, 'results': results,
        'limitations': ['One permutation per fixed checkpoint; not a guarantee along an optimization trajectory.',
                       'No new class predictions were optimized or selected.', 'Floating point nondeterminism remains unresolved.'],
        'registration_sha256': sha(DEST / 'probe_registration.json')})


if __name__ == '__main__':
    main()
