"""Inspect fixed model members and deterministic runtime; zero optimizer steps."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from v116_preflight import ROOT, DEST as PREV, VIEW, save, sha
from v117_selection_failure_audit import DEST
from v104_phase_b import SparseTabM, csr_tensor, SEED, K
from v75_views import BYTE_FEATURES


def main():
    assert not (DEST / 'frozen_member_audit.json').exists()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = sparse.load_npz(VIEW)
    m = pd.read_parquet(PREV / 'inner_split_manifest.parquet')
    d = m[m.outer_fold.eq(1)].reset_index(drop=True)
    # Repeat exactly one unchanged model and batch; no optimizer is constructed.
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = SparseTabM().cuda()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    fit = d.fold.ne(1).to_numpy()
    count = np.bincount(d.loc[fit, 'local'].to_numpy() * 3 + d.loc[fit, 'truth'].to_numpy(),
                        minlength=x.shape[0] * 3).reshape(-1, 3)
    used = np.flatnonzero(count.sum(1))
    batch = used[np.random.default_rng(SEED + 1).permutation(len(used))[:256]]
    block = x[batch]
    facts = torch.as_tensor(block[:, BYTE_FEATURES:].toarray(), device='cuda')
    mass = torch.as_tensor(count[batch], device='cuda', dtype=torch.float32)
    repeats, reference, first = [], None, None
    try:
        for n in range(3):
            model.zero_grad(set_to_none=True)
            z = model(csr_tensor(block, 'cuda'), facts)
            (-(torch.log_softmax(z, -1) * mass[:, None, :]).sum() / K).backward()
            grad = {k: v.grad.detach().cpu().clone() for k, v in model.named_parameters()}
            current = z.detach().cpu().clone()
            if reference is None:
                reference, first = grad, current
            else:
                repeats.append({'logits_bitwise_equal': torch.equal(current, first),
                                'all_gradients_bitwise_equal': all(torch.equal(grad[k], reference[k]) for k in grad),
                                'max_gradient_abs_difference': max(float((grad[k] - reference[k]).abs().max()) for k in grad)})
        deterministic = {'supported_for_this_probe': True, 'repeats': repeats}
    except RuntimeError as e:
        deterministic = {'supported_for_this_probe': False, 'error': str(e)}
    assert all(torch.equal(v.detach().cpu(), state[k]) for k, v in model.state_dict().items())
    # Even a successful probe is not a complete repeated-training guarantee.
    sources = [Path(__file__), PREV / 'inner_split_manifest.parquet', VIEW]
    rows = []
    for fold in range(3):
        p = PREV / f'outer_fold{fold}/epoch25_model.pt'
        checkpoints = json.loads((p.parent / 'checkpoints.json').read_text(encoding='utf-8'))
        assert sha(p) == next(c for c in checkpoints if c['epoch'] == 25)['model_sha256']
        model.load_state_dict(torch.load(p, map_location='cpu', weights_only=True)['state_dict'])
        model.eval()
        member_pred, average = [], []
        with torch.no_grad():
            for start in range(0, x.shape[0], 512):
                block = x[start:start + 512]
                fact = torch.as_tensor(block[:, BYTE_FEATURES:].toarray(), device='cuda')
                probs = model(csr_tensor(block, 'cuda'), fact).softmax(-1)
                member_pred.append(probs.argmax(-1).cpu().numpy())
                average.append(probs.mean(1).cpu().numpy())
        members = np.concatenate(member_pred)
        averaged = np.concatenate(average)
        original = np.load(p.parent / 'epoch25_prob.npy')
        assert np.allclose(averaged, original, atol=2e-6)
        assert np.array_equal(averaged.argmax(1), original.argmax(1))
        sources.extend([p, p.parent / 'epoch25_prob.npy'])
        for role, q in [('fit', d[d.fold.ne(fold)]), ('outer_test', d[d.fold.eq(fold)])]:
            c = pd.crosstab(q.local, q.truth).reindex(columns=[1, 2], fill_value=0)
            empirical_floor = int(c.sum(1).sum() - c.max(1).sum())
            for cls in (1, 2):
                z = q[q.truth.eq(cls)]
                local = z.local.to_numpy()
                correct = averaged[local].argmax(1) == cls
                votes = (members[local] == cls).sum(1)
                counts = np.bincount(votes[~correct], minlength=17)
                conflicting = set(c.index[(c[1] > 0) & (c[2] > 0)])
                rows.append({'fold': fold, 'role': role, 'truth': cls, 'rows': len(z),
                    'errors': int((~correct).sum()), 'error_rows_with_no_correct_member': int(((votes == 0) & ~correct).sum()),
                    'error_rows_with_any_correct_member': int(((votes > 0) & ~correct).sum()),
                    'error_rows_with_majority_members_correct': int(((votes > 8) & ~correct).sum()),
                    'correct_member_vote_histogram_on_errors': counts.tolist(),
                    'error_rows_on_same_input_mixed_labels': int((z.local.isin(conflicting).to_numpy() & ~correct).sum()),
                    'population_empirical_same_input_floor': empirical_floor})
    output = {'status': 'frozen_members_and_determinism_probe', 'new_classifier_fits': 0, 'optimizer_steps': 0,
              'deterministic_probe': deterministic, 'unchanged_probe_parameters': True,
              'torch': torch.__version__, 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
              'replayed_outer25_probabilities_and_decisions_match': True,
              'member_audit': rows,
              'limits': ['Member oracle uses true labels and is diagnostic, never a deployable routing rule.',
                        'Same-batch determinism is not complete training or cross-device reproducibility.',
                        'Empirical same-input floor is not irreducible population or Bayes error.',
                        'No-correct-member errors do not by themselves distinguish optimization, representation and target identifiability.'],
              'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}}
    save(DEST / 'frozen_member_audit.json', output)
    print(json.dumps({'deterministic_probe': deterministic, 'member_audit': rows}, ensure_ascii=False))


if __name__ == '__main__':
    main()
