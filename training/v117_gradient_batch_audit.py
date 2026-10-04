"""One preregistered zero-update probe of mass-preserving stratified batches."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from v116_preflight import ROOT, DEST as PREV, VIEW, save, sha
from v117_selection_failure_audit import DEST
from v104_phase_b import SparseTabM, csr_tensor, K, SEED
from v75_views import BYTE_FEATURES


def stratified(counts, batches, rng):
    """Each class has exactly its original mass/B per batch; never reweight CE."""
    plans = [dict() for _ in range(batches)]
    for cls in (1, 2):
        used = np.flatnonzero(counts[:, cls])
        used = used[rng.permutation(len(used))]
        target = float(counts[:, cls].sum()) / batches
        bucket, allocated = 0, 0.
        for idx in used:
            remaining = float(counts[idx, cls])
            while remaining > 1e-9:
                if bucket == batches - 1:
                    part = remaining
                else:
                    part = min(remaining, target - allocated)
                mass = plans[bucket].setdefault(int(idx), np.zeros(3, dtype=np.float64))
                mass[cls] += part
                remaining -= part
                allocated += part
                if allocated >= target - 1e-9 and bucket < batches - 1:
                    bucket += 1
                    allocated = 0.
    return [(np.array(sorted(p)), np.array([p[i] for i in sorted(p)])) for p in plans]


def main():
    assert not (DEST / 'gradient_batch_probe_registration.json').exists()
    ckpt = PREV / 'outer_fold1/epoch25_model.pt'
    registration = {'status': 'registered_before_gradient_measurement', 'fold': 1, 'epoch': 25,
                    'classifier_fits': 0, 'optimizer_steps': 0, 'batch_seed': SEED + 1,
                    'comparison': 'Original uniform unique-input batches vs exact class-mass stratification; original weighted CE in both.',
                    'activation_rule': 'Only recommend a matched training trial if full-parameter batch-gradient variance falls >=20%, exact input/class mass conserved, mean gradient relative discrepancy <=1e-4.',
                    'limitations': 'One fixed checkpoint probe, not a speed or generalization result. Never alter the threshold after observing this result.',
                    'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in [ckpt, VIEW, Path(__file__)]}}
    save(DEST / 'gradient_batch_probe_registration.json', registration)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = sparse.load_npz(VIEW)
    m = pd.read_parquet(PREV / 'inner_split_manifest.parquet')
    d = m[m.outer_fold.eq(1) & m.fold.ne(1)]
    counts = np.bincount(d.local.to_numpy() * 3 + d.truth.to_numpy(), minlength=x.shape[0] * 3).reshape(-1, 3)
    used = np.flatnonzero(counts.sum(1)); batches = math.ceil(len(used) / 256)
    original = []
    order = used[np.random.default_rng(SEED + 1).permutation(len(used))]
    for start in range(0, len(order), 256):
        ids = order[start:start + 256]
        original.append((ids, counts[ids].astype(np.float64)))
    alternative = stratified(counts, batches, np.random.default_rng(SEED + 1))
    plans = {'original': original, 'mass_stratified': alternative}
    for name, plan in plans.items():
        actual = np.zeros_like(counts, dtype=np.float64)
        for ids, mass in plan:
            actual[ids] += mass
        assert np.allclose(actual, counts, atol=1e-8, rtol=0), name
        assert len(plan) == batches
    model = SparseTabM().cuda()
    saved = torch.load(ckpt, weights_only=True, map_location='cpu')['state_dict']
    model.load_state_dict(saved); model.eval()
    parameters = list(model.parameters())
    outputs, means = {}, {}
    for name, plan in plans.items():
        started = time.monotonic()
        total = [torch.zeros_like(p) for p in parameters]
        squared = 0.; ce = 0.; batch_mass = []
        for ids, weight in plan:
            block = x[ids]
            facts = torch.as_tensor(block[:, BYTE_FEATURES:].toarray(), device='cuda')
            mass = torch.as_tensor(weight, dtype=torch.float32, device='cuda')
            model.zero_grad(set_to_none=True)
            z = model(csr_tensor(block, 'cuda'), facts)
            loss = -(torch.log_softmax(z, -1) * mass[:, None, :]).sum() / K / (len(d) / batches)
            loss.backward()
            ce += float(loss) / batches
            for acc, p in zip(total, parameters):
                g = p.grad.detach()
                acc.add_(g / batches)
                squared += float(g.double().square().sum()) / batches
            batch_mass.append(weight.sum(0).tolist())
        norm = sum(float(g.double().square().sum()) for g in total)
        means[name] = total
        outputs[name] = {'mean_member_CE': ce, 'mean_gradient_squared_norm': norm,
                         'mean_batch_gradient_squared_norm': squared,
                         'batch_gradient_variance': max(0., squared - norm),
                         'unique_input_forward_evaluations': sum(len(ids) for ids, _ in plan),
                         'batch_class_mass': batch_mass, 'seconds': time.monotonic() - started}
    delta = sum(float((a.double() - b.double()).square().sum()) for a, b in zip(means['original'], means['mass_stratified']))
    relative = math.sqrt(delta / max(outputs['original']['mean_gradient_squared_norm'], 1e-30))
    ratio = outputs['mass_stratified']['batch_gradient_variance'] / outputs['original']['batch_gradient_variance']
    unchanged = all(torch.equal(saved[k], v.detach().cpu()) for k, v in model.state_dict().items())
    assert unchanged
    result = {'status': 'zero_update_full_gradient_batch_probe', 'classifier_fits': 0, 'optimizer_steps': 0,
              'checkpoint_parameters_unchanged': unchanged, 'input_class_mass_exactly_conserved': True,
              'batches_per_plan': batches, 'results': outputs, 'mean_gradient_relative_discrepancy': relative,
              'gradient_variance_ratio_alternative_over_original': ratio,
              'registered_activation_passed': bool(ratio <= .8 and relative <= 1e-4),
              'registration_sha256': sha(DEST / 'gradient_batch_probe_registration.json'),
              'limits': registration['limitations']}
    save(DEST / 'gradient_batch_probe.json', result)
    print({k: v for k, v in result.items() if k != 'results'})
    print({k: {a: b for a, b in v.items() if a != 'batch_class_mass'} for k, v in outputs.items()})


if __name__ == '__main__':
    main()
