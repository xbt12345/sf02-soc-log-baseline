"""Independent offline input/derivative-support and actual prepared callgraph review."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from experiment_review import ROOT, sha, check_bindings

OUT = ROOT / 'artifacts/v169_root_input_support_and_callgraph_review_20261002'


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    xfile = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    x = load_npz(xfile).tocsr()
    assert x.shape == (22546, 66287) and x.has_canonical_format
    assert x.nnz == 4006092
    paths = {Path(__file__).resolve(), xfile,
             ROOT / 'training/v169_prior_pair_training_entry_v3.py',
             ROOT / 'training/v169_pair_lifecycle_v2.py',
             ROOT / 'training/v169_prior_pair_model.py',
             ROOT / 'artifacts/v169_saved_budget_scope_and_callgraph_v2_20261002/review.json'}
    roles, totals = [], {k:0 for k in ['heads', 'features', 'fixed_target_derivatives',
                                      'margin_derivatives', 'QP', 'proposals', 'fits', 'updates']}
    for role in range(3):
        rowfile = ROOT / f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/legal_FIT_reference.parquet'
        targetfile = ROOT / f'artifacts/v161_independent_frozen_error_cohort_review_20261002/role{role}/target_original_counts.npy'
        paths.update([rowfile, targetfile])
        rows = pd.read_parquet(rowfile)
        target = np.load(targetfile)
        assert rows.row_position.is_unique and target.shape == (22546, 3)
        chunks = (rows.local.nunique()+2047)//2048
        gradient_calls = 21*2*2
        margin_calls = 19*2*64*2
        proposals = 1+19*(8+2)
        heads = gradient_calls*chunks+12+(proposals+1)*(chunks+12)+margin_calls
        caps = dict(heads=int(heads), features=int(heads), fixed_target_derivatives=gradient_calls,
                    margin_derivatives=margin_calls, QP=19*3, proposals=proposals, fits=1, updates=20)
        for key,value in caps.items(): totals[key] += 2*value
        supports = []
        for cls in [1,2]:
            ids = np.flatnonzero(target[:,cls]>0)
            columns = np.unique(x[ids].indices)
            supports.append(dict(class_id=cls, target_locals=len(ids),
                original_target_rows=int(target[:,cls].sum()), union_observation_columns=len(columns),
                conditional_complete_B_parameter_support_bound=16*len(columns)+241))
        roles.append(dict(role=role, original_rows=len(rows), unique_locals=int(rows.local.nunique()),
                          OOF_chunks=int(chunks), per_fit_caps=caps, target_support=supports))
    assert totals == dict(heads=56328, features=56328, fixed_target_derivatives=504,
                         margin_derivatives=29184, QP=342, proposals=1146, fits=6, updates=120)
    refsfile = ROOT / 'artifacts/v168_decision_floor_diagnostic_20261002/role1/correction0/active_normal_references.json'
    paths.add(refsfile)
    refs = json.loads(refsfile.read_text(encoding='utf-8'))
    union = np.zeros(1060832, bool)
    for ref in refs.values():
        path = ROOT / ref['gradient']
        paths.add(path)
        g = np.load(path)
        assert g.shape == union.shape and g.dtype == np.float64 and np.isfinite(g).all()
        union |= g.view(np.uint64) != 0
    assert len(refs) == 25 and int(union.sum()) == 5120
    bindings = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    worker = json.loads((ROOT / 'artifacts/v169_saved_budget_scope_and_callgraph_v2_20261002/review.json').read_text(encoding='utf-8'))
    assert all(worker['new_prepared_entry_caps'][k] == v for k,v in totals.items())
    assert worker['full_resource_budget_and_registration_not_ready']
    check_bindings(bindings)
    OUT.mkdir()
    report = dict(status='V169_prepared_callgraph_original_input_and_saved_support_independently_recomputed',
        roles=roles, new_caps=totals, all_new_complete_derivatives=29688,
        actual_input_maximum_row_nnz=int(np.diff(x.indptr).max()),
        conditional_B_single_query_parameter_support_bound=3617,
        saved_V168_role1_normals=25, saved_nonzero_bit_coordinate_union=5120,
        structural_support_not_official_runtime_qualification=True,
        full_resource_budget_still_not_ready=True, free_disk_bytes=shutil.disk_usage(ROOT).free,
        official_heads=0, official_features=0, official_derivatives=0, fits=0,
        permanent_updates=0, supports_physical_seal=False,
        source_sha256=bindings,
        scope='Only original CSR, registered original counts/roles, saved vectors and prepared source. Support bounds must retain every bit including negative zero and be qualified/enforced in the actual backend; no new model/gradient/proposal was executed.')
    (OUT/'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','roles']}, ensure_ascii=False))


if __name__ == '__main__': main()
