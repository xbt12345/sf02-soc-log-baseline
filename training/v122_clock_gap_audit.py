"""Audit a concrete header-parser gap without changing any historical input/model.

Both patched-header and date-change inferences are frozen diagnostics only.
No fit, no new label, no prediction promotion. All raw bodies remain untouched.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from v122_evidence_review import ROOT, DEST, PREV, TRACE, sha, save
from v75_views import SYSLOG, view, byte_matrix, BYTE_FEATURES, matrix_hashes
from v75_corrective import stable
from v99_normalization_feasibility import normalize
from v104_phase_b import SparseTabM, predict_all, DEVICE

IDENTITY_CLOCK = r'(?:(?:USER|HOST|CRED|ORG)-)+[0-9]+(?:-[0-9]+)*'
HEADER = re.compile(r'^<\d{1,3}>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(?:(?:\d{4}|' + IDENTITY_CLOCK + r')\s+)?(?:\d{2}:\d{2}:\d{2}|' + IDENTITY_CLOCK + r'):?\s*(?=USER-0010-0324\s+Deny\b)')
VIEW = ROOT/'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'


def canonical(raw):
    return normalize(stable(view(raw)[0]), 'placeholder_cluster')


def main():
    target = DEST/'clock_gap_audit.json'
    if target.exists():
        raise FileExistsError(target)
    d = pd.read_parquet(DEST/'row_diagnosis.parquet')
    trace = pd.read_parquet(TRACE)
    x = sparse.load_npz(VIEW)
    headers = trace.raw_message.map(HEADER.match)
    assert headers.notna().all()
    gap = trace.raw_message.map(lambda s: SYSLOG.match(s) is None)
    q = d[gap].copy()
    q['raw_message'] = trace.loc[gap, 'raw_message']
    texts, patched, dates = [], [], []
    for raw in q.raw_message:
        m = HEADER.match(raw)
        assert m and 'USER-' in m[0] and raw[m.end():].startswith('USER-0010-0324 ')
        # The only recognized new grammar is the already-sanitized clock slot.
        assert re.search(IDENTITY_CLOCK + r':?\s*$', m[0])
        body = raw[m.end():]
        fixed = ' <ABSOLUTE_CLOCK> ' + view(body)[0]
        texts.append(canonical(raw))
        patched.append(normalize(stable(fixed), 'placeholder_cluster'))
        variant = '<164>Jan 01 2000 USER-1856: ' + body
        dates.append(canonical(variant))
        # This candidate leaves the exact body bytes untouched and is idempotent
        # under the current canonical identity-wrapper normalization.
        assert raw == m[0] + body
        assert normalize(patched[-1], 'placeholder_cluster') == patched[-1]
    old_byte = byte_matrix(texts)
    base = x[q.local.to_numpy()]
    diff = old_byte-base[:, :BYTE_FEATURES]
    assert not diff.nnz or np.abs(diff.data).max() < 1e-7
    candidate = sparse.hstack([byte_matrix(patched), base[:, BYTE_FEATURES:]], format='csr')
    date_variant = sparse.hstack([byte_matrix(dates), base[:, BYTE_FEATURES:]], format='csr')
    assert (candidate[:, BYTE_FEATURES:]-base[:, BYTE_FEATURES:]).nnz == 0
    hashes = matrix_hashes(x)
    full_hash = [hashes[i] for i in d.local]
    after = full_hash.copy()
    for i, digest in zip(q.index, matrix_hashes(candidate)):
        after[i] = digest
    floor = {}
    for name, identities in [('original_N1', full_hash), ('patched_header', after)]:
        keys, _ = pd.factorize(identities)
        frame = d[['root', 'fold', 'truth']].copy(); frame['key'] = keys
        label_counts = pd.crosstab(frame.key, frame.truth)
        cross = frame.groupby('key').fold.nunique()
        cross_keys = cross[cross.gt(1)].index
        floor[name] = {'empirical_minimum_errors': int((label_counts.sum(1)-label_counts.max(1)).sum()),
            'mixed_label_keys': int(label_counts.gt(0).sum(1).gt(1).sum()),
            'unique_inputs': int(frame.key.nunique()), 'cross_fold_input_keys': len(cross_keys),
            'cross_fold_rows': int(frame.key.isin(cross_keys).sum())}
        if name == 'patched_header':
            frame['row_position'] = d.row_position
            frame.to_parquet(DEST/'patched_header_input_groups.parquet', index=False)
    q['patched_text'] = patched
    q['original_N1_text'] = texts
    q['diagnostic_date_variant_text'] = dates
    for arm in ('A', 'B'):
        for kind in ('patched', 'date_variant'):
            q[f'{arm}_{kind}_prediction'] = -1
        for fold in range(3):
            state_path = PREV/f'fold{fold}_{arm}/epoch25_model.pt'
            ckpt = torch.load(state_path, map_location='cpu', weights_only=True)
            model = SparseTabM().to(DEVICE)
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
            mask = q.fold.eq(fold).to_numpy()
            replay = predict_all(model, 'TabM', base[mask], DEVICE).argmax(1)
            assert np.array_equal(replay, q.loc[mask, f'expert_pred_{arm}'])
            for kind, data in [('patched', candidate), ('date_variant', date_variant)]:
                q.loc[mask, f'{arm}_{kind}_prediction'] = predict_all(model, 'TabM', data[mask], DEVICE).argmax(1)
            del model
    metrics = []
    for arm in ('A', 'B'):
        for label in (1, 2):
            z = q[q.truth.eq(label)]
            for kind in ('patched', 'date_variant'):
                pred = z[f'{arm}_{kind}_prediction']
                old = z[f'expert_pred_{arm}']
                metrics.append({'arm':arm,'truth':label,'variant':kind,'rows':len(z),
                    'old_errors':int(old.ne(label).sum()), 'new_errors':int(pred.ne(label).sum()),
                    'repaired':int((old.ne(label)&pred.eq(label)).sum()),
                    'regressed':int((old.eq(label)&pred.ne(label)).sum())})
    q.to_parquet(DEST/'clock_gap_rows.parquet', index=False)
    result = {'status':'confirmed_header_gap_and_frozen_intervention_only', 'classifier_fits':0,
       'all_ASA_headers_matched':len(d),'gap_rows':len(q),'gap_unique_old_local_ids':int(q.local.nunique()),
       'gap_class_counts':q.truth.value_counts().sort_index().to_dict(), 'collision_and_isolation':floor,
       'frozen_interventions':metrics,'candidate_fact_metadata_change_nnz':0,
       'body_preservation_checked_rows':len(q),'old_model_input_reconstruction_checked_rows':len(q),
       'old_model_decisions_replayed_all_six':True,
       'pre_write_issue':'An initial audit regex could backtrack and treat the year placeholder as clock when the clock used ORG-/nested CRED-. An event-body lookahead and complete observed placeholder grammar now prevent partial-header matches; no prior model or data was modified.',
       'scope':'Causal effect of these specific synthetic header changes on fixed model decisions, not causal proof about labels, not trained-candidate score.',
       'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
           [Path(__file__), TRACE, VIEW, ROOT/'training/v75_views.py', ROOT/'training/v75_corrective.py',
            ROOT/'training/v99_normalization_feasibility.py', DEST/'row_diagnosis.parquet']}}
    save(target,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
