"""No-fit paired raw-wrapper audit of frozen v104 TabM25 models.

Variants are registered before scoring. They only rewrite synthetic CRED chains
inside one ASA source/destination port token. Visible numeric prefixes and all
other raw spans survive. This tests representation sensitivity, not restored
original ports, real attack labels, or retraining quality.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from run_v75 import ROOT, OUT, adapter, save, sha
from v75_views import view, byte_matrix, matrix_hashes, BYTE_FEATURES
from v75_corrective import stable
from v99_normalization_feasibility import normalize
from v104_phase_b import DEST as TRAINING, PREP, ASA_IDS, SparseTabM, predict_all, DEVICE
from v104_phase_a import FID

DEST = ROOT / 'artifacts/v106_frozen_audit_20260928'
ENDPOINT = re.compile(r'\b(?:src|dst)\s+[^\s:]+:[^\s/]+/(?P<port>[^\s]+)')
PORT = re.compile(r'(?P<prefix>[0-9]*)(?P<chain>(?:(?:CRED-)+[0-9]+(?:-[0-9]+)*)+)')
MARKER = re.compile(r'CRED-[0-9]+(?:-[0-9]+)*')
NESTED = re.compile(r'\dCRED-\dCRED-')
MODES = ('collapse', 'expand', 'renumber')


def cooked(raw):
    return normalize(stable(view(raw)[0]), 'placeholder_cluster')


def rewrite(raw, mode):
    spans = []
    for endpoint in ENDPOINT.finditer(raw):
        token = endpoint['port']
        m = PORT.fullmatch(token)
        if not m:
            continue
        body = m['chain']
        if mode == 'collapse':
            if body.count('CRED-') < 2:
                continue
            replacement = m['prefix'] + 'CRED-90001'
        elif mode == 'expand':
            replacement = token + 'CRED-90001'
        elif mode == 'renumber':
            replacement = m['prefix'] + MARKER.sub('CRED-90001', body)
        else:
            raise ValueError(mode)
        if replacement != token:
            spans.append((endpoint.start('port'), endpoint.end('port'), token, replacement))
    pieces = []; cursor = 0
    for a, b, token, replacement in spans:
        assert raw[a:b] == token
        assert PORT.fullmatch(token)['prefix'] == PORT.fullmatch(replacement)['prefix']
        pieces.extend((raw[cursor:a], replacement)); cursor = b
    pieces.append(raw[cursor:])
    new = ''.join(pieces)
    # Reconstruct the original from changed spans rather than assert semantic truth.
    back = []; cursor = 0; offset = 0
    for a, b, token, replacement in spans:
        start = a + offset
        back.extend((new[cursor:start], token)); cursor = start + len(replacement)
        offset += len(replacement) - len(token)
    back.append(new[cursor:])
    assert ''.join(back) == raw
    return new, spans


def metrics(y, before, after, p, q, mask):
    y, before, after, p, q = (a[mask] for a in (y, before, after, p, q))
    return {'rows': int(len(y)), 'before_errors': int((before != y).sum()),
            'after_errors': int((after != y).sum()), 'flips': int((before != after).sum()),
            'repairs': int(((before != y) & (after == y)).sum()),
            'regressions': int(((before == y) & (after != y)).sum()),
            'M_correct_before': int(((y == 1) & (before == 1)).sum()),
            'M_correct_after': int(((y == 1) & (after == 1)).sum()),
            'S_correct_before': int(((y == 2) & (before == 2)).sum()),
            'S_correct_after': int(((y == 2) & (after == 2)).sum()),
            'mean_abs_S_probability_change': float(np.abs(p[:, 2]-q[:, 2]).mean()) if len(y) else None,
            'max_abs_probability_change': float(np.abs(p-q).max()) if len(y) else None}


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.mkdir()
    start = time.monotonic()
    ledger_path = TRAINING/'phase_B_ASA_OOF_ledger.parquet'
    raw_path = ROOT/'data/official/train.parquet'
    models = [TRAINING/f'fold{k}_C_TabM/epoch25_model.pt' for k in range(3)]
    paths = [ledger_path, raw_path, OUT/'rows.parquet', OUT/'text_dictionary.parquet',
             OUT/'projections.parquet', PREP/'N1_ASA.npz', FID, ASA_IDS,
             ROOT/'training/v75_views.py', ROOT/'training/v75_corrective.py',
             ROOT/'training/v99_normalization_feasibility.py',
             ROOT/'training/v104_phase_b.py', *models]
    registration = {'status': 'registered_before_variant_scoring', 'fits': 0,
        'source_sha256': sha(__file__), 'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        'variants': {'collapse': 'one CRED token per already redacted single port field; visible prefix retained',
                     'expand': 'append one synthetic CRED wrapper inside already redacted single port field',
                     'renumber': 'change synthetic CRED IDs only, expected exact N1 equality'},
        'label_use': 'postfit diagnosis only; no threshold or model selected',
        'safety': 'assert raw outside spans, numeric prefixes, parser facts, and fixed fact matrix unchanged',
        'limitations': ['Synthetic ID convention is inferred from official field naming; sanitizer source is unavailable.',
                        'Partial hidden original ports cannot be recovered; semantic equivalence refers to observable facts.',
                        'All folds are previously inspected development data.',
                        'Source_symbol in v75 is structured src_ip, not body source identity.']}
    save(DEST/'registration.json', registration)
    ledger = pd.read_parquet(ledger_path)
    meta = pd.read_parquet(OUT/'rows.parquet', columns=['row_position', 'new_text_id', 'projection_id', 'source_symbol'])
    meta = meta.iloc[ledger.row_position.to_numpy()].reset_index(drop=True)
    raw = pd.read_parquet(raw_path, columns=['message_sanitized', 'src_ip']).iloc[ledger.row_position.to_numpy()].reset_index(drop=True)
    strings = raw.message_sanitized.fillna('').to_numpy()
    raw_ids, unique_raw = pd.factorize(strings, sort=False)
    dictionary = pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    ids = np.load(ASA_IDS); fid = np.load(FID, mmap_mode='r')
    mapping = {int(f): k for k, f in enumerate(ids)}
    local = np.array([mapping[int(f)] for f in fid[ledger.row_position]], dtype=np.int32)
    x = sparse.load_npz(PREP/'N1_ASA.npz')
    facts = pd.read_parquet(OUT/'projections.parquet').facts
    parser = adapter()
    text_before = [cooked(t) for t in unique_raw]
    assert all(text_before[r] == normalize(stable(dictionary.loc[int(t)]), 'placeholder_cluster')
               for r, t in zip(raw_ids, meta.new_text_id))
    baseline = sparse.hstack([byte_matrix([text_before[i] for i in raw_ids]), x[local, BYTE_FEATURES:]], format='csr')
    diff = baseline-x[local]
    assert not diff.nnz or abs(diff.data).max() < 1e-7
    del baseline
    variant_texts = {}; changed_rows = {}; provenance = []; checks = {}
    for mode in MODES:
        cooked_new = []; changed = []; facts_checks = 0
        for rid, text in enumerate(unique_raw):
            new, spans = rewrite(text, mode)
            cooked_new.append(cooked(new)); changed.append(new != text)
            if new != text:
                a = parser.prepare_record({'message_sanitized': text})
                b = parser.prepare_record({'message_sanitized': new})
                assert a['route'] == b['route'] == 'asa'
                assert a['facts'] == b['facts'], (mode, rid, a['facts'], b['facts'])
                facts_checks += 1
                provenance.append({'mode': mode, 'raw_id': rid,
                    'raw_sha256': __import__('hashlib').sha256(text.encode()).hexdigest(),
                    'spans_json': json.dumps(spans), 'before_N1': text_before[rid], 'after_N1': cooked_new[-1]})
            if mode == 'collapse':
                assert rewrite(new, mode)[0] == new
        variant_texts[mode] = cooked_new
        changed_rows[mode] = np.array(changed)[raw_ids]
        checks[mode] = {'raw_changed_rows': int(changed_rows[mode].sum()),
            'raw_changed_unique_strings': int(sum(changed)), 'parser_fact_pair_checks': facts_checks,
            'N1_text_changed_rows': int(sum(cooked_new[i] != text_before[i] for i in raw_ids))}
        print(json.dumps({'stage': 'raw_audit', 'mode': mode, **checks[mode]}), flush=True)
    assert all(a == b for a, b in zip(text_before, variant_texts['renumber']))
    pd.DataFrame(provenance).to_parquet(DEST/'variant_span_ledger.parquet', index=False)
    y = ledger.truth.to_numpy(); fold = ledger.fold.to_numpy()
    baseline_prob = np.zeros((len(ledger), 3), np.float32)
    outputs = {mode: np.zeros_like(baseline_prob) for mode in ('collapse', 'expand')}
    matrices = {}; lookups = {}; group_audits = {}
    for mode in outputs:
        pairs = list(zip(local.tolist(), [variant_texts[mode][i] for i in raw_ids]))
        pair_id, unique_pairs = pd.factorize(pd.Series(pairs), sort=False)
        newx = sparse.hstack([byte_matrix([p[1] for p in unique_pairs]),
                             x[[p[0] for p in unique_pairs], BYTE_FEATURES:]], format='csr')
        assert (newx[:, BYTE_FEATURES:] - x[[p[0] for p in unique_pairs], BYTE_FEATURES:]).nnz == 0
        hashes = matrix_hashes(newx); groups, _ = pd.factorize(hashes, sort=False)
        order = np.argsort(groups, kind='stable')
        for a, b in zip(order[:-1], order[1:]):
            if groups[a] == groups[b]:
                assert (newx[a]-newx[b]).nnz == 0
        group_rows = groups[pair_id]
        audit = pd.DataFrame({'new_input': group_rows, 'fold': fold, 'label': y,
                              'row_position': ledger.row_position.to_numpy(), 'root': ledger.root.to_numpy()})
        counts = audit.groupby(['new_input', 'label']).size().unstack(fill_value=0)
        crossing = audit.groupby('new_input').fold.nunique().gt(1)
        group_audits[mode] = {'unique_inputs': int(len(counts)),
            'mixed_label_inputs': int((counts.gt(0).sum(axis=1)>1).sum()),
            'empirical_min_errors': int((counts.sum(axis=1)-counts.max(axis=1)).sum()),
            'crossfold_equal_inputs': int(crossing.sum()),
            'crossfold_equal_rows': int(audit.new_input.isin(crossing.index[crossing]).sum())}
        audit.to_parquet(DEST/f'{mode}_input_groups.parquet', index=False)
        matrices[mode] = newx; lookups[mode] = pair_id
    for k in range(3):
        report = json.loads((TRAINING/f'fold{k}_C_TabM/fit.json').read_text())
        assert sha(models[k]) == report['checkpoint_sha256']['25']['model']
        state = torch.load(models[k], map_location='cpu', weights_only=True)
        assert state['fold'] == k and state['epoch'] == 25
        model = SparseTabM().to(DEVICE); model.load_state_dict(state['state_dict'])
        replay = predict_all(model, 'TabM', x, DEVICE)
        saved = np.load(TRAINING/f'fold{k}_C_TabM/epoch25_ASA_input_prob.npy')
        np.testing.assert_allclose(replay, saved, atol=2e-6, rtol=2e-6)
        take = fold == k; baseline_prob[take] = replay[local[take]]
        for mode, newx in matrices.items():
            selected = np.unique(lookups[mode][take])
            scores = predict_all(model, 'TabM', newx[selected], DEVICE)
            outputs[mode][take] = scores[np.searchsorted(selected, lookups[mode][take])]
        del model
        if DEVICE == 'cuda': torch.cuda.empty_cache()
        print(json.dumps({'stage': 'frozen_replay', 'fold': k}), flush=True)
    before = baseline_prob.argmax(1)
    assert np.array_equal(before, ledger.C_TabM_epoch25.to_numpy())
    nested = np.array([bool(NESTED.search(t)) for t in strings])
    strata = {'all_ASA': np.ones(len(y), bool), 'nested_pattern': nested, 'without_nested': ~nested,
              'outside_two_roots': ~ledger.root.isin([2868, 637660]).to_numpy()}
    for k in range(3): strata[f'fold{k}'] = fold == k
    for root in (2868, 637660, 27221, 3929, 46091, 2300):
        strata[f'root_{root}'] = ledger.root.eq(root).to_numpy()
    result = {'status': 'completed_frozen_model_diagnosis_no_training', 'fits': 0,
        'source_sha256': sha(__file__), 'registration_sha256': sha(DEST/'registration.json'),
        'unique_raw_messages': len(unique_raw), 'all_original_N1_vectors_reconstructed': True,
        'all_TabM25_original_scores_replayed': True, 'renumber_N1_equal': True,
        'raw_checks': checks, 'input_equivalence_audit': group_audits,
        'variants': {}, 'limitations': registration['limitations'], 'model_promoted': False}
    paired = ledger[['row_position', 'root', 'fold', 'truth', 'C_TabM_epoch25']].copy()
    paired['original_S_prob'] = baseline_prob[:, 2]; paired['nested_pattern'] = nested
    for mode, prob in outputs.items():
        pred = prob.argmax(1)
        result['variants'][mode] = {name: metrics(y, before, pred, baseline_prob, prob, mask)
            for name, mask in dict(strata, raw_changed=changed_rows[mode]).items()}
        paired[mode+'_pred'] = pred; paired[mode+'_S_prob'] = prob[:, 2]
        paired[mode+'_raw_changed'] = changed_rows[mode]
    source_check = pd.DataFrame({'symbol': meta.source_symbol, 'structured_src_ip': raw.src_ip})
    result['source_symbol_origin_check'] = {
        'symbol_to_structured_src_ip_max_unique': int(source_check.groupby('symbol').structured_src_ip.nunique(dropna=False).max()),
        'structured_src_ip_to_symbol_max_unique': int(source_check.groupby('structured_src_ip', dropna=False).symbol.nunique().max()),
        'origin': 'run_v75.py prepare maps raw src_ip to source_symbol; not message-body source'}
    paired.to_parquet(DEST/'paired_OOF_predictions.parquet', index=False)
    result['elapsed_seconds'] = time.monotonic()-start
    result['outputs_sha256'] = {p.name: sha(p) for p in DEST.glob('*.parquet')}
    assert all(sha(ROOT/p) == digest for p, digest in registration['input_sha256'].items())
    result['all_bound_inputs_unchanged_after_audit'] = True
    save(DEST/'diagnosis.json', result)
    print(json.dumps({'output': str(DEST/'diagnosis.json'), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__': main()
