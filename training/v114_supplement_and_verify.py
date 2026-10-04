"""Independent recount of frozen diagnostic conclusions, no fitting."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from run_v75 import ROOT, save, sha
from v114_mechanism_audit import DEST


def main():
    j = json.loads((DEST/'diagnosis.json').read_text(encoding='utf-8'))
    v = json.loads((DEST/'verification.json').read_text(encoding='utf-8'))
    assert j['source_sha256'] == sha(ROOT/'training/v114_mechanism_audit.py')
    for path, h in v['artifact_sha256'].items():
        assert sha(ROOT/path) == h, path
    d = pd.read_parquet(ROOT/'artifacts/v113_case_training_20260929/OOF_ASA_decisions.parquet')
    r = pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet')
    assert np.array_equal(d.row_position, r.row_position)
    y = d.truth.to_numpy(); pp = np.load(DEST/'frozen_cross_input_probabilities.npz')
    assert not (DEST/'supplement.json').exists()
    for name in pp.files:
        p = pp[name]; pred = p.argmax(1)
        assert int((pred != y).sum()) == j['metrics_2x2'][name]['all']['errors']
        assert not (pred == 0).any()
        fpr, tpr, _ = roc_curve(y == 2, p[:, 2]/p[:, 1:].sum(1), drop_intermediate=False)
        fp = np.rint(fpr*(y == 1).sum()).astype(int)
        tp = np.rint(tpr*(y == 2).sum()).astype(int)
        assert int(tp[fp <= 318].max()) == j['oracle_not_a_candidate'][name]['maximum_S_correct']
    totals = pd.read_csv(DEST/'training_only_support.csv')
    fact = totals[(totals.key == 'fact_no_srcport')&(totals.state == 'B_error')]
    assert int(fact[fact.truth == 2].zero_same_class_roots.sum()) == 1408
    assert int(fact[fact.truth == 1].zero_same_class_roots.sum()) == 314
    cells = j['apparent_supported_S_error_cells']
    assert sum(c['error_rows'] for c in cells) == 190
    assert sum(c['error_rows'] for c in cells if c['known_exact_protocol_parameter']) == 14
    repaired = (y == 2)&d.repaired.to_numpy()
    src_tokens = r.loc[repaired, 'raw_message'].str.extract(r'\bsrc\s+[^\s:]+:[^\s/]+/([^\s]+)')[0]
    nested = src_tokens.str.count('CRED-').ge(2)
    assert int(nested.sum()) == 276
    assert int(src_tokens.eq('5CRED-2CRED-24947').sum()) == 272
    result = {'status': 'supplement_independently_recounted_no_fit', 'classifier_fits': 0,
        'source_sha256': sha(__file__), 'diagnosis_sha256': sha(DEST/'diagnosis.json'),
        'S_repairs_total': int(repaired.sum()), 'S_repairs_nested_CRED_raw_src_port': int(nested.sum()),
        'S_repairs_one_literal_src_port_token': int(src_tokens.eq('5CRED-2CRED-24947').sum()),
        'S_repairs_raw_source_addresses': int(r.loc[repaired, 'raw_message'].str.extract(r'\bsrc\s+[^\s:]+:([^\s/]+)')[0].nunique()),
        'S_errors_no_same_observed_fact_profile_training_root': 1408,
        'M_errors_no_same_observed_fact_profile_training_root': 314,
        'S_errors_apparent_multiple_root_support': 190,
        'of_these_missing_destination_parameter': 176,
        'of_these_known_destination_parameter': 14,
        'checks': {'bound_artifacts_match': True, 'four_frozen_error_totals_recounted': True,
            'all_four_threshold_oracles_independently_recounted_with_ROC': True,
            'support_missingness_distinction_recounted': True, 'repair_redaction_pattern_recounted': True},
        'all_checks_passed': True,
        'scope': 'Diagnostic arithmetic and artifact identity; correlation is not causal proof, no classifier quality acceptance.',
        'preliminary_audit_note': 'The first draft support table conflated incomplete strict keys with zero support. It is superseded by the r2 directory. No old training files were changed.'}
    save(DEST/'supplement.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
