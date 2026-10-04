"""Verify finished development outputs and the frozen model input boundary."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from v61_common import read, save, sha
from train_v63_factorial import ROOT, ARMS, load_development, inputs_for
from analyze_v63_factorial import paired_source


def main():
    run = ROOT/'artifacts/v63_factorial_20260920'
    analysis = read(run/'analysis.json')
    assert analysis['registered_screen_candidates'] == []
    frame, pool, context = load_development()
    changed = frame.copy()
    raw_fields = ['timestamp', 'product_name', 'vendor_name', 'src_ip', 'dst_ip',
                  'src_host', 'dst_host', 'username', 'event_id', 'pipeline']
    for name in raw_fields:
        changed[name] = ['arbitrary_replacement_' + str(i % 13) for i in range(len(frame))]
    changed['group'] = 'arbitrary_subject'
    changed['label'] = (changed['label'] + 1) % 3
    boundary = {}
    torch.set_num_threads(4)
    for arm in ['A0', 'P0']:
        base = inputs_for(frame, context, torch.device('cpu'), arm)
        altered = inputs_for(changed, context, torch.device('cpu'), arm)
        assert base.vocab == altered.vocab
        for name in ['facts', 'text_codes']:
            assert torch.equal(getattr(base, name), getattr(altered, name)), name
        for name in base.tokens:
            assert torch.equal(base.tokens[name], altered.tokens[name]), name
        assert not torch.equal(base.y, altered.y)
        boundary[arm] = {'rows':len(frame), 'token_and_fact_inputs_identical':True,
                         'metadata_and_target_excluded_from_forward_inputs':True}
        del base, altered
    dev = frame[frame.role.eq('selection')].reset_index(drop=True)
    base = None; direct = {}; checks = {}
    for arm in ARMS:
        selected = analysis['selected'][arm]
        path = run/arm/selected['checkpoint']['model_file']
        assert sha(path) == selected['model_sha256']
        predictions = run/arm/selected['checkpoint']['predictions_file']
        assert sha(predictions) == selected['predictions_sha256']
        d = pd.read_parquet(predictions).sort_values('row_position').reset_index(drop=True)
        assert d[['row_position','group','label']].equals(dev[['row_position','group','label']])
        p = d[['p_B','p_M','p_S']].to_numpy()
        if base is None: base = p
        direct[arm] = paired_source(dev,base,p)
        checks[arm] = {'model_sha256':sha(path), 'prediction_sha256':sha(predictions),
                       'probability_sum_max_abs_error':float(np.abs(p.sum(1)-1).max())}
    suite = unittest.defaultTestLoader.discover(str(ROOT/'training'),pattern='test_v63*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    save(run/'verification.json', {
        'all_checks_passed':True, 'quality_acceptance':False,
        'source_sha256':sha(__file__), 'tests_run':result.testsRun,
        'input_hashes_reverified':True, 'input_boundary':boundary,
        'perturbed_metadata_fields':raw_fields+['group','label'],
        'boundary_scope':'Frozen records to model inputs only; does not retest raw-log preprocessing or label-dependent training/split construction.',
        'selected_artifact_identity':checks, 'paired_source_against_A0':direct,
        'scope':'Implementation, artifact identity and adaptive development diagnostics. No external validation or live ChatGPT tunnel test.'})
    print('DELIVERY_VERIFICATION_COMPLETE',flush=True)


if __name__ == '__main__': main()
