"""Recount original labels and CE coefficient masses; no model fit or updates."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v133_training_design_review_20260930'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Preserve prior evidence')
    prev = ROOT / 'artifacts/v132_training_mastery_review_20260930'
    receipt = json.loads((prev/'output_receipt.json').read_text(encoding='utf-8'))
    for name, expected in receipt['output_sha256'].items():
        assert sha(prev/name) == expected, name
    d = pd.read_parquet(prev/'training_mastery_ledger.parquet')
    held = pd.read_parquet(ROOT/'artifacts/v131_learning_trial_20260930/ASA_prediction_ledger.parquet')
    official = pq.read_table(ROOT/'data/official/train.parquet', columns=['label_binary']).column(0).to_pandas()
    truth = official.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert len(truth) == 2056871 and not pd.isna(truth).any()
    for frame in (d, held):
        assert np.array_equal(frame.truth, truth[frame.row_position.to_numpy()])
    weights=[]
    for fold in (0,1,2):
        f=d[d.arm.eq('R') & d.outer_fit_role.eq(fold)].copy()
        assert not f.row_position.duplicated().any()
        mixed=f.groupby('canonical_key').truth.nunique()
        pure=f.canonical_key.map(mixed).eq(1)
        assert np.array_equal(pure,f.pure_TRAIN_input)
        n=len(f);nm=int((pure & f.truth.eq(1)).sum());ns=int((pure & f.truth.eq(2)).sum())
        # Original row coefficient 1/N; O has .5/N + .25/N_pure_c on pure inputs.
        coef=np.full(n,.5/n)
        coef[pure & f.truth.eq(1)]+=.25/nm
        coef[pure & f.truth.eq(2)]+=.25/ns
        assert abs(coef.sum()-1)<1e-12
        weights.append({'fold':fold,'original_rows':n,'M_rows':int(f.truth.eq(1).sum()),
            'S_rows':int(f.truth.eq(2).sum()),'pure_M_rows':nm,'pure_S_rows':ns,
            'original_S_coefficient_mass':float(f.truth.eq(2).mean()),
            'O_S_coefficient_mass':float(coef[f.truth.eq(2)].sum()),
            'O_M_coefficient_mass':float(coef[f.truth.eq(1)].sum()),
            'pure_M_per_row_coefficient_multiplier':.5+.25*n/nm,
            'pure_S_per_row_coefficient_multiplier':.5+.25*n/ns,
            'mixed_input_M_S_relative_weights_changed':False,
            'scope':'CE coefficient accounting, not measured gradient amplification or security truth prior.'})
    common=held[held.fold.eq(1)]
    pair=[]
    for cls in (1,2):
        q=common[common.truth.eq(cls)]
        r=q.pred_R.ne(cls);o=q.pred_O.ne(cls)
        pair.append({'truth':cls,'support':len(q),'R_errors':int(r.sum()),'O_errors':int(o.sum()),
            'O_repairs_vs_R':int((r&~o).sum()),'O_new_errors_vs_R':int((~r&o).sum()),
            'scope':'Matched fold1 new fits only; no mixed old/new whole-task comparison.'})
    OUT.mkdir()
    sources=[Path(__file__),prev/'output_receipt.json',prev/'training_mastery_ledger.parquet',
             ROOT/'data/official/train.parquet',ROOT/'artifacts/v131_learning_trial_20260930/ASA_prediction_ledger.parquet',
             ROOT/'artifacts/v131_learning_trial_20260930/delivery.json',ROOT/'training/v131_model.py',
             ROOT/'training/review_policy/v132_training_mastery_plan.json']
    audit={'status':'zero_fit_design_audit','new_fits':0,'new_updates':0,'roles':weights,
           'fold1_R_O_paired_errors':pair,'classification_causality_proven':False,
           'limits':['O changes the training objective; greater coefficient mass is not greater independent evidence.',
                     'The historical held fold is reused design evidence, not a new unseen test.',
                     'Restricted pure-input weighting does not justify a global prior correction.'],
           'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources}}
    (OUT/'supervision_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(weights).to_csv(OUT/'supervision_coefficients.csv',index=False)
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':
    main()
