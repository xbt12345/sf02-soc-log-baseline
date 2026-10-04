"""Independent official-label/count/cache replay for the V129 no-fit audit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v129_mechanism_review_20260930'
RUN = ROOT / 'artifacts/v128_nested_score_trial_20260929_r3'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    target = OUT/'verification.json'
    if target.exists():
        raise FileExistsError(target)
    a = json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    receipt = json.loads((OUT/'output_receipt.json').read_text(encoding='utf-8'))
    for rel, h in a['input_sha256'].items():
        assert sha(ROOT/rel) == h, rel
    for rel, h in receipt['output_sha256'].items():
        assert sha(OUT/rel) == h, rel
    seal = json.loads((RUN/'run_seal.json').read_text(encoding='utf-8'))
    for rel, h in seal['source_sha256'].items():
        assert sha(ROOT/rel) == h, rel
    official = pd.read_parquet(ROOT/'data/official/train.parquet', columns=['label_binary'])
    y = official.label_binary.map({'benign':0, 'malicious':1, 'suspicious':2}).to_numpy()
    d = pd.read_parquet(OUT/'row_mechanism_ledger.parquet')
    i = pd.read_parquet(OUT/'frozen_interventions.parquet')
    assert len(y) == 2056871 and len(d) == len(i) == 112807
    assert not d.row_position.duplicated().any()
    assert np.array_equal(y[d.row_position], d.truth)
    for name in ('row_position','local','root','fold','truth'):
        assert np.array_equal(d[name], i[name]), name
    assert d.groupby('root').fold.nunique().max() == 1
    rows = []
    for fold in range(3):
        m = d.fold == fold
        local = d.loc[m, 'local'].to_numpy()
        truth = d.loc[m, 'truth'].to_numpy()
        outer = np.load(ROOT/f'artifacts/v127_frozen_branch_trial_20260929/fold{fold}_base_member_logits.npy')
        teachers = [np.load(RUN/f'teacher{fold}_{j}/canonical_member_logits.npy') for j in range(3)]
        for base, z in [('outer_AH',outer), ('inner_bank_equal',np.concatenate(teachers, axis=1))] + [
                (f'inner_teacher_{j}',z) for j,z in enumerate(teachers)]:
            for arm in ('none','P_IS','P_CF'):
                extra = np.zeros((len(outer),3)) if arm == 'none' else np.load(RUN/f'fold{fold}_{arm}/epoch50_residual.npy')
                pred = softmax(z[local] + extra[local,None,:],axis=-1).mean(1).argmax(1)
                assert np.array_equal(pred, i.loc[m, base+'_'+arm]), (fold,base,arm)
                er = {str(c):{'rows':int((truth==c).sum()),'errors':int(((truth==c)&(pred!=c)).sum())} for c in (1,2)}
                item = next(r for r in a['interventions'] if r['fold']==fold and r['base']==base and r['residual']==arm)
                assert er == item['per_class']
                rows.append((fold,base,arm))
    for key, expected in a['projections'].items():
        g=d.groupby([key,'truth']).size().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
        mixed=(g[1]>0)&(g[2]>0)
        actual={'unique_keys':len(g),'mixed_keys':int(mixed.sum()),'empirical_min_errors':int(g.min(1).sum()),
                'rows_in_mixed_keys':int(g.loc[mixed].sum().sum())}
        assert actual == expected
    assert a['new_classifier_fits'] == a['optimizer_steps'] == 0
    result={'status':'v129_no_fit_evidence_independently_verified','all_checks_passed':True,
            'official_rows':len(y),'ASA_rows':len(d),'frozen_intervention_replays':len(rows),
            'input_bindings_checked':len(a['input_sha256']),
            'historical_V128_source_bindings_checked':len(seal['source_sha256']),
            'audit_sha256':sha(OUT/'audit.json'),'verifier_sha256':sha(__file__),
            'scope':'Hash identity, official original labels and frozen prediction/count replay; not training or model-quality acceptance.'}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
