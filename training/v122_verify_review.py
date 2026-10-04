"""Independent arithmetic and immutable source verification for V122 review."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v122_evidence_review import ROOT, DEST, sha, read, save


def main():
    target=DEST/'verification.json'
    if target.exists():raise FileExistsError(target)
    counts={}
    for name,paths in [('v121_delivery',read(ROOT/'artifacts/v121_paired_batch_training_20260929/delivery.json')['artifact_sha256']),
                       ('v122_plan',read(ROOT/'training/review_policy/v122_next_plan.json')['evidence_sha256'])]:
        for rel,h in paths.items():assert sha(ROOT/rel)==h,rel
        counts[name]=len(paths)
    for name in ['review.json','clock_gap_audit.json','date_only_probe.json']:
        for rel,h in read(DEST/name)['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    catalog=read(ROOT/'mcp_readonly/catalog.json')
    for e in catalog['documents']:assert sha(ROOT/e['path'])==e['sha256'],e['id']
    counts['catalog']=len(catalog['documents'])
    d=pd.read_parquet(DEST/'row_diagnosis.parquet')
    q=pd.read_parquet(DEST/'clock_gap_rows.parquet')
    v=pd.read_parquet(DEST/'strict_date_only_predictions.parquet')
    assert not d.row_position.duplicated().any() and len(d)==112807
    assert q.row_position.tolist()==v.row_position.tolist() and len(q)==682
    for arm in ['A','B']:
        g=d.assign(ok=d[f'expert_pred_{arm}'].eq(d.truth)).groupby(['root','truth']).ok.agg(['mean','sum'])
        for cls in [1,2]:
            z=g.xs(cls,level='truth')
            expected=[x for x in read(DEST/'review.json')['group_metrics'] if x['arm']==arm and x['truth']==cls][0]
            assert abs(z['mean'].mean()-expected['group_macro_recall'])<1e-12
            assert int(z['sum'].eq(0).sum())==expected['zero_recall_roots']
    assert sum((d.truth==2)&d.A_all_seven_wrong)==2236
    for card in read(DEST/'case_cards.json')['cards']:
        assert {e['truth'] for e in card['fit_examples']}=={1,2}
        assert all(e['fold']!=card['target']['fold'] for e in card['fit_examples'])
    inputs=pd.read_parquet(DEST/'patched_header_input_groups.parquet')
    assert np.array_equal(inputs.row_position,d.row_position)
    assert inputs.groupby('key').fold.nunique().max()==1
    ct=pd.crosstab(inputs.key,inputs.truth)
    assert int((ct.sum(1)-ct.max(1)).sum())==28
    bindings={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in DEST.iterdir() if p.is_file()}
    for rel in ['docs/V122_COMPLETE_REVIEW_AND_EVIDENCE_FIRST_PLAN.md','training/review_policy/v122_next_plan.json',
                'training/v122_verify_review.py']:
        bindings[rel]=sha(ROOT/rel)
    result={'status':'v122_evidence_and_publication_verified','new_classifier_fits':0,'optimizer_steps':0,
       'latest_actual_training':'V121','current_direction':'V122','classification_quality_passed':False,
       'verified':counts,'artifact_sha256':bindings,
       'publication_issues_resolved':['Explicit UTF-8 required for JSON reads on this Windows host.',
            'Mutable training-plan catalog hash refreshed after prepending V122 direction; historical delivery hashes unchanged.'],
       'scope':'Review integrity, independent ledger arithmetic and source publication only; not a trained V122 model or external test.'}
    save(target,result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
