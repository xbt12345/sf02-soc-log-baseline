"""Disambiguate source-fold absence from absence in all official data; no fits."""
import json
import numpy as np
import pandas as pd
from v102_root_review import ROOT, OLD, DEST, behavior, sha
from v102_verify_and_coverage import graph_roots


def main():
    out=DEST/'support_expansion_detail.json'
    assert not out.exists()
    r=pd.read_parquet(ROOT/'artifacts/v75_four_arm_20260921_r2/rows.parquet',columns=['row_position','component','projection_id','route','label_index'])
    r['fid']=np.load(ROOT/'artifacts/v79_execution_20260927/row_feature_id.npy',mmap_mode='r')
    n1=pd.read_parquet(OLD/'N1_ASA_group_map.parquet')
    a=r[r.route=='asa'].merge(n1[['row_position','N1_group']],on='row_position',validate='1:1')
    roots=graph_roots(r,a);a['root']=roots[a.row_position.to_numpy()]
    ledger=pd.read_parquet(DEST/'source_error_and_support_ledger.parquet')
    srcpos=set(ledger.row_position)
    source_roots=set(a[a.row_position.isin(srcpos)].root)
    facts=pd.read_parquet(ROOT/'artifacts/v75_four_arm_20260921_r2/projections.parquet',columns=['facts'])
    keys={int(p):behavior(json.loads(facts.facts.iat[int(p)])) for p in a.projection_id.unique()}
    a['behavior']=a.projection_id.map(keys)
    # Only entirely new full-population merged units count as added support.
    new=a[~a.root.isin(source_roots)&a.behavior.notna()]
    support=new.groupby(['behavior','label_index']).root.agg(set).to_dict()
    z=ledger[ledger.support_bucket=='zero_same_class'].copy()
    z['new_non_source_same_class_roots']=[len(support.get((b,int(c)),set())) for b,c in zip(z.behavior,z.label_index)]
    z[['row_position','label_index','new_non_source_same_class_roots']].to_parquet(DEST/'new_non_source_support_rows.parquet',index=False)
    counts=[]
    for c,g in z.groupby('label_index'):
        for bucket,t in [('none',g.new_non_source_same_class_roots==0),('one',g.new_non_source_same_class_roots==1),('two_or_more',g.new_non_source_same_class_roots>=2)]:
            h=g[t];counts.append({'class':int(c),'new_support':bucket,'rows':len(h),'candidate_errors':int((h.R0_MAG_200!=h.label_index).sum())})
    grouped=a[a.behavior.notna()].groupby(['behavior','label_index']).root.nunique().unstack(fill_value=0)
    prior=pd.Series(ledger.fold.to_numpy(),index=ledger.row_position)
    old_source=a[a.row_position.isin(srcpos)].copy()
    old_source['prior_fold']=prior.loc[old_source.row_position].to_numpy()
    cross=old_source.groupby('root').prior_fold.nunique()
    result={'status':'descriptive_support_expansion_no_fit','new_non_source_support':counts,
        'complete_behavior_keys':len(grouped),
        'full_official_behavior_keys_with_M_and_S':int(((grouped.get(1,0)>0)&(grouped.get(2,0)>0)).sum()),
        'full_official_behavior_keys_two_roots_each_class':int(((grouped.get(1,0)>=2)&(grouped.get(2,0)>=2)).sum()),
        'global_duplicate_connection_roots_spanning_prior_source_folds':int((cross>1).sum()),
        'source_rows_in_these_global_roots':int(old_source.root.isin(cross[cross>1].index).sum()),
        'interpretation':'Expansion uses only official labels from entirely non-source merged units; not a gain estimate or fresh blind evaluation. Two roots is an eligibility screen, not a sufficiency proof.',
        'script_sha256':sha(__file__)}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
