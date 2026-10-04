"""Count original-input changes and actual decision flips after V113 fit."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT,save,sha
from v113_case_train import DEST,AUDIT,check


def main():
    check()
    target=DEST/'postfit_diagnosis.json'
    assert not target.exists()
    e=json.loads((DEST/'evaluation.json').read_text(encoding='utf-8'))
    d=pd.read_parquet(DEST/'OOF_ASA_decisions.parquet')
    v=pd.read_parquet(AUDIT/'interface_views.parquet',columns=['local','before','name_spans'])
    changed={}
    for row in v.itertuples(index=False):
        spans=json.loads(row.name_spans)
        changed[row.local]=any(row.before[a:b]!=row.before[a:b].lower() for a,b,_ in spans)
    d['case_input_changed']=d.local.map(changed)
    assert int(d.case_input_changed.sum())==57818
    d['decision_flipped']=d.A_N1_prediction!=d.B_case_prediction
    keys=['fold','truth','case_input_changed']
    counts=d.groupby(keys,dropna=False).agg(rows=('row_position','size'),
        flips=('decision_flipped','sum'),repaired=('repaired','sum'),
        regressed=('regressed','sum')).reset_index()
    root=d.groupby(['root','truth'],dropna=False).agg(rows=('row_position','size'),
        repaired=('repaired','sum'),regressed=('regressed','sum')).reset_index()
    result={'status':'postfit_diagnostic_no_new_fit','new_classifier_fits':0,
        'source_sha256':sha(__file__),'evaluation_sha256':sha(DEST/'evaluation.json'),
        'decision_sha256':sha(DEST/'OOF_ASA_decisions.parquet'),
        'case_views_sha256':sha(AUDIT/'interface_views.parquet'),
        'by_fold_class_actual_input_change':counts.to_dict('records'),
        'all_S_repairs_from_root29':int(d[(d.truth==2)&d.repaired].root.nunique())==1
           and int(d[(d.truth==2)&d.repaired].root.iloc[0])==29,
        'S_repairs_with_unchanged_current_input':int(((d.truth==2)&d.repaired&~d.case_input_changed).sum()),
        'S_repairs_total':int(((d.truth==2)&d.repaired).sum()),
        'M_regressions_by_root':root[(root.truth==1)&(root.regressed>0)]
           [['root','regressed']].to_dict('records'),
        'S_regressions_by_root':root[(root.truth==2)&(root.regressed>0)]
           [['root','regressed']].to_dict('records'),
        'qualitative_limit':'Input-unchanged S repairs establish a changed learned boundary, not a causal cause or transferable behavior rule.'}
    assert result['all_S_repairs_from_root29'] and result['S_repairs_with_unchanged_current_input']==276
    assert sorted(x['regressed'] for x in result['M_regressions_by_root'])==[4,8,40]
    assert sum(x['regressed'] for x in result['S_regressions_by_root'])==2
    assert e['ASA']['class_comparison']['S']['repaired']==276
    save(target,result)
    print(json.dumps({'S_repairs_from_root29':276,'S_repairs_input_unchanged':276,
        'M_regressions_by_root':result['M_regressions_by_root']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
