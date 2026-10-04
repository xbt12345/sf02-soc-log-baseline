"""Repair missing normal-class validation coverage by moving whole source groups only."""
import hashlib
import pandas as pd
from train_v65_rank_heads import ROOT,load_fit
from v61_common import save,sha


def supported_folds(frame):
    result=frame.fold.copy()
    groups=frame.loc[frame.label.eq(0),'group'].unique().tolist()
    if len(groups)<3:raise ValueError('Not enough independent normal sources for three folds')
    ordered=sorted(groups,key=lambda g:hashlib.sha256(('v66-normal-'+str(g)).encode()).hexdigest())
    for i,g in enumerate(ordered):result.loc[frame.group.eq(g)]=i%3
    assert frame.assign(new_fold=result).groupby('group').new_fold.nunique().max()==1
    if 'body_group' in frame:
        assert frame.assign(new_fold=result).groupby('body_group').new_fold.nunique().max()==1
    for fold in range(3):
        assert frame.loc[result.eq(fold),'label'].eq(0).any()
        assert frame.loc[result.ne(fold),'label'].eq(0).any()
    return result


def main():
    frame,ctx=load_fit();new=supported_folds(frame)
    valid=ctx['neighbors']>=0
    assert ((new.to_numpy()[ctx['neighbors'].clip(min=0)]==new.to_numpy()[:,None])|~valid).all()
    result=frame[['row_position','label','group','body_group','fold']].copy().rename(columns={'fold':'old_fold'})
    result['fold']=new
    out=ROOT/'artifacts/v66_issue_resolution_20260920'
    result.to_parquet(out/'supported_fold_manifest.parquet',index=False)
    summary={'changed_rows':int(new.ne(frame.fold).sum()),'moved_source_groups':int(frame.loc[new.ne(frame.fold),'group'].nunique()),
      'validation_counts':{str(k):frame[new.eq(k)].label.value_counts().to_dict() for k in range(3)},
      'rule':'Deterministic hash order of existing normal source groups, round-robin across three folds; move entire groups including their threat rows. No score/seed search, no role crossing.',
      'warning':'This is a new adaptive development manifest. Old v65 heads are not valid fold-matched models for this manifest and must not be reused as such.',
      'source_sha256':sha(__file__)}
    save(out/'supported_folds.json',summary);print(summary)


if __name__=='__main__':main()
