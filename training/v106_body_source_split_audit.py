"""Audit observed ASA body-source overlap; propose closure without fitting."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from run_v75 import ROOT, OUT, save, sha
from v104_phase_a import FOLDS
from v106_frozen_wrapper_audit import DEST


def main():
    target=DEST/'body_source_split_audit.json'; assert not target.exists()
    d=pd.read_parquet(DEST/'paired_OOF_predictions.parquet')
    rawpath=ROOT/'data/official/train.parquet'
    raw=pd.read_parquet(rawpath,columns=['message_sanitized']).iloc[d.row_position].reset_index(drop=True)
    d['body']=raw.message_sanitized.str.extract(r'\bsrc\s+[^\s:]+:([^\s/]+)',expand=False)
    a=d.groupby('body').agg(folds=('fold','nunique'),roots=('root','nunique'),labels=('truth','nunique'))
    overlap=d.body.isin(a.index[a.folds>1]).to_numpy()
    performance={}
    for name,mask in [('shared_body',overlap),('unshared_body',~overlap)]:
        z=d[mask]
        performance[name]={'rows':len(z),'errors':int((z.truth!=z.C_TabM_epoch25).sum()),
            'by_class':{str(c):{'rows':int((z.truth==c).sum()),'errors':int(((z.truth==c)&(z.C_TabM_epoch25!=c)).sum())} for c in (1,2)}}
    # Close the original full-format roots, including all existing R0 and N1
    # equivalence, over identical ASA body-source symbols. No label in union/hash.
    f=pd.read_parquet(FOLDS,columns=['row_position','root','proposed_fold'])
    roots=np.sort(f.root.unique())
    pairs=d[['body','root']].drop_duplicates()
    lead=pairs.groupby('body').root.transform('min').to_numpy()
    left=np.searchsorted(roots,pairs.root); right=np.searchsorted(roots,lead)
    graph=sparse.coo_matrix((np.ones(len(left),dtype=np.int8),(left,right)),shape=(len(roots),len(roots))).tocsr()
    n,clusters=connected_components(graph,directed=False)
    minimum=pd.Series(roots).groupby(clusters).min()
    newroot=minimum.loc[clusters[np.searchsorted(roots,f.root.to_numpy())]].to_numpy()
    unique_roots=np.unique(newroot)
    foldmap={int(root):int(hashlib.sha256(f'10203:{int(root)}'.encode()).hexdigest()[:8],16)%3 for root in unique_roots}
    newfold=np.array([foldmap[int(i)] for i in newroot],dtype=np.int8)
    r=pd.read_parquet(OUT/'rows.parquet',columns=['route','label_index'])
    asa=r.route.eq('asa').to_numpy(); y=r.label_index.to_numpy()
    proposal=pd.DataFrame({'row_position':f.row_position,'root':newroot,'proposed_fold':newfold})
    af=d.assign(newfold=newfold[d.row_position],newroot=newroot[d.row_position])
    assert af.groupby('body').newfold.nunique().max()==1
    assert proposal.groupby('root').proposed_fold.nunique().max()==1
    counts={}
    for k in range(3):
        counts[str(k)]={'all_format_class_rows':np.bincount(y[newfold==k],minlength=3).tolist(),
            'ASA_class_rows':np.bincount(y[asa&(newfold==k)],minlength=3).tolist(),
            'ASA_class_roots':{str(c):int(np.unique(newroot[asa&(newfold==k)&(y==c)]).size) for c in (1,2)}}
    proposals=DEST/'proposed_body_closed_folds.parquet'
    proposal.to_parquet(proposals,index=False)
    result={'status':'no_fit_observed_body_source_overlap_and_split_preflight',
        'source_sha256':sha(__file__),'inputs_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in
            (FOLDS,rawpath,OUT/'rows.parquet',DEST/'paired_OOF_predictions.parquet')},
        'body_sources':len(a),'body_sources_spanning_old_roots':int((a.roots>1).sum()),
        'body_sources_spanning_old_folds':int((a.folds>1).sum()),'mixed_label_body_sources':int((a.labels>1).sum()),
        'performance_by_overlap':performance,'proposal':{'old_global_roots':len(roots),'new_global_roots':n,
            'rows_changed_fold':int((newfold!=f.proposed_fold.to_numpy()).sum()),
            'ASA_rows_changed_fold':int(((newfold!=f.proposed_fold.to_numpy())&asa).sum()),
            'remaining_crossfold_body_sources':0,'class_counts':counts,
            'path':proposals.relative_to(ROOT).as_posix(),'sha256':sha(proposals),
            'chosen_using_model_scores':False,'hash_rule':'unchanged 10203:min_root sha256 modulo 3'},
        'limitations':['This conservatively isolates observed body symbols, not certified real actors.',
            'Overlap does not prove direct label leakage: body IDs are masked from model inputs.',
            'New split is development, not a fresh blind or external test.',
            'Old saved predictions cannot be used as a matched baseline under this new split.'],
        'fits':0}
    save(target,result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
