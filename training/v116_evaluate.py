"""Evaluate paired checkpoints only AFTER all inner selections are locked."""
import json

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from v116_preflight import ROOT, DEST, VIEW, save, sha
from v116_train import read, check, choose, metrics
from v104_phase_b import DEVICE, FID, ROWS, SparseTabM, predict_all
from v107_matched_training import DEST as OLD, FOLDS
from v107_evaluate import report_metric


def main():
    reg=check();lock=read(DEST/'locked_selection.json')
    assert not (DEST/'evaluation.json').exists()
    manifest=pd.read_parquet(DEST/'inner_split_manifest.parquet')
    d=manifest[manifest.outer_fold.eq(0)][['row_position','local','root','fold','truth','parameter']].reset_index(drop=True)
    a=np.empty(len(d),np.int8);b=np.empty_like(a)
    ap=np.empty((len(d),3),np.float32);bp=np.empty_like(ap)
    x=sparse.load_npz(VIEW);fits=[];replays=[]
    for fold in range(3):
        inner=DEST/f'inner_fold{fold}';outer=DEST/f'outer_fold{fold}'
        inner_fit=read(inner/'fit.json');outer_fit=read(outer/'fit.json')
        assert sha(inner/'checkpoints.json')==inner_fit['checkpoint_report_sha256']
        selected=choose(read(inner/'checkpoints.json'))['selected_epoch']
        assert selected==lock['folds'][str(fold)]['selected_epoch']==outer_fit['selected_epoch']
        assert read(outer/'started.json')['selection_sha256']==sha(DEST/'locked_selection.json')
        assert read(outer/'started.json')['created_unix']>lock['created_unix']
        assert sha(outer/'checkpoints.json')==outer_fit['checkpoint_report_sha256']
        outer_records={q['epoch']:q for q in read(outer/'checkpoints.json')}
        for epoch in sorted({selected,25}):
            entry=outer_records[epoch];p=outer/f'epoch{epoch}_prob.npy'
            mp=outer/f'epoch{epoch}_model.pt'
            assert sha(p)==entry['prob_sha256'] and sha(mp)==entry['model_sha256']
            net=SparseTabM().to(DEVICE)
            state=torch.load(mp,map_location='cpu',weights_only=True)
            assert state['fold']==fold and state['epoch']==epoch and state['scope']=='outer'
            net.load_state_dict(state['state_dict'])
            probability=predict_all(net,'TabM',x,DEVICE)
            stored=np.load(p)
            assert np.array_equal(probability.argmax(1),stored.argmax(1))
            assert np.allclose(probability,stored,atol=2e-6,rtol=2e-6)
            replays.append({'fold':fold,'epoch':epoch,'max_probability_diff':float(abs(probability-stored).max())})
            del net
        mask=d.fold.eq(fold).to_numpy();ids=d.loc[mask,'local'].to_numpy()
        ap[mask]=np.load(outer/'epoch25_prob.npy')[ids]
        bp[mask]=np.load(outer/f'epoch{selected}_prob.npy')[ids]
        fits.extend([inner_fit,outer_fit])
    a[:]=ap.argmax(1);b[:]=bp.argmax(1);y=d.truth.to_numpy()
    d['A_epoch25']=a;d['B_selected']=b;d['A_S_probability']=ap[:,2];d['B_S_probability']=bp[:,2]
    d['repaired']=(a!=y)&(b==y);d['regressed']=(a==y)&(b!=y)
    d.to_parquet(DEST/'OOF_ASA_decisions.parquet',index=False)
    byfold={str(k):{'A':metrics(d[d.fold.eq(k)],a[d.fold.eq(k)]),
                   'B':metrics(d[d.fold.eq(k)],b[d.fold.eq(k)])} for k in range(3)}
    byclass={}
    for c,label in [(1,'M'),(2,'S')]:
        q=y==c
        byclass[label]={'support':int(q.sum()),'A_correct':int((q&(a==c)).sum()),'B_correct':int((q&(b==c)).sum()),
                       'A_errors':int((q&(a!=c)).sum()),'B_errors':int((q&(b!=c)).sum()),
                       'repaired':int((q&(a!=c)&(b==c)).sum()),'regressed':int((q&(a==c)&(b!=c)).sum())}
    groups=d.assign(A_wrong=a!=y,B_wrong=b!=y).groupby(['root','truth']).agg(rows=('local','size'),
             A_errors=('A_wrong','sum'),B_errors=('B_wrong','sum'),repaired=('repaired','sum'),regressed=('regressed','sum')).reset_index()
    groups.to_csv(DEST/'source_group_changes.csv',index=False)
    top3=d[d.truth.eq(2)].groupby('root').size().nlargest(3).index
    outside=d.truth.eq(2)&~d.root.isin(top3)
    outside_stat={'support':int(outside.sum()),'A_correct':int((outside&(a==2)).sum()),'B_correct':int((outside&(b==2)).sum())}
    # Report familiar behavior/parameter strata; NEVER used in epoch selection.
    sd=pd.read_parquet(ROOT/'artifacts/v115_stability_support_20260929/support_decisions.parquet',
                       columns=['row_position','support_bin'])
    assert np.array_equal(sd.row_position,d.row_position)
    d['support_bin']=sd.support_bin
    strata=d.assign(A_wrong=a!=y,B_wrong=b!=y).groupby(['support_bin','truth']).agg(rows=('local','size'),
             A_errors=('A_wrong','sum'),B_errors=('B_wrong','sum'),repaired=('repaired','sum'),regressed=('regressed','sum')).reset_index()
    strata.to_csv(DEST/'support_strata_results.csv',index=False)
    prior=pd.read_parquet(OLD/'OOF_ASA_ledger.parquet',columns=['row_position','N1_TabM25'])
    assert np.array_equal(prior.row_position,d.row_position)
    identity={'fresh_A_vs_V107_decision_changes':int((a!=prior.N1_TabM25.to_numpy()).sum())}
    # Frozen full-task teacher and benign gate, no new fit or threshold changes.
    r=pd.read_parquet(ROWS,columns=['route','label_index'])
    f=pd.read_parquet(FOLDS,columns=['proposed_fold'])
    fid=np.load(FID,mmap_mode='r');all_y=r.label_index.to_numpy(dtype=np.int8)
    teacher=np.empty(len(r),np.int8)
    for k in range(3):
        p=OLD/f'fold{k}_N1_teacher'/'scores_all_input_ids.npy'
        receipt=read(p.parent/'fit.json');assert sha(p)==receipt['scores_sha256']
        score=np.load(p,mmap_mode='r');idx=np.flatnonzero(f.proposed_fold.eq(k))
        teacher[idx]=score[np.asarray(fid[idx],np.int64)].argmax(1)
    pos=np.flatnonzero(r.route.eq('asa'));assert np.array_equal(pos,d.row_position)
    gate=teacher[pos]!=0;fa=teacher.copy();fb=teacher.copy()
    fa[pos[gate]]=a[gate];fb[pos[gate]]=b[gate]
    assert np.array_equal(fa[~r.route.eq('asa')],fb[~r.route.eq('asa')])
    full={'A':report_metric(all_y,fa),'B':report_metric(all_y,fb)}
    asa={'A':metrics(d,a),'B':metrics(d,b),'per_fold':byfold,'class_comparison':byclass,'S_outside_top3':outside_stat}
    gates={'ASA_M_errors_at_most_318':byclass['M']['B_errors']<=318,
           'ASA_S_errors_at_most_2094':byclass['S']['B_errors']<=2094,
           'ASA_total_errors_at_most_2170':asa['B']['errors']<=2170,
           'ASA_M_S_not_worse_than_fresh_A':all(q['B_errors']<=q['A_errors'] for q in byclass.values()),
           'two_folds_improve':sum(v['B']['errors']<v['A']['errors'] for v in byfold.values())>=2,
           'S_outside_top3_not_worse':outside_stat['B_correct']>=outside_stat['A_correct'],
           'full_each_class_recall_and_F1_not_worse':all(full['B']['class'][c][m]>=full['A']['class'][c][m]
               for c in ('benign','malicious','suspicious') for m in ('recall','f1'))}
    result={'status':'six_trajectories_executed_and_paired_evaluated','classifier_fits_new':6,
        'calibration_fits':0,'baseline_fits_reused':0,'selected_epochs':{k:v['selected_epoch'] for k,v in lock['folds'].items()},
        'registration_sha256':sha(DEST/'training_registration.json'),'locked_selection_sha256':sha(DEST/'locked_selection.json'),
        'ASA':asa,'full_task':full,'historical_baseline_check':identity,'gates':gates,'all_gates_passed':all(gates.values()),
        'quality_acceptance':False,'model_promoted':False,'replays':replays,'fits':fits,
        'issues':reg['issues']+([] if all(gates.values()) else ['Candidate fails promotion gates; stop epoch-selection branch without further epoch/weight search.']),
        'scope':'Repeatedly inspected developmental OOF; no new independent test; all original classes remain in full-task evaluation.'}
    save(DEST/'evaluation.json',result)
    save(DEST/'verification.json',{'all_checks_passed':True,'checks':{'source_and_input_hashes':True,
        'selection_independently_recomputed':True,'selection_precedes_all_outer_fits':True,
        'saved_outer_model_replay':True,'unchanged_non_ASA_routes':True,'six_fits_recorded':len(fits)==6},
        'scope':'Identity, selection causality, replay and arithmetic; not model quality acceptance.',
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (DEST/'evaluation.json',DEST/'OOF_ASA_decisions.parquet',
                            DEST/'source_group_changes.csv',DEST/'support_strata_results.csv',DEST/'locked_selection.json')}})
    print(json.dumps({'stage':'evaluated','selected':result['selected_epochs'],'class_comparison':byclass,
                      'A_errors':asa['A']['errors'],'B_errors':asa['B']['errors'],'gates':gates,'historical_check':identity},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
