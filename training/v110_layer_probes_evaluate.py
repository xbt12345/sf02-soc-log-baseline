"""Independent V110 development-fold evaluation and frozen full-task replay."""
import json
import re
import time

import numpy as np
import pandas as pd

from run_v75 import ROOT, save, sha
from v110_layer_probes import DEST, LEDGER, PREV, FID, ROWS, PLAN, check_registration, read_json
from v107_evaluate import report_metric


def class_root_stats(d,pred,c):
    take=d.truth.to_numpy()==c
    a=pd.DataFrame({'root':d.root.to_numpy()[take],'good':(pred[take]==c).astype(np.int8)})
    g=a.groupby('root',sort=False).good.agg(['size','sum'])
    recall=g['sum']/g['size']
    return {'groups':int(len(g)),'mean_recall':float(recall.mean()),
      'zero_recall_groups':int((g['sum']==0).sum()),
      'at_least_10_row_group_mean_recall':float(recall[g['size']>=10].mean()) if (g['size']>=10).any() else None}


def paired(d,old,new):
    y=d.truth.to_numpy();root=d.root.to_numpy();fold=d.fold.to_numpy()
    records=[]
    for a in np.unique(root):
        take=root==a
        row={'root':int(a),'fold':int(fold[take][0]),'rows':int(take.sum())}
        for c,label in [(1,'M'),(2,'S')]:
            cc=take&(y==c);oc=old[cc]==c;nc=new[cc]==c
            row.update({label+'_rows':int(cc.sum()),label+'_old_correct':int(oc.sum()),label+'_new_correct':int(nc.sum()),
              label+'_repaired':int((~oc&nc).sum()),label+'_regressed':int((oc&~nc).sum())})
        records.append(row)
    g=pd.DataFrame(records).sort_values('root').reset_index(drop=True)
    return g


def bootstrap(d,old,new,reps=500):
    """Paired source-root bootstrap with fixed fold strata; development uncertainty only."""
    ids,inverse=np.unique(d.root.to_numpy(),return_inverse=True)
    folds=np.zeros(len(ids),np.int8)
    np.maximum.at(folds,inverse,d.fold.to_numpy(dtype=np.int8))
    droot=pd.DataFrame({'root':d.root,'fold':d.fold}).groupby('root').fold.nunique()
    assert droot.max()==1
    y=d.truth.to_numpy()
    def cm(pred):
        assert set(np.unique(pred))<={1,2}
        return np.bincount(inverse*4+(y-1)*2+(pred-1),minlength=len(ids)*4).reshape(-1,4)
    ref=cm(old);candidate=cm(new)
    rng=np.random.default_rng(110)
    diff=[]
    for _ in range(reps):
        inds=np.concatenate([rng.choice(np.flatnonzero(folds==k),size=int((folds==k).sum()),replace=True)
          for k in (0,1,2)])
        b=ref[inds].sum(0);n=candidate[inds].sum(0)
        def ms_f1(a):
            # Binary M/S confusion flattened as MM, MS, SM, SS.
            mm,ms,sm,ss=map(float,a)
            return ((2*mm/(2*mm+ms+sm) if 2*mm+ms+sm else 0)+
                    (2*ss/(2*ss+ms+sm) if 2*ss+ms+sm else 0))/2
        diff.append(ms_f1(n)-ms_f1(b))
    return {'unit':'proposed_source_root_within_fixed_fold','roots':int(len(ids)),
      'repetitions':reps,'delta_MS_F1_CI_95':np.quantile(diff,[.025,.975]).tolist(),
      'scope':'Heavily reused development roots; not external transfer confidence interval.'}


def full_predictions(d,asa_pred):
    rows=pd.read_parquet(ROWS,columns=['route','label_index'])
    y=rows.label_index.to_numpy(dtype=np.int8)
    asa=rows.route.eq('asa').to_numpy();pos=np.flatnonzero(asa)
    assert len(pos)==len(d)==112807 and np.array_equal(pos,d.row_position.to_numpy())
    fid=np.load(FID,mmap_mode='r')
    assert len(fid)==len(y)==2056871
    folds=np.empty(len(fid),np.int8)
    folds[pos]=d.fold.to_numpy(dtype=np.int8)
    # Non-ASA fold assignments come from the same registered V106 split.
    f=pd.read_parquet(ROOT/'artifacts/v106_frozen_audit_20260928/proposed_body_closed_folds.parquet',columns=['proposed_fold'])
    folds=f.proposed_fold.to_numpy(dtype=np.int8)
    assert np.array_equal(folds[pos],d.fold.to_numpy())
    base=np.empty(len(fid),np.int8)
    for k in (0,1,2):
        scores=np.load(PREV/f'fold{k}_N1_teacher'/'scores_all_input_ids.npy',mmap_mode='r')
        take=folds==k
        base[take]=scores[fid[take]].argmax(1).astype(np.int8)
    ref=base.copy();gate=base[pos]!=0
    old=d.N1_TabM25.to_numpy(dtype=np.int8)
    ref[pos[gate]]=old[gate]
    existing=read_json(PREV/'evaluation.json')['full_task']['N1_TabM25_composite']
    assert report_metric(y,ref)['confusion_matrix_true_rows_pred_columns']==existing['confusion_matrix_true_rows_pred_columns']
    new=base.copy();new[pos[gate]]=asa_pred[gate]
    assert np.array_equal(new[~asa],ref[~asa])
    return report_metric(y,ref),report_metric(y,new),int((ref[asa]!=new[asa]).sum()),int(gate.sum())


def evaluate():
    assert not (DEST/'evaluation.json').exists()
    reg=check_registration();contract=read_json(PLAN)
    d=pd.read_parquet(LEDGER)
    assert len(d)==112807
    y=d.truth.to_numpy(dtype=np.int8);fold=d.fold.to_numpy(dtype=np.int8);local=d.local.to_numpy(dtype=np.int32)
    old={'N1':d.N1_TabM25.to_numpy(dtype=np.int8),'N2':d.N2_TabM25.to_numpy(dtype=np.int8)}
    top3=d.loc[d.truth==2,'root'].value_counts().head(3).index.to_numpy()
    assert int(((d.truth==2)&d.root.isin(top3)).sum())==31057
    trial=d[['row_position','root','fold','truth','local','changed','behavior','coarse','behavior_same_roots','behavior_other_roots']].copy()
    trial['known_top3_S_root']=trial.root.isin(top3)
    trial['destination_port_unavailable']=trial.behavior.str.contains('"dst_port_fixed": 65536',regex=False,na=False)
    for key,p in old.items():trial[key+'_old_prediction']=p
    result={'status':'six_frozen_probe_fits_evaluated_developmental','actual_classifier_fits':6,'calibration_fits':0,
      'source_sha256':sha(__file__),'registration_sha256':sha(DEST/'registration.json'),
      'contract_sha256':sha(PLAN),'input_sha256':reg['input_sha256'],
      'no_private_answer_used':True,'all_existing_folds_previously_inspected':True,
      'class_population_ASA':[0,int((y==1).sum()),int((y==2).sum())],
      'top3_S_roots':list(map(int,top3)),'variants':{},'folds':{},'paired':{},
      'full_task':{},'bootstrap':{},'gates':{},
      'limitations':['Feature extractors were trained on probe-training rows, so probe fit is not independent DFR data.',
        'Root is a conservative source grouping proxy; historical study of all folds prevents blind-test claims.',
        'Two-layer linear probes test this restricted readout, not all information decodability or external transfer.']}
    oldN1=report_metric(y,old['N1']);oldN2=report_metric(y,old['N2'])
    assert oldN1['class']['malicious']['missed']==318 and oldN1['class']['suspicious']['missed']==2094
    assert oldN2['class']['malicious']['missed']==296 and oldN2['class']['suspicious']['missed']==5194
    result['ASA_references']={'N1':oldN1,'N2':oldN2}
    for arm in ('P1','P2'):
        probs=np.empty(len(d),np.float32);fitreceipts={}
        for k in (0,1,2):
            folder=DEST/f'fold{k}_{arm}';fit=read_json(folder/'fit.json')
            assert fit['converged_to_registered_gradient_gate'] and fit['classifier_fits']==1
            assert fit['registration_sha256']==sha(DEST/'registration.json')
            assert fit['source_sha256']==reg['source_sha256']
            assert sha(folder/'probe.npz')==fit['probe_sha256']
            assert sha(folder/'ASA_input_S_probability.npy')==fit['prob_sha256']
            p=np.load(folder/'ASA_input_S_probability.npy',mmap_mode='r')
            take=fold==k
            probs[take]=p[local[take]]
            fitreceipts[str(k)]={'fit_sha256':sha(folder/'fit.json'),'probe_sha256':fit['probe_sha256'],
                'input_probability_sha256':fit['prob_sha256'],'converged':True,
                'gradient_inf':fit['gradient_inf'],'train_original_rows':fit['train_original_rows']}
        pred=np.where(probs>=.5,2,1).astype(np.int8)
        trial[arm+'_S_probability']=probs;trial[arm+'_prediction']=pred
        stats=report_metric(y,pred)
        for k in (0,1,2):
            take=fold==k
            result['folds'].setdefault(str(k),{})[arm]=report_metric(y[take],pred[take])
        result['variants'][arm]={'ASA':stats,'root_M':class_root_stats(d,pred,1),
          'root_S':class_root_stats(d,pred,2),'fit_receipts':fitreceipts,
          'outside_top3_S_roots':report_metric(y[~d.root.isin(top3)],pred[~d.root.isin(top3)]),
          'changed_wrapper':report_metric(y[d.changed],pred[d.changed]) if d.changed.any() else None,
          'unchanged_wrapper':report_metric(y[~d.changed],pred[~d.changed]),
          'destination_port_unavailable':report_metric(y[trial.destination_port_unavailable],pred[trial.destination_port_unavailable]) if trial.destination_port_unavailable.any() else None}
        for refname,refpred in old.items():
            g=paired(d,refpred,pred)
            g.to_parquet(DEST/f'{arm}_vs_{refname}_source_group_delta.parquet',index=False)
            delta={}
            for c,name in [(1,'M'),(2,'S')]:
                take=y==c;oc=refpred[take]==c;nc=pred[take]==c
                delta[name]={'repaired':int((~oc&nc).sum()),'regressed':int((oc&~nc).sum()),
                  'net':int(nc.sum()-oc.sum()),'improved_roots':int((g[name+'_new_correct']>g[name+'_old_correct']).sum()),
                  'regressed_roots':int((g[name+'_new_correct']<g[name+'_old_correct']).sum())}
            result['paired'][arm+'_vs_'+refname]=delta
            result['bootstrap'][arm+'_vs_'+refname]=bootstrap(d,refpred,pred)
        ref_full,new_full,changed,gate_rows=full_predictions(d,pred)
        result['full_task'][arm]=new_full
        result['full_task'].setdefault('N1_reference',ref_full)
        result['variants'][arm]['full_composite_changed_ASA_rows']=changed
        result['variants'][arm]['teacher_nonbenign_ASA_gate_rows']=gate_rows
        refgroup={c:class_root_stats(d,old['N2'],c) for c in (1,2)}
        newgroup={c:class_root_stats(d,pred,c) for c in (1,2)}
        outside=(y==2)&(~d.root.isin(top3))
        s_gain_outside=int((pred[outside]==2).sum()-(old['N2'][outside]==2).sum())
        g=paired(d,old['N2'],pred)
        goodfolds=0
        for k in (0,1,2):
            gg=g.loc[g.fold==k]
            if int((gg.S_new_correct>gg.S_old_correct).sum())>=2:goodfolds+=1
        a_m,a_s=stats['class']['malicious']['missed'],stats['class']['suspicious']['missed']
        rowgate=bool(a_m<=296 and a_s<5194)
        fullgate=bool(a_m<=318 and a_s<=2094 and (a_m<318 or a_s<2094)
           and all(new_full['class'][c]['correct']>=ref_full['class'][c]['correct']
                   and new_full['class'][c]['f1']>=ref_full['class'][c]['f1'] for c in ('malicious','suspicious'))
           and new_full['class']['benign']['missed']<=ref_full['class']['benign']['missed'])
        gategroup=bool(all(newgroup[c]['mean_recall']>=refgroup[c]['mean_recall'] for c in (1,2)))
        result['gates'][arm]={'N2_row_level_research_entry':rowgate,
          'N1_full_task_replacement_count_entry':fullgate,
          'net_S_repairs_outside_top3_S_roots':s_gain_outside,
          'multiple_S_improving_roots_folds':goodfolds,
          'source_macro_M_S_no_decrease_vs_N2':gategroup,
          'research_continuation_preliminary':bool((rowgate or fullgate) and s_gain_outside>0 and goodfolds>=2 and gategroup),
          'formal_model_replacement':False,
          'formal_replacement_reason':'Requires frozen-wrapper audit and an independent selection/acceptance decision; development evidence alone cannot prove external transfer.'}
        print(json.dumps({'stage':'arm_evaluated','arm':arm,'ASA_M_errors':a_m,'ASA_S_errors':a_s,
          'MS_F1':stats['MS_equal_F1'],'outside_top3_S_gain':s_gain_outside,
          'full_macro_F1':new_full['macro_F1'],'gates':result['gates'][arm]},ensure_ascii=False),flush=True)
    trial.to_parquet(DEST/'OOF_probe_comparison.parquet',index=False)
    result['OOF_ledger_sha256']=sha(DEST/'OOF_probe_comparison.parquet')
    invariant=read_json(ROOT/'artifacts/v106_frozen_audit_20260928/verification_and_N2_preflight.json')
    assert invariant['all_checks_passed'] and invariant['checks']['N2_collapse_expand_invariance_all_unique_raw']
    result['frozen_wrapper_invariance']={'prior_verified_same_N2_input':True,
      'both_probes_deterministic_on_same_N2_input':True,
      'quality_scope':'Input identity implies identical probe prediction; it does not prove correct labeling.'}
    result['model_promoted']=False
    save(DEST/'evaluation.json',result)
    print(json.dumps({'stage':'evaluation_complete','model_promoted':False,'OOF_ledger_sha256':result['OOF_ledger_sha256']}),flush=True)


if __name__=='__main__':evaluate()
