"""Independent V127 endpoint replay and complete original-row classification audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from experiment_review import class_counts, check_bindings
from v127_experiment_review import ROOT, require_run_seal, require_checkpoint
from v127_model import make_branch, branch_batch
from v127_train import OUT, PARENT, PLAN, TRACE, MANIFEST, OFFICIAL, row_counts, class_diagnostic
from v125_evaluate import ROWS, FOLDS, LADDER, teacher_predictions, root_stats
from v126_frozen_audit import sha, read, save

TOP3_S_ROOTS = (21702, 20849, 29)
BUCKETS = ('known_parameter_two_roots_per_class', 'unknown_parameter_pooled_support',
           'same_class_support_but_insufficient_dual_root_coverage',
           'no_same_class_at_destination_resolution')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_network(fold, arm, plan):
    folder=OUT/f'fold{fold}_{arm}'
    receipt=read(folder/'fit.json')
    require_checkpoint(plan,receipt)
    require(receipt['fold']==fold and receipt['arm']==arm and receipt['seed']==12701
            and receipt['seal_sha256']==sha(OUT/'run_seal.json'), 'Wrong model identity')
    for name, expected in [('epoch50_model.pt','model_sha256'),('epoch50_prob.npy','prob_sha256'),
                           ('epoch50_residual.npy','residual_sha256'),('steps.jsonl','steps_sha256'),
                           ('progress.json','progress_sha256'),('checkpoints.json','checkpoints_sha256')]:
        require(sha(folder/name)==receipt[expected], 'Fit file changed '+str(folder/name))
    check=read(folder/'checkpoints.json')
    require([v['epoch'] for v in check]==plan['training']['checkpoints'], 'Checkpoint trajectory incomplete')
    for cp in check:
        for name,key in [(f"epoch{cp['epoch']}_model.pt",'model_sha256'),
                         (f"epoch{cp['epoch']}_prob.npy",'prob_sha256'),
                         (f"epoch{cp['epoch']}_residual.npy",'residual_sha256')]:
            require(sha(folder/name)==cp[key], 'Checkpoint changed '+str(folder/name))
    model=make_branch(arm,12701,'cuda').eval()
    state=torch.load(folder/'epoch50_model.pt',map_location='cpu',weights_only=True)
    require((state['fold'],state['arm'],state['seed'],state['epoch'],state['optimizer_steps'])==
            (fold,arm,12701,50,plan['training']['expected_steps_per_arm'][fold]),'Endpoint state wrong')
    model.load_state_dict(state['branch'])
    return receipt,check,model


def replay_residual(model,body,lengths):
    out=np.empty((len(lengths),3),np.float32)
    with torch.no_grad():
        for start in range(0,len(lengths),256):
            ids=np.arange(start,min(start+256,len(lengths)))
            out[ids]=branch_batch(model,body,lengths,ids,'cuda').cpu().numpy()
    return out


def constant_stationarity(fold,fit,base,counts,plan):
    b=np.asarray(fit['bias'],dtype=np.float64)
    mass=counts.sum(1).astype(np.float64);n=mass.sum()
    v=base.astype(np.float64)+b
    logp=v-np.logaddexp.reduce(v,axis=-1)[:,:,None]
    value=float(-(counts*logp.mean(1)).sum()/n+plan['constant_control']['l2_coefficient']/2*np.dot(b,b))
    q=softmax(v,axis=-1).mean(1)
    gb=(q*mass[:,None]-counts).sum(0)/n+plan['constant_control']['l2_coefficient']*b
    grad=np.array([gb[0]-gb[2],gb[1]-gb[2]])
    require(fit['status']=='fit_executed' and np.max(np.abs(grad))<=1e-7
            and np.isclose(value,fit['objective'],rtol=0,atol=1e-10)
            and np.allclose(b.sum(),0,atol=1e-10), 'K not independently stationary')
    return {'objective':value,'gradient_inf':float(np.max(np.abs(grad))),'iterations':fit['nit']}


def main():
    plan=require_run_seal(OUT/'run_seal.json',ROOT/'training/v127_train.py')
    # V125 sealed the unchanged complete-row map, source folds and frozen
    # non-ASA teacher files before this run; require their original bindings.
    check_bindings(read(PARENT/'run_seal.json')['source_sha256'])
    target=OUT/'primary_evaluation.json'
    if target.exists():raise FileExistsError(target)
    reg=read(OUT/'registration.json')
    require(reg['fit_order']==plan['training']['fit_order'], 'Fit order changed')
    trace=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth'])
    ladder=pd.read_parquet(LADDER,columns=['row_position','diagnostic_bucket','A_all_seven_wrong','destination_key'])
    require(len(trace)==112807 and trace.row_position.equals(ladder.row_position), 'ASA ladder mismatch')
    manifest=pd.read_parquet(MANIFEST)
    body=np.load(PARENT/'ordered_body_bytes.npy',mmap_mode='r')
    lengths=np.load(PARENT/'body_lengths.npy')
    original_local=trace.local.to_numpy();folds=trace.fold.to_numpy();y=trace.truth.to_numpy()
    predictions={a:np.empty(len(trace),dtype=np.int8) for a in ('A0','AH','K','W','P')}
    probabilities={a:np.empty(len(trace),dtype=np.float32) for a in predictions}
    fit_receipts=[];endpoint_replays=[];permutations=[];fit_diagnostic={};K_diagnostic={}
    fit_nonconflict={arm:{'1':0,'2':0} for arm in predictions}
    endpoint_extra={}
    for fold in range(3):
        fit,held,fit_counts,held_counts=row_counts(manifest,fold,len(lengths))
        base=np.load(OUT/f'fold{fold}_base_member_logits.npy')
        heldmask=folds==fold;loc=original_local[heldmask]
        a=np.load(PARENT/f'fold{fold}_A/epoch25_prob.npy')
        h=np.load(ROOT/'artifacts/v127_plan_review_20260929'/f'fold{fold}_A_header_probability.npy')
        require(np.array_equal(softmax(base,axis=-1).mean(1).argmax(1),h.argmax(1))
                and np.max(np.abs(softmax(base,axis=-1).mean(1)-h))<2e-6,'AH replay differs')
        for name,p in [('A0',a),('AH',h)]:
            predictions[name][heldmask]=p[loc].argmax(1)
            probabilities[name][heldmask]=p[loc,2]
        for arm in ('A0','AH'):
            if arm=='AH':
                _,diag=class_diagnostic(base,np.zeros((len(base),3),np.float32),fit_counts,held_counts)
                for c in (1,2):fit_nonconflict[arm][str(c)]+=diag['fit'][str(c)]['nonconflict_errors']
            else:
                mixed=(fit_counts[:,1]>0)&(fit_counts[:,2]>0)
                label=a.argmax(1)
                for c in (1,2):fit_nonconflict[arm][str(c)]+=int((fit_counts[:,c]*(label!=c)*(~mixed)).sum())
        kfolder=OUT/f'fold{fold}_K';k=read(kfolder/'fit.json')
        require(k['fold']==fold and k['arm']=='K' and k['seal_sha256']==sha(OUT/'run_seal.json')
                and sha(kfolder/'endpoint_prob.npy')==k['prob_sha256'],'K receipt wrong')
        K_diagnostic[str(fold)]=constant_stationarity(fold,k,base,fit_counts,plan)
        kp=softmax(base.astype(np.float64)+np.asarray(k['bias']),axis=-1).mean(1).astype(np.float32)
        require(np.max(np.abs(kp-np.load(kfolder/'endpoint_prob.npy')))<2e-6,'K probability replay failed')
        predictions['K'][heldmask]=kp[loc].argmax(1);probabilities['K'][heldmask]=kp[loc,2]
        mixed=(fit_counts[:,1]>0)&(fit_counts[:,2]>0)
        for c in (1,2):
            fit_nonconflict['K'][str(c)]+=int((fit_counts[:,c]*(kp.argmax(1)!=c)*(~mixed)).sum())
        fit_receipts.append({'fold':fold,'arm':'K','fit_sha256':sha(kfolder/'fit.json'),'iterations':k['nit']})
        for arm in ('W','P'):
            receipt,checks,model=load_network(fold,arm,plan)
            extra=replay_residual(model,body,lengths)
            p=softmax(base+extra[:,None,:],axis=-1).mean(1)
            saved=np.load(OUT/f'fold{fold}_{arm}/epoch50_prob.npy')
            residual_saved=np.load(OUT/f'fold{fold}_{arm}/epoch50_residual.npy')
            dp=float(np.max(np.abs(p-saved)));de=float(np.max(np.abs(extra-residual_saved)))
            require(dp<=2e-6 and de<=2e-5 and np.array_equal(p.argmax(1),saved.argmax(1)),
                    'Endpoint replay failed '+str((fold,arm,dp,de)))
            predictions[arm][heldmask]=p[loc].argmax(1)
            probabilities[arm][heldmask]=p[loc,2]
            fit_receipts.append({'fold':fold,'arm':arm,'fit_sha256':sha(OUT/f'fold{fold}_{arm}/fit.json'),
                                 'optimizer_steps':receipt['optimizer_steps'],'seconds':receipt['seconds']})
            endpoint_replays.append({'fold':fold,'arm':arm,'max_abs_probability_difference':dp,
                                     'max_abs_residual_difference':de})
            fit_diagnostic[f'{fold}_{arm}']=checks[-1]
            for c in (1,2):
                fit_nonconflict[arm][str(c)]+=checks[-1]['fit_and_held']['fit'][str(c)]['nonconflict_errors']
            if arm=='P':
                endpoint_extra[fold]=extra
            del model
        # The diagnostic permutation is original-row based, separately within fit and held.
        extra=endpoint_extra[fold]
        for role_id,mask in ((0,folds!=fold),(1,folds==fold)):
            positions=np.flatnonzero(mask)
            original=extra[original_local[positions]]
            shuffled=original[np.random.default_rng(12711+100*fold+role_id).permutation(len(positions))]
            require(np.array_equal(np.sort(original,axis=0),np.sort(shuffled,axis=0)),
                    'Residual marginal changed')
            baseline=base[original_local[positions]]
            real=softmax(baseline+original[:,None,:],axis=-1).mean(1).argmax(1)
            random=softmax(baseline+shuffled[:,None,:],axis=-1).mean(1).argmax(1)
            role_y=y[positions]
            permutations.append({'fold':fold,'role':'fit' if role_id==0 else 'held',
                                 'rows':len(positions),'seed':12711+100*fold+role_id,
                                 'real_M_errors':int(((role_y==1)&(real!=1)).sum()),
                                 'real_S_errors':int(((role_y==2)&(real!=2)).sum()),
                                 'permuted_M_errors':int(((role_y==1)&(random!=1)).sum()),
                                 'permuted_S_errors':int(((role_y==2)&(random!=2)).sum()),
                                 'prediction_flips':int((real!=random).sum())})
    require(len(fit_receipts)==9 and sum(r.get('optimizer_steps',0) for r in fit_receipts)==17800,
            'Incomplete registered nine-fit run')
    require(np.array_equal(predictions['A0'],pd.read_parquet(PARENT/'expert_ASA_predictions.parquet').expert_pred_A.to_numpy()),
            'Hard baseline changed')
    ledger=trace.copy()
    for arm,p in predictions.items():
        ledger['pred_'+arm]=p;ledger['S_probability_'+arm]=probabilities[arm]
    ledger.to_parquet(OUT/'expert_ASA_predictions.parquet',index=False)

    official=pd.read_parquet(OFFICIAL,columns=['event_id','label_binary'])
    rows=pd.read_parquet(ROWS,columns=['row_position','event_id','label_index','route'])
    folds_all=pd.read_parquet(FOLDS,columns=['row_position','root','proposed_fold'])
    n=len(rows)
    require(n==len(official)==len(folds_all)==2056871 and np.array_equal(rows.row_position,np.arange(n))
            and np.array_equal(rows.event_id,official.event_id),'Full population or event identity changed')
    truth=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(dtype=np.int8)
    require(np.array_equal(truth,rows.label_index.to_numpy(np.int8)), 'Independent official truth mismatch')
    asa=np.flatnonzero(rows.route.eq('asa'))
    require(np.array_equal(asa,trace.row_position) and np.array_equal(truth[asa],y)
            and np.array_equal(folds_all.proposed_fold.to_numpy()[asa],folds)
            and np.array_equal(folds_all.root.to_numpy()[asa],trace.root),'ASA identity changed')
    teacher=teacher_predictions(n,folds_all)
    other=~rows.route.eq('asa').to_numpy()
    require(int((teacher[other]!=truth[other]).sum())==107,'Frozen non-ASA route changed')
    full={arm:teacher.copy() for arm in predictions}
    for arm in predictions:full[arm][asa]=predictions[arm]
    fullledger=pd.DataFrame({'row_position':rows.row_position,'teacher_prediction':teacher,
                             **{'final_pred_'+arm:full[arm] for arm in predictions}})
    fullledger.to_parquet(OUT/'full_prediction_ledger.parquet',index=False)
    metrics={'ASA':{},'full_task':{},'source_groups':{}}
    for arm in predictions:
        metrics['ASA'][arm]=class_counts(trace.assign(pred=predictions[arm]),'pred')
        metrics['full_task'][arm]=class_counts(pd.DataFrame({'truth':truth,'pred':full[arm]}),'pred')
        metrics['source_groups'][arm]=root_stats(trace,predictions[arm])
    transitions={}
    for arm in ('AH','K','W','P'):
        transitions[arm]={}
        for c in (1,2):
            m=y==c;a=predictions['A0'];b=predictions[arm]
            transitions[arm][str(c)]={'repaired':int((m&(a!=c)&(b==c)).sum()),
                                       'regressed':int((m&(a==c)&(b!=c)).sum())}
    source=trace[['root','truth','fold']].copy()
    for arm,p in predictions.items():source[arm+'_wrong']=p!=y
    source=source.groupby(['root','truth','fold'],as_index=False).agg(
        rows=('A0_wrong','size'),**{arm+'_errors':(arm+'_wrong','sum') for arm in predictions})
    source.to_csv(OUT/'source_group_changes.csv',index=False)
    slices={
        'header_682':trace.row_position.isin(pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet',columns=['row_position']).row_position).to_numpy(),
        'prior_40_projection_regressions':(y==2)&(predictions['A0']==2)&(predictions['AH']!=2),
        'historical_persistent_S':ladder.A_all_seven_wrong.to_numpy()&(y==2),
        'S_outside_largest3_roots':(y==2)&~np.isin(trace.root.to_numpy(),TOP3_S_ROOTS),
        'remaining_ASA':np.ones(len(y),dtype=bool)}
    for bucket in BUCKETS:slices[bucket]=ladder.diagnostic_bucket.eq(bucket).to_numpy()
    for fold in range(3):
        for c in (1,2):slices[f'fold{fold}_class{c}']=(folds==fold)&(y==c)
    slice_report={name:{'rows':int(mask.sum()),**{arm+'_errors':int(((predictions[arm]!=y)&mask).sum())
                                               for arm in predictions}} for name,mask in slices.items()}
    cards=read(ROOT/'artifacts/v124_header_trial_20260929/capability_qualification.json')['cards']
    priority={}
    for arm in predictions:
        roles=[]
        for fold in range(3):
            if arm=='A0':p=np.load(PARENT/f'fold{fold}_A/epoch25_prob.npy')
            elif arm=='AH':p=np.load(ROOT/'artifacts/v127_plan_review_20260929'/f'fold{fold}_A_header_probability.npy')
            elif arm=='K':p=np.load(OUT/f'fold{fold}_K/endpoint_prob.npy')
            else:p=np.load(OUT/f'fold{fold}_{arm}/epoch50_prob.npy')
            for card in cards:
                if card['fold']!=fold:continue
                selected=(trace.fold.ne(fold)&ladder.destination_key.eq(card['destination_key'])&trace.truth.eq(2))
                for row in trace.loc[selected,['row_position','local','root']].itertuples(index=False):
                    roles.append((int(row.row_position),int(row.root),int(p[int(row.local)].argmax())!=2))
        q=pd.DataFrame(roles,columns=['row_position','root','wrong'])
        require(len(q)==206 and q.row_position.nunique()==136,'Priority fit population changed')
        priority[arm]={'S_fit_roles':len(q),'S_unique_original_rows':q.row_position.nunique(),
                       'S_wrong_roles':int(q.wrong.sum()),
                       'S_wrong_unique_original_rows':q.loc[q.wrong,'row_position'].nunique(),
                       'S_wrong_roots':q.loc[q.wrong,'root'].nunique()}
    fold_report=[]
    for fold in range(3):
        q=folds==fold
        fold_report.append({'fold':fold,**{arm+'_errors':int((predictions[arm][q]!=y[q]).sum())
                                              for arm in predictions}})
    a,p,k=metrics['ASA']['A0'],metrics['ASA']['P'],metrics['ASA']['K']
    fullA,fullP=metrics['full_task']['A0'],metrics['full_task']['P']
    outside=slices['S_outside_largest3_roots'];groups=metrics['source_groups']
    all_errors={arm:sum(metrics['ASA'][arm][str(c)]['missed'] for c in (1,2)) for arm in predictions}
    gates={
        'all_nine_fits_and_endpoints_replayed':True,
        'constant_control_stationary':len(K_diagnostic)==3,
        'ASA_M_errors_le_318':p['1']['missed']<=318,
        'ASA_S_errors_le_2074':p['2']['missed']<=2074,
        'ASA_total_errors_le_2170':all_errors['P']<=2170,
        'P_beats_converged_K_total_errors':all_errors['P']<all_errors['K'],
        'ASA_M_S_recall_precision_F1_no_regression':all(p[str(c)][metric]>=a[str(c)][metric]
             for c in (1,2) for metric in ('recall','precision','f1')),
        'full_task_each_class_recall_precision_F1_no_regression':all(fullP[str(c)][metric]>=fullA[str(c)][metric]
             for c in (0,1,2) for metric in ('recall','precision','f1')),
        'at_least_two_improving_folds':sum(v['P_errors']<v['A0_errors'] for v in fold_report)>=2,
        'outside_top3_S_roots_no_regression':int((predictions['P'][outside]!=y[outside]).sum())<=int((predictions['A0'][outside]!=y[outside]).sum()),
        'S_group_macro_no_regression':groups['P']['mean_recall']>=groups['A0']['mean_recall'],
        'S_zero_recall_groups_no_increase':groups['P']['zero_recall_roots']<=groups['A0']['zero_recall_roots'],
        'registered_header_variants_zero_flips':read(ROOT/'artifacts/v127_plan_review_20260929/header_projection.json')['full_ASA_variant_equivalence_rows']==112807,
        'original_rows_and_non_ASA_preserved':len(asa)==112807 and int((teacher[other]!=truth[other]).sum())==107,
    }
    quality_passed=all(gates.values())
    result={'status':'v127_nine_fit_source_closed_evaluated','latest_actual_training':'V127',
            'classifier_fits_new':9,'network_fits':6,'constant_fits':3,'network_optimizer_steps':17800,
            'constant_optimizer_iterations':sum(v['iterations'] for v in K_diagnostic.values()),
            'calibration_fits':0,'confirmation_fits':0,'full_rows':n,'ASA_rows':len(trace),
            'official_full_class_support':{str(c):int((truth==c).sum()) for c in (0,1,2)},
            'fit_receipts':fit_receipts,'endpoint_model_replay':endpoint_replays,
            'constant_control_stationarity':K_diagnostic,'metrics':metrics,'ASA_total_errors':all_errors,
            'paired_changes_vs_A0':transitions,'folds':fold_report,'slices':slice_report,
            'input_alignment_permutation':permutations,
            'fit_endpoint_diagnostic':fit_diagnostic,
            'fit_nonconflict_errors':fit_nonconflict,'priority_fit_S_136_unique_206_roles':priority,
            'mechanism_progress':{'P_fit_nonconflict_S_below_A0_and_K':fit_nonconflict['P']['2']<min(fit_nonconflict['A0']['2'],fit_nonconflict['K']['2']),
                                  'P_fit_nonconflict_M_no_regression':fit_nonconflict['P']['1']<=fit_nonconflict['A0']['1']},
            'quality_gates':{name:bool(value) for name,value in gates.items()},
            'primary_quality_passed':quality_passed,'confirmation_allowed':quality_passed,
            'quality_acceptance':False,'model_promoted':False,
            'limitations':['Inspected local source-closed folds, not independent blind data.',
                'Permutation is diagnostic and not a deployable classifier.',
                'Registered header invariance follows the verified shared canonical input contract, not arbitrary unseen formats.',
                'K certifies the registered regularized CE optimum numerically, not global minimum classification error.']}
    save(target,result)
    print(json.dumps({'stage':result['status'],'ASA_errors':{arm:{str(c):metrics['ASA'][arm][str(c)]['missed'] for c in (1,2)} for arm in predictions},
                      'gates':gates,'confirmation_allowed':quality_passed},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
