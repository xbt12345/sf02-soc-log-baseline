"""Independent V128 fixed-endpoint replay and complete official-row quality audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.special import softmax

from experiment_review import class_counts, check_bindings, sha
from v128_experiment_review import check_plan, require_run_seal, require_checkpoint, read, save, ROOT
from v128_train_r3 import OUT, BASE, PARENT, TRACE, OFFICIAL, MANIFEST, BODY, LENGTHS, CANON, role_frame, check_roles, score_role, ORDER
from v125_evaluate import ROWS, FOLDS, LADDER, teacher_predictions, root_stats
from v125_model import Expert
from v127_model import make_branch, branch_batch

ARMS=('A0','AH','K_IS','K_CF','P_IS','P_CF')
TOP3=(21702,20849,29)


def require(ok,message):
    if not ok:raise ValueError(message)


def replay_teacher(fold,j,canonical):
    folder=OUT/f'teacher{fold}_{j}'
    r=read(folder/'fit.json');require_checkpoint(check_plan(),r,'teacher')
    require(r['fold']==fold and r['excluded_inner']==j and
            r['seal_sha256']==sha(OUT/'run_seal.json'),'Teacher identity changed')
    for name,key in [('epoch25_model.pt','model_sha256'),('canonical_member_logits.npy','logits_sha256'),
                     ('steps.jsonl','steps_sha256'),('progress.json','progress_sha256')]:
        require(sha(folder/name)==r[key],'Teacher artifact changed '+name)
    state=torch.load(folder/'epoch25_model.pt',map_location='cpu',weights_only=True)
    require((state['fold'],state['excluded_inner'],state['epoch'],state['optimizer_steps'])==
            (fold,j,25,r['optimizer_steps']),'Teacher state endpoint changed')
    model=Expert('A',10201).to('cuda').eval();model.base.load_state_dict(state['base'])
    saved=np.load(folder/'canonical_member_logits.npy',mmap_mode='r')
    # Recompute a deterministic distributed panel, including distinct class/source roles.
    ids=np.unique(np.concatenate((np.arange(0,22546,211),np.array([0,1,22544,22545]))))
    replay=np.empty((len(ids),16,3),np.float32)
    with torch.no_grad():
        for start in range(0,len(ids),256):
            ix=ids[start:start+256];replay[start:start+len(ix)]=model(canonical[ix]).cpu().numpy()
    gap=float(np.max(np.abs(replay-saved[ids])))
    require(gap<=2e-5,'Teacher saved canonical score differs from model')
    return {'fold':fold,'excluded_inner':j,'endpoint_sha256':r['model_sha256'],
            'score_replay_panel_inputs':len(ids),'score_replay_max_abs':gap,
            'fit_vs_crossfit':r['role_diagnostic'],'steps':r['optimizer_steps']}


def replay_branch(fold,arm,outer,body,lengths):
    folder=OUT/f'fold{fold}_{arm}';r=read(folder/'fit.json')
    require_checkpoint(check_plan(),r,'branch')
    require(r['fold']==fold and r['arm']==arm and r['seal_sha256']==sha(OUT/'run_seal.json'),
            'Branch identity changed')
    for name,key in [('epoch50_model.pt','model_sha256'),('epoch50_prob.npy','prob_sha256'),
                     ('epoch50_residual.npy','residual_sha256'),('steps.jsonl','steps_sha256'),
                     ('progress.json','progress_sha256'),('checkpoints.json','checkpoints_sha256')]:
        require(sha(folder/name)==r[key],'Branch artifact changed '+name)
    checks=read(folder/'checkpoints.json')
    require([v['epoch'] for v in checks]==[0,1,2,5,10,15,20,25,35,50],
            'Checkpoint trajectory incomplete')
    for cp in checks:
        for name,key in [(f"epoch{cp['epoch']}_model.pt",'model_sha256'),
                         (f"epoch{cp['epoch']}_prob.npy",'prob_sha256'),
                         (f"epoch{cp['epoch']}_residual.npy",'residual_sha256')]:
            require(sha(folder/name)==cp[key],'Intermediate checkpoint changed')
    state=torch.load(folder/'epoch50_model.pt',map_location='cpu',weights_only=True)
    require((state['fold'],state['arm'],state['seed'],state['epoch'],state['optimizer_steps'])==
            (fold,arm,12701,50,r['optimizer_steps']),'Wrong branch state')
    model=make_branch('P',12701,'cuda').eval();model.load_state_dict(state['branch'])
    extra=np.empty((len(lengths),3),np.float32)
    with torch.no_grad():
        for start in range(0,len(lengths),256):
            ids=np.arange(start,min(start+256,len(lengths)))
            extra[ids]=branch_batch(model,body,lengths,ids).cpu().numpy()
    p=softmax(outer+extra[:,None,:],axis=-1).mean(1).astype(np.float32)
    saved=np.load(folder/'epoch50_prob.npy');saved_extra=np.load(folder/'epoch50_residual.npy')
    dp=float(np.max(np.abs(p-saved)));de=float(np.max(np.abs(extra-saved_extra)))
    require(dp<=2e-6 and de<=2e-5 and np.array_equal(p.argmax(1),saved.argmax(1)),
            'Branch endpoint replay differs')
    return p,extra,checks[-1],{'fold':fold,'arm':arm,'steps':r['optimizer_steps'],
        'prob_replay_max_abs':dp,'residual_replay_max_abs':de,'receipt_sha256':sha(folder/'fit.json')}


def main():
    target=OUT/'primary_evaluation.json'
    if target.exists():raise FileExistsError(target)
    plan=require_run_seal(OUT/'run_seal.json',ROOT/'training/v128_train_r3.py')
    check_bindings(read(PARENT/'run_seal.json')['source_sha256'])
    check_bindings(read(BASE/'run_seal.json')['source_sha256'])
    require(len(ORDER)==21 and all((OUT/(f'teacher{f}_{j}' if kind=='teacher' else f'fold{f}_{kind}')/'fit.json').exists()
                                   for f,j,kind in ORDER),'Not all registered fits complete')
    trace=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth'])
    y=trace.truth.to_numpy(np.int8);loc=trace.local.to_numpy(np.int64)
    fold_id=trace.fold.to_numpy(np.int8)
    official=pd.read_parquet(OFFICIAL,columns=['label_binary']).label_binary.map(
        {'benign':0,'malicious':1,'suspicious':2}).to_numpy(dtype=np.int8)
    require(len(official)==2056871 and len(trace)==112807 and
            np.array_equal(official[trace.row_position],y),'Independent official truth changed')
    canonical=sparse.load_npz(CANON)
    body=np.load(BODY,mmap_mode='r');lengths=np.load(LENGTHS)
    teacher_receipts=[replay_teacher(f,j,canonical) for f in range(3) for j in range(3)]
    pred={a:np.empty(len(y),np.int8) for a in ARMS}
    sprob={a:np.empty(len(y),np.float32) for a in ARMS}
    replays=[];fit_diagnostics={};mean_pred=np.empty(len(y),np.int8)
    for fold in range(3):
        roles=role_frame(fold);check_roles(roles,trace,fold)
        outer=np.load(BASE/f'fold{fold}_base_member_logits.npy')
        held=fold_id==fold;ids=loc[held]
        a=np.load(PARENT/f'fold{fold}_A/epoch25_prob.npy')
        h=np.load(ROOT/f'artifacts/v127_plan_review_20260929/fold{fold}_A_header_probability.npy')
        require(np.max(np.abs(softmax(outer,axis=-1).mean(1)-h))<=2e-6,
                'Frozen AH baseline replay changed')
        for arm,p in [('A0',a),('AH',h)]:
            pred[arm][held]=p[ids].argmax(1);sprob[arm][held]=p[ids,2]
        for role in ('IS','CF'):
            score,counts,provenance=score_role(fold,role)
            folder=OUT/f'fold{fold}_K_{role}';r=read(folder/'fit.json')
            require(r['status']=='fit_executed' and r['role_provenance']==provenance and
                    r['seal_sha256']==sha(OUT/'run_seal.json'),'Constant role/receipt wrong')
            b=np.asarray(r['bias'],dtype=np.float64)
            used=np.flatnonzero(counts.sum(1));z=score[used].astype(np.float64)
            w=counts[used].astype(np.float64);v=z+b
            lp=v-np.logaddexp.reduce(v,axis=-1)[:,:,None]
            n=w.sum();lam=plan['matched_primary']['constant_solver']['l2_coefficient']
            objective=float(-(w*lp.mean(1)).sum()/n+lam/2*np.dot(b,b))
            q=softmax(v,axis=-1).mean(1)
            gb=(q*w.sum(1)[:,None]-w).sum(0)/n+lam*b
            grad=np.array([gb[0]-gb[2],gb[1]-gb[2]])
            require(abs(objective-r['objective'])<=1e-10 and np.max(np.abs(grad))<=1e-7
                    and abs(b.sum())<=1e-10,'Constant solver not stationary')
            kp=softmax(outer.astype(np.float64)+b,axis=-1).mean(1).astype(np.float32)
            require(sha(folder/'endpoint_prob.npy')==r['prob_sha256'] and
                    np.max(np.abs(kp-np.load(folder/'endpoint_prob.npy')))<2e-6,
                    'Constant endpoint changed')
            arm='K_'+role;pred[arm][held]=kp[ids].argmax(1);sprob[arm][held]=kp[ids,2]
            p,extra,check,replay=replay_branch(fold,'P_'+role,outer,body,lengths)
            arm='P_'+role;pred[arm][held]=p[ids].argmax(1);sprob[arm][held]=p[ids,2]
            fit_diagnostics[arm+'_fold'+str(fold)]=check
            replays.append(replay)
            if role=='CF':
                mean=np.asarray(check['train_mean_residual'],dtype=np.float64)
                meanp=softmax(outer.astype(np.float64)+mean,axis=-1).mean(1)
                mean_pred[held]=meanp[ids].argmax(1)
    require(np.array_equal(pred['A0'],pd.read_parquet(PARENT/'expert_ASA_predictions.parquet')
                           .expert_pred_A.to_numpy()),'Hard A0 predictions changed')
    ledger=trace.copy()
    for arm in ARMS:
        ledger['pred_'+arm]=pred[arm];ledger['S_probability_'+arm]=sprob[arm]
    ledger['pred_P_CF_train_mean']=mean_pred
    ledger.to_parquet(OUT/'expert_ASA_predictions.parquet',index=False)
    rows=pd.read_parquet(ROWS,columns=['row_position','label_index','route'])
    folds=pd.read_parquet(FOLDS,columns=['row_position','root','proposed_fold'])
    require(len(rows)==len(folds)==len(official)==2056871 and
            np.array_equal(rows.row_position,np.arange(len(rows))) and
            np.array_equal(rows.label_index,official),'Complete task mapping/label changed')
    asa=np.flatnonzero(rows.route.eq('asa'))
    require(np.array_equal(asa,trace.row_position) and
            np.array_equal(folds.proposed_fold.to_numpy()[asa],fold_id) and
            np.array_equal(folds.root.to_numpy()[asa],trace.root),'ASA original-row mapping changed')
    teacher=teacher_predictions(len(rows),folds)
    other=~rows.route.eq('asa').to_numpy()
    require(int((teacher[other]!=official[other]).sum())==107,'Frozen non-ASA route changed')
    full={arm:teacher.copy() for arm in ARMS}
    for arm in ARMS:full[arm][asa]=pred[arm]
    fullledger=pd.DataFrame({'row_position':rows.row_position,'teacher_prediction':teacher,
                             **{'final_pred_'+arm:full[arm] for arm in ARMS}})
    fullledger.to_parquet(OUT/'full_prediction_ledger.parquet',index=False)
    metrics={'ASA':{},'full_task':{},'source_groups':{}}
    for arm in ARMS:
        metrics['ASA'][arm]=class_counts(pd.DataFrame({'truth':y,'pred':pred[arm]}),'pred')
        metrics['full_task'][arm]=class_counts(pd.DataFrame({'truth':official,'pred':full[arm]}),'pred')
        metrics['source_groups'][arm]=root_stats(trace,pred[arm])
    errors={arm:int((pred[arm]!=y).sum()) for arm in ARMS}
    mean_errors=int((mean_pred!=y).sum())
    folds_report=[{'fold':f,**{arm+'_errors':int(((pred[arm]!=y)&(fold_id==f)).sum())
                               for arm in ARMS}} for f in range(3)]
    transitions={arm:{str(cl):{'repaired':int(((y==cl)&(pred['A0']!=cl)&(pred[arm]==cl)).sum()),
                                   'regressed':int(((y==cl)&(pred['A0']==cl)&(pred[arm]!=cl)).sum())}
                      for cl in (1,2)} for arm in ARMS if arm!='A0'}
    source_groups=trace[['root','truth','fold']].copy()
    for arm in ARMS:source_groups[arm+'_wrong']=pred[arm]!=y
    source_groups=source_groups.groupby(['root','truth','fold'],as_index=False).agg(
        rows=('A0_wrong','size'),**{arm+'_errors':(arm+'_wrong','sum') for arm in ARMS})
    source_groups.to_csv(OUT/'source_group_changes.csv',index=False)
    ladder=pd.read_parquet(LADDER,columns=['row_position','diagnostic_bucket','A_all_seven_wrong','destination_key'])
    require(ladder.row_position.equals(trace.row_position),'Historical panel mapping changed')
    slices={'header_682':trace.row_position.isin(pd.read_parquet(
         ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet',columns=['row_position'])
         .row_position).to_numpy(),
        'persistent_S':(y==2)&ladder.A_all_seven_wrong.to_numpy(),
        'outside_top3_S':(y==2)&~np.isin(trace.root.to_numpy(),TOP3),
        'unknown_pooled_S':(y==2)&ladder.diagnostic_bucket.eq('unknown_parameter_pooled_support').to_numpy(),
        'V127_new_M_regressions':(y==1)&(pred['A0']==1)&
            (pd.read_parquet(BASE/'expert_ASA_predictions.parquet').pred_P.to_numpy()!=1)}
    for f in range(3):
        for cl in (1,2):slices[f'fold{f}_class{cl}']=(fold_id==f)&(y==cl)
    slice_report={name:{'original_rows':int(mask.sum()),**{arm+'_errors':int(((pred[arm]!=y)&mask).sum())
                        for arm in ARMS}} for name,mask in slices.items()}
    priority={}
    cards=read(ROOT/'artifacts/v124_header_trial_20260929/capability_qualification.json')['cards']
    for arm in ARMS:
        roles=[]
        for f in range(3):
            if arm=='A0':p=np.load(PARENT/f'fold{f}_A/epoch25_prob.npy')
            elif arm=='AH':p=np.load(ROOT/f'artifacts/v127_plan_review_20260929/fold{f}_A_header_probability.npy')
            else:p=np.load(OUT/f'fold{f}_{arm}'/('endpoint_prob.npy' if arm.startswith('K_') else 'epoch50_prob.npy'))
            for card in cards:
                if card['fold']!=f:continue
                q=(trace.fold.ne(f)&ladder.destination_key.eq(card['destination_key'])&trace.truth.eq(2))
                for row in trace.loc[q,['row_position','local','root']].itertuples(index=False):
                    roles.append((int(row.row_position),int(row.root),int(p[int(row.local)].argmax())!=2))
        q=pd.DataFrame(roles,columns=['row_position','root','wrong'])
        require(len(q)==206 and q.row_position.nunique()==136,'Priority S population changed')
        priority[arm]={'S_fit_roles':len(q),'S_unique_original_rows':q.row_position.nunique(),
            'S_wrong_roles':int(q.wrong.sum()),'S_wrong_unique_original_rows':q.loc[q.wrong,'row_position'].nunique(),
            'S_wrong_roots':q.loc[q.wrong,'root'].nunique()}
    a=metrics['ASA']['A0'];p=metrics['ASA']['P_CF'];fa=metrics['full_task']['A0'];fp=metrics['full_task']['P_CF']
    outside=slices['outside_top3_S'];groups=metrics['source_groups']
    gates={'all_21_fits_and_fixed_endpoints':len(teacher_receipts)==9 and len(replays)==6 and
              sum(x['steps'] for x in teacher_receipts)==8900 and sum(x['steps'] for x in replays)==17800,
        'ASA_M_errors_le_318':p['1']['missed']<=318,
        'ASA_S_errors_le_2074':p['2']['missed']<=2074,
        'ASA_total_errors_le_2170':errors['P_CF']<=2170,
        'P_CF_beats_P_IS_total':errors['P_CF']<errors['P_IS'],
        'P_CF_beats_K_CF_total':errors['P_CF']<errors['K_CF'],
        'P_CF_beats_own_train_mean':errors['P_CF']<mean_errors,
        'ASA_M_S_precision_recall_F1_no_regression':all(p[str(c)][m]>=a[str(c)][m]
            for c in (1,2) for m in ('precision','recall','f1')),
        'full_each_class_precision_recall_F1_no_regression':all(fp[str(c)][m]>=fa[str(c)][m]
            for c in (0,1,2) for m in ('precision','recall','f1')),
        'at_least_2_improving_folds':sum(q['P_CF_errors']<q['A0_errors'] for q in folds_report)>=2,
        'outside_top3_S_no_regression':int((pred['P_CF'][outside]!=y[outside]).sum())<=1373,
        'S_group_mean_no_regression':groups['P_CF']['mean_recall']>=groups['A0']['mean_recall'],
        'S_zero_recall_groups_no_increase':groups['P_CF']['zero_recall_roots']<=198,
        'header_682_no_regression':slice_report['header_682']['P_CF_errors']<=slice_report['header_682']['A0_errors'],
        'non_ASA_frozen_and_full_population':len(fullledger)==2056871 and int((teacher[other]!=official[other]).sum())==107}
    result={'status':'V128_21_fit_source_closed_evaluated','latest_actual_training':'V128',
        'classifier_fits_new':21,'teacher_fits':9,'constant_fits':6,'branch_fits':6,
        'neural_optimizer_steps':sum(x['steps'] for x in teacher_receipts)+sum(x['steps'] for x in replays),
        'ASA_original_rows':len(y),'full_official_rows':len(official),
        'teacher_replay':teacher_receipts,'branch_replay':replays,'metrics':metrics,
        'ASA_total_errors':errors,'P_CF_train_mean_replacement_total_errors':mean_errors,
        'paired_changes_vs_A0':transitions,'folds':folds_report,'slices':slice_report,
        'priority_fit_S_136_unique_206_roles':priority,'fit_endpoint_diagnostic':fit_diagnostics,
        'quality_gates':{k:bool(v) for k,v in gates.items()},
        'primary_quality_passed':bool(all(gates.values())),
        'confirmation_allowed':bool(all(gates.values())),
        'confirmation_executed':False,'model_promoted':False,'quality_acceptance':False,
        'evaluation_source_sha256':sha(__file__),'training_seal_sha256':sha(OUT/'run_seal.json'),
        'limitations':['These source-closed development folds have been repeatedly inspected.',
            'No independent source/time/external validation or official submission in V128.',
            'Cross-fitting changes score exposure and teacher fit size; it adds no new attack evidence.']}
    save(target,result)
    print(json.dumps({'stage':result['status'],'ASA_errors':{arm:{str(c):metrics['ASA'][arm][str(c)]['missed']
          for c in (1,2)} for arm in ARMS},'total_errors':errors,'mean_replacement':mean_errors,
          'gates':gates,'confirmation_allowed':result['confirmation_allowed']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
