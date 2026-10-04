"""No-fit attribution of the executed 2x2 trial, with original-row counts."""
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,read,save,sha,load_sparse
from v79_execute import rows
from v82_capacity import LAST
from v85_protection import Residual,csr_tensor

DEST=ROOT/'artifacts/v87_solver_supervision_r2_20260927'
OLD=ROOT/'artifacts/v85_protection_20260927'
ARMS=['A0','A1','A2','A3']


def representation(x,ids,state):
    model=Residual().cuda();model.load_state_dict(state);result=np.empty((len(ids),64),np.float32)
    with torch.no_grad():
        for start in range(0,len(ids),8192):
            v=csr_tensor(x[ids[start:start+8192]])
            first=torch.nn.functional.gelu(torch.sparse.mm(v,model.first.weight.T),approximate='tanh')
            result[start:start+8192]=torch.nn.functional.gelu(model.second(first),approximate='tanh').cpu().numpy()
    return result


def main():
    assert read(DEST/'verification.json')['status']=='passed'
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X')
    folder=DEST/'fold1';selected=np.load(folder/'selected_rows.npy');old=np.load(OLD/'fold1/teacher_prediction.npy')
    cover=pd.read_parquet(folder/'pair_coverage.parquet');paired=cover.loc[cover.eligible,'row_position'].to_numpy()
    fit=~r.fold.isin([0,1,2]).to_numpy()
    full=np.bincount(fid[fit]*3+y[fit],minlength=x.shape[0]*3).reshape(-1,3)
    pmass=full[np.arange(len(old)),old]
    wrong=fit&(old[fid]!=y);blocked=wrong&(pmass[fid]>0);free=wrong&~blocked
    feasibility={'old_train_errors':int(wrong.sum()),
        'same_encoded_input_has_protected_correct_rows':int(blocked.sum()),
        'no_same_input_protected_correct_rows':int(free.sum()),
        'blocked_per_class':np.bincount(y[blocked],minlength=3).tolist(),
        'unblocked_per_class':np.bincount(y[free],minlength=3).tolist(),
        'unblocked_unique_inputs':int(np.unique(fid[free]).size),
        'scope':'Same fixed encoded-input row constraint only. Not proof that raw inputs are identical, nor that shared model parameters can repair every unblocked input while preserving all other boundaries.'}
    for k,v in read(DEST/'verification.json')['same_input_feasibility'].items():assert feasibility[k]==v,k
    cc=selected[(r.route.to_numpy()[selected]=='asa')&(y[selected]>0)]
    cohorts=[]
    for cls in [1,2]:
        for wrong in [False,True]:
            for eligibility in [False,True]:
                take=cc[(y[cc]==cls)&((old[fid[cc]]!=y[cc])==wrong)&(np.isin(cc,paired)==eligibility)]
                cohorts.append({'class':cls,'old_wrong':wrong,'pair_eligible':eligibility,'rows':take})
    summaries=[];pairmetrics=[];margin_cases=[];decomposition=[]
    pairs=pd.read_parquet(folder/'pairs.parquet');pids=np.unique(pairs[['anchor_fid','positive_fid','negative_fid']].to_numpy())
    pidx=[np.searchsorted(pids,pairs[k]) for k in ['anchor_fid','positive_fid','negative_fid']]
    z0=np.load(OLD/'fold1/teacher_scores.npy',mmap_mode='r')
    for arm in ARMS:
        report=read(folder/(arm+'_fit.json'));name=report['saved_states'][-1]
        pred=np.load(folder/(name+'_all_prediction.npy'));progress=read(folder/(arm+'_progress.json'))
        qp=[a for p in progress for a in p['QP_records'] if 'active_constraints' in a]
        summary={'arm':arm,'final':name,'attempts':len(progress),'accepted':sum(p['accepted'] for p in progress),
                 'accepted_after_epoch32':sum(p['accepted'] for p in progress if p['epoch']>32),
                 'QP_solves':len(qp),'max_active_constraints':max([a['active_constraints'] for a in qp] or [0]),
                 'min_nonzero_proposal_fraction':min([p['proposal_fraction'] for p in progress if p['accepted']] or [0]),
                 'first_and_last_losses':{k:[progress[0][k],progress[-1][k]] for k in ['main_before','contrastive_before','hinge_before']},
                 'cohorts':[],'stop':report['stopped_reason'],
                 'trajectory':{'max_observed_train_repairs':max(p['selected_fit_positive'] for p in progress),
                    'best_observed_train_error':min(p['selected_fit_errors'] for p in progress),
                    'final_train_repairs':progress[-1]['selected_fit_positive'],
                    'earlier_best_epoch':min(progress,key=lambda p:(p['selected_fit_errors'],p['epoch']))['epoch'],
                    'objective_increasing_transitions':[
                        {'from_epoch':a['epoch'],'to_epoch':b['epoch'],'objective_change':b['objective_before']-a['objective_before']}
                        for a,b in zip(progress,progress[1:]) if b['objective_before']-a['objective_before']>1e-7],
                    'classification_worse_while_objective_decreases':[
                        {'epoch':a['epoch'],'objective_change':b['objective_before']-a['objective_before'],
                         'error_change':a['selected_fit_errors']-(315 if i==0 else progress[i-1]['selected_fit_errors'])}
                        for i,(a,b) in enumerate(zip(progress,progress[1:]))
                        if b['objective_before']<a['objective_before'] and a['selected_fit_errors']>(315 if i==0 else progress[i-1]['selected_fit_errors'])],
                    'scope':'Observed before-step objective sequence and after-step decision counts; not new checkpoint selection.'}}
        for c in cohorts:
            take=c['rows'];new=pred[fid[take]];base=old[fid[take]];true=y[take]
            summary['cohorts'].append({k:v for k,v in c.items() if k!='rows'}|{'rows':len(take),
                'correct':int((new==true).sum()),'repairs':int(((base!=true)&(new==true)).sum()),
                'regressions':int(((base==true)&(new!=true)).sum()),'unique_inputs':int(np.unique(fid[take]).size)})
        summaries.append(summary)
        for stage in [report['saved_states'][0],name]:
            state=torch.load(folder/(stage+'.pt'),map_location='cpu',weights_only=True)['model']
            h=representation(x,pids,state);h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-8)
            ap=(h[pidx[0]]*h[pidx[1]]).sum(1);an=(h[pidx[0]]*h[pidx[2]]).sum(1)
            for (cls,grade),positions in pairs.groupby(['anchor_class','grade']).groups.items():
                ix=np.asarray(positions);pairmetrics.append({'name':stage,'class':int(cls),'grade':grade,'rows':len(ix),
                    'mean_positive_similarity':float(ap[ix].mean()),'mean_negative_similarity':float(an[ix].mean()),
                    'positive_gt_negative_fraction':float((ap[ix]>an[ix]).mean()),
                    'scope':'Train pairing representation statistic only; not classifier accuracy or transfer acceptance.'})
            del state;torch.cuda.empty_cache()
        # Targeted old regression inputs are diagnostic probes, never loss data.
        state=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True)['model']
        from v85_verify import cpu_manual
        for c in cohorts:
            if not c['old_wrong'] or len(c['rows'])==0:continue
            take=c['rows'];uf,inverse=np.unique(fid[take],return_inverse=True)
            delta=cpu_manual(x[uf],state)[inverse];base_scores=np.asarray(z0[fid[take]])
            ix=np.arange(len(take));true=y[take];winner=old[fid[take]]
            wrong_gap=base_scores[ix,winner]-base_scores[ix,true]
            gain=delta[ix,true]-delta[ix,winner]
            final_gap=wrong_gap-gain
            assert np.array_equal((base_scores+delta).argmax(1),pred[fid[take]]), (arm,c['class'],c['pair_eligible'])
            decomposition.append({'arm':arm,'class':c['class'],'pair_eligible':c['pair_eligible'],'rows':len(take),
                'positive_true_vs_old_winner_gain_rows':int((gain>0).sum()),
                'old_winner_overcome_rows':int((final_gap<0).sum()),
                'total_correct_rows':int(((base_scores+delta).argmax(1)==true).sum()),
                'median_old_wrong_gap':float(np.median(wrong_gap)),
                'median_true_vs_old_winner_gain':float(np.median(gain)),
                'scope':'Original-row-weighted CPU decomposition of teacher plus residual logits; positive compensation is not standalone residual accuracy or proof of learned intent.'})
        ids=np.array([2128,12962,18997]);d=cpu_manual(x[ids],state)
        for f,delta in zip(ids,d):
            margin_cases.append({'arm':arm,'fid':int(f),'old_M_S_margin':float(z0[f,1]-z0[f,2]),
                 'final_M_S_margin':float(z0[f,1]-z0[f,2]+float(delta[1])-float(delta[2])),
                 'final_prediction':int(pred[f]),'scope':'Previously inspected diagnostic inputs, not a held-out new test.'})
    out={'source_sha256':sha(__file__),'verification_sha256':sha(DEST/'verification.json'),
         'new_classifier_fits_in_audit':0,'new_calibration_fits_in_audit':0,
         'output_margin_decomposition':decomposition,'same_input_feasibility':feasibility,'arms':summaries,'pair_representation_statistics':pairmetrics,'targeted_margin_cases':margin_cases,
         'scope':'Post-selection no-fit counts and trained representation diagnostics. Loss, representation separation and accepted updates are mechanisms, not quality acceptance.'}
    save(DEST/'problem_ledger.json',out)
    print(json.dumps({'arms':summaries,'targeted_margin_cases':margin_cases},ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=4):main()
