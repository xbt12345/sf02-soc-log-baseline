"""Frozen-model audit only: no optimizer, no LP/QP solver, no parameter updates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from v138_runtime import ROOT,OUT as PREVIOUS,read,save,sha,require_run_seal
from v135_runtime import load_data,fit_context
from v138_train import configure,cache_tensors
from v135_model import tensor_hash

OUT=ROOT/'artifacts/v139_training_decision_review_20261001'


def main():
    require_run_seal(ROOT/'training/v138_train.py');configure()
    if OUT.exists():raise FileExistsError('Preserve diagnosis evidence')
    OUT.mkdir();_,d=load_data()
    trace=pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet',columns=['row_position','facts_json'])
    facts_by_row=trace.set_index('row_position').facts_json
    panels=[];errors=[]
    for fold in range(3):
        frame,c,pure,_,ids=fit_context(d,fold);hidden,facts,model=cache_tensors(fold)
        state=torch.load(PREVIOUS/f'fold{fold}_H_L/endpoint_readout.pt',map_location='cpu',weights_only=True)['readout']
        model.load_state_dict(state)
        for v in model.parameters():v.requires_grad_(False)
        before=tensor_hash(model.state_dict());zs=[]
        with torch.no_grad():
            for start in range(0,len(c),2048):zs.append(model(hidden[start:start+2048],facts[start:start+2048]).cpu().numpy())
        z=np.concatenate(zs);q=torch.softmax(torch.from_numpy(z),-1).mean(1).numpy();saved=np.load(PREVIOUS/f'fold{fold}_H_L/sealed_all_prob.npy')
        if not np.array_equal(q.argmax(1),saved.argmax(1)) or np.abs(q-saved).max()>2e-6:raise ValueError('Actual frozen prediction replay changed')
        loc=frame.local.to_numpy();y=frame.truth.to_numpy();pr=pure[loc].astype(bool);pred=q[loc].argmax(1);members=z.argmax(-1)
        own=z[loc[:,None],np.arange(16)[None,:],y[:,None]]
        allmargin=own[:,:,None]-z[loc];allmargin[np.arange(len(frame))[:,None],np.arange(16)[None,:],y[:,None]]=np.inf
        member_correct=(members[loc]==y[:,None]).sum(1)
        member_wrong=(member_correct<16);margin1bad=(allmargin.min((1,2))<1-1e-6)
        facts_frame=frame.assign(facts_json=frame.row_position.map(facts_by_row))
        supports=facts_frame.groupby(['facts_json','truth']).agg(rows=('row_position','size'),roots=('root','nunique'))
        # Exact feature collision across legal TRAIN labels: concatenated frozen h2 and facts.
        hh=np.load(PREVIOUS/f'fold{fold}_hidden.npy');ff=np.load(PREVIOUS/'facts.npy')
        featurekeys=[hashlib.sha256(hh[i].tobytes()+ff[i].tobytes()).hexdigest() for i in range(len(c))]
        role=frame.assign(frozen_readout_key=frame.local.map(dict(enumerate(featurekeys))))
        classes=role.groupby('frozen_readout_key').truth.nunique()
        conflicting=set(classes[classes>1].index)
        blocked=role.frozen_readout_key.isin(conflicting).to_numpy()
        residual=frame.loc[pr&(pred!=y)].copy();residual['training_role']=fold
        for cl in [1,2]:
            mask=pr&(y==cl);good=mask&(pred==y)
            panels.append({'fold':fold,'class':cl,'pure_original_role_rows':int(mask.sum()),'ensemble_errors':int((mask&(pred!=y)).sum()),
                'ensemble_correct_but_some_member_wrong':int((good&member_wrong).sum()),
                'ensemble_correct_but_some_margin_below_one':int((good&margin1bad).sum()),
                'member_correct_count_histogram':{str(k):int((mask&(member_correct==k)).sum()) for k in range(17)},
                'pure_rows_with_exact_frozen_feature_label_collision':int((mask&blocked).sum())})
        same=frame.groupby(['local','truth']).agg(rows=('row_position','size'),roots=('root','nunique'))
        for row in residual.itertuples():
            ii=int(row.local);yy=int(row.truth);fact=facts_by_row.loc[row.row_position];record={
                'training_role':fold,'row_position':int(row.row_position),'local':ii,'root':int(row.root),'truth':yy,'pred':int(q[ii].argmax()),
                'canonical_key':row.canonical_key,'correct_members':int((members[ii]==yy).sum()),
                'truth_probability':float(q[ii,yy]),'class_probability_margin':float(q[ii,yy]-np.delete(q[ii],yy).max()),
                'minimum_member_logit_margin':float(allmargin[np.where(loc==ii)[0][0]].min()),
                'same_input_class_TRAIN_rows':int(same.loc[(ii,yy),'rows']),'same_input_class_TRAIN_roots':int(same.loc[(ii,yy),'roots']),
                'exact_frozen_feature_label_conflict':featurekeys[ii] in conflicting,
                'facts_sha256':hashlib.sha256(fact.encode()).hexdigest()}
            for cl in [1,2]:
                s=supports.loc[(fact,cl)] if (fact,cl) in supports.index else None
                record[f'same_facts_class{cl}_rows']=int(s.rows) if s is not None else 0
                record[f'same_facts_class{cl}_roots']=int(s.roots) if s is not None else 0
            errors.append(record)
        if tensor_hash(model.state_dict())!=before:raise ValueError('Inference mutated model')
        del hidden,facts,model,hh,ff,z;gc.collect();torch.cuda.empty_cache()
    err=pd.DataFrame(errors);err.to_parquet(OUT/'remaining16_diagnosis.parquet',index=False)
    save(OUT/'member_requirement_audit.json',{'status':'frozen_actual_model_diagnosis','new_classifier_fits':0,'optimizer_updates':0,'supervised_solver_runs':0,
        'pure_role_rows':sum(p['pure_original_role_rows'] for p in panels),'ensemble_correct_but_some_member_wrong':sum(p['ensemble_correct_but_some_member_wrong'] for p in panels),
        'ensemble_correct_but_some_margin_below_one':sum(p['ensemble_correct_but_some_margin_below_one'] for p in panels),
        'remaining_role_errors':len(err),'remaining_independent_rows':int(err.row_position.nunique()),'remaining_input_keys':int(err.canonical_key.nunique()),
        'residual_exact_frozen_feature_label_conflicts':int(err.exact_frozen_feature_label_conflict.sum()),'fold_class_panels':panels,
        'residual_correct_members_histogram':{str(k):int(err.correct_members.eq(k).sum()) for k in range(17)},
        'limitations':['Member correctness is not a label-free routing method.',
            'No exact feature collision found does not prove affine separability or missing semantic evidence absent.',
            'All-member margin>=1 is sufficient but stronger than actual mean-softmax inference.',
            'CPU/GPU probability rounding differences tolerated only within2e-6 with exact original decisions.']})
    bindings={}
    for f in range(3):
        for path in [PREVIOUS/f'fold{f}_H_L/endpoint_readout.pt',PREVIOUS/f'fold{f}_hidden.npy',PREVIOUS/f'fold{f}_H_L/sealed_all_prob.npy']:
            bindings[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in [Path(__file__),PREVIOUS/'facts.npy',PREVIOUS/'final_delivery.json',ROOT/'training/v138_runtime.py',ROOT/'training/v138_readout.py',ROOT/'training/v138_train.py',ROOT/'training/v135_runtime.py',ROOT/'training/v135_model.py',ROOT/'training/v131_common.py',ROOT/'training/v131_model.py',ROOT/'training/experiment_review.py']:
        bindings[path.relative_to(ROOT).as_posix()]=sha(path)
    save(OUT/'diagnosis_receipt.json',{'status':'readonly_replay_finished','new_classifier_fits':0,'optimizer_updates':0,'supervised_solver_runs':0,
        'all_source_seals_intact':True,'source_and_input_sha256':bindings,
        'output_sha256':{p.name:sha(p) for p in OUT.glob('*') if p.is_file()},'not_task_quality_or_training_authorization':True})
    print(read(OUT/'member_requirement_audit.json'),flush=True)


if __name__=='__main__':main()
