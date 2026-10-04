"""Register and run matched frozen representations and four protected readout fits."""
import argparse
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from v89_common import *
from v89_solver import Problem,self_check,MAX_CUTS,MAX_ROUNDS,SOLVE_SECONDS,PROBLEM_SECONDS,REFINE_STEPS
from v85_protection import Residual,csr_tensor
from v87r2_execute import hidden
from v81_training_contract import compare


def register():
    assert read(DEST/'preparation.json')['status']=='preparation_complete'
    assert read(DEST/'contract_preparation.json')['status']=='contract_prepared'
    assert not (DEST/'registration.json').exists()
    sources=list((ROOT/'training').glob('v89*.py'))+[ROOT/'training/test_v89_solver.py']
    inputs=[ROOT/'docs/V88_ROOT_CAUSE_AND_TRAINING_PLAN.md',ROOT/'data/official/train.parquet',OUT/'rows.parquet',OUT/'projections.parquet',
      LAST/'X.data',LAST/'X.indices',LAST/'X.indptr',LAST/'row_feature_id.npy',
      PRIOR/'fold1/selected_rows.npy',OLD/'teacher_scores.npy',OLD/'teacher_prediction.npy',
      PRIOR/'fold1/A0_epoch020.pt',PRIOR/'fold1/A3_epoch020.pt',DEST/'preparation.json',DEST/'contract_preparation.json',DEST/'actual_R0_stress.json',DEST/'support_design.json',DEST/'partial_facts.parquet']
    inputs+=list(DEST.glob('support_*_rows.npy'))
    reg={'version':'v89-readout-and-support-v1','source_bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
      'input_bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
      'plan':'v8.8 phase1 partial facts -> matched readout -> independently executed support controls -> conditional expression/rotation/final fit',
      'teacher_and_input':'Unchanged R0 and fold1 v85 teacher for readout isolation; same epoch20 A0/A3 frozen weights. Partial facts audited separately; no silent input replacement in readout contrast.',
      'arms':{'R00':['A0',False],'R01':['A0',True],'R10':['A3',False],'R11':['A3',True]},
      'hidden_execution':'float64 forward of fixed float32 parameters, same GELU-tanh, GPU batched; readout and all guards float64 CPU; independent CPU replay required',
      'conditioning':'Invertible per-column RMS over training rows only, original-frequency mass, no centering; constant1 not rescaled. Returned model records scales.',
      'positive_error_margin':DELTA,'P':'All original teacher-correct train rows; exact epsilon min(.001,old_margin/2), search buffer min(1e-5,old_margin/4)',
      'constraint_tolerance':TOL,'max_cuts':MAX_CUTS,'max_cut_rounds':MAX_ROUNDS,'solver_seconds':SOLVE_SECONDS,'problem_seconds':PROBLEM_SECONDS,
      'readout_objective':'All nonconflict E target inequalities; if infeasible minimize equal-M/S mean row-frequency hinge slack with hard P, then L1 nearest-parameter solution, then original-frequency OVR risk+alpha/2 L2 under these constraints',
      'norm':'L1 distance in fixed RMS-conditioned basis for nearest feasible step; ties refined by registered risk, no validation tuning',
      'slack_lexicographic_tolerance':1e-8,'alpha':ALPHA,'risk_refine_max_steps':REFINE_STEPS,
      'head_initialization':'zero residual, free teacher transform=identity; 22 input-conflict E kept in main risk/evaluation, not assigned contradictory hard target constraints',
      'linear_feasibility':'Each arm noP/P; resource or solver error unresolved. Numerical Farkas residual recorded, no raw-data or other-capacity impossibility claim.',
      'selection':'Primary eligibility requires full P and original train zero NF, actual train repair, inner exact per-class nonregression and zero NF, ASA inner positive repairs. Rank inner ASA errors then inner total errors then fixed arm name. Frozen before C/H.',
      'minimum_effect':'Fixed candidate across folds1/3/4: aggregate ASA original errors decrease>=10%, repairs in>=2folds; zero old-correct M/S regression per fold plus previous class guards. No further epoch selection.',
      'support_controls':['full','withdraw_1','control_1','withdraw_2','control_2'],
      'support_independent_of_quality':True,'support_fits':'Fresh zero-initialized original-frequency R0 OVR+1e-6 L2, L-BFGS-B maxiter1000/gtol1e-6/ftol1e-12; gradient_inf<=1e-5 convergence. No teacher/representation/cache trained on removed labels.',
      'support_seconds_per_fit':900,'support_max_groups':3,'development_not_blind':True,
      'fallback':'If readout infeasible/has no ASA-transfer gain and verified new facts exist, register one partial-evidence expression branch before its fit; freeze training-only design. No parameter search on C/H.',
      'final_fit':'Only after registered threefold, support and locked-regression gates; all native_flow M32596 then included. Complete raw-input engineering replay and original v79 gates still required.',
      'external_data':False,'pseudo_labels':False,'platform_used':False,'optimizer_tests':self_check(),
      'memory_scope':'Disk-backed R0; batch hidden and P scans; cut representation covers full population via exact separation, no dropped guards',
      'preparation_failure_receipt':'artifacts/v89_preparation_failure_20260927.json',
      'runtime':{'numpy':np.__version__,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}}
    save(DEST/'registration.json',reg);emit(stage='registered',arms=list(reg['arms']),support_conditions=reg['support_controls'])


def encode_hidden():
    check();x=load_sparse(LAST/'X');ids=np.arange(x.shape[0]);start=time.monotonic()
    for arm in ['A0','A3']:
        path=DEST/(arm+'_hidden.npy');assert not path.exists()
        model=Residual().double().cuda();bundle=torch.load(PRIOR/f'fold1/{arm}_epoch020.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(bundle['model']);h=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(len(ids),64))
        with torch.no_grad():
            for beg in range(0,len(ids),1024):
                ix=ids[beg:beg+1024];xt=csr_tensor(x[ix]).double();h[beg:beg+len(ix)]=hidden(model,xt).cpu().numpy()
                if beg%65536==0:emit(stage='frozen_hidden',arm=arm,rows=beg,seconds=round(time.monotonic()-start,1))
        h.flush();del model,bundle,h;torch.cuda.empty_cache();gc.collect()
    save(DEST/'hidden_receipt.json',{'new_classifier_fits':0,'precision':'float64','source_sha256':sha(__file__),
      'outputs_sha256':{a+'_hidden.npy':sha(DEST/(a+'_hidden.npy')) for a in ['A0','A3']},'seconds':time.monotonic()-start})


def readout(name):
    reg=check();arm,free=reg['arms'][name];assert not (DEST/(name+'_fit.json')).exists()
    li=ledger_start('classifier_fit',name);start=time.monotonic()
    r,y,fid,z0,old,sel,fit=data();n=len(old)
    full=raw_counts(fid,y,fit,n);mask=np.zeros(len(r),bool);mask[sel]=True;cc=raw_counts(fid,y,mask,n);used=np.flatnonzero(cc.sum(1))
    pmass=full[np.arange(n),old];P=np.flatnonzero(pmass);truth=full.argmax(1)
    E=np.flatnonzero((full.sum(1)>0)&(pmass==0));assert (full[E]>0).sum(1).max()==1
    assert int(full[E].sum())==293 and len(E)==146
    weights=np.zeros(len(E));et=truth[E]
    for cls in [1,2]:
        m=et==cls;weights[m]=full[E[m],cls]/full[E[m],cls].sum()/2
    hh=np.load(DEST/(arm+'_hidden.npy'),mmap_mode='r');block=np.c_[np.asarray(z0[used]),hh[used]] if free else hh[used]
    scale=np.sqrt(np.square(block).T@cc[used].sum(1)/cc.sum());scale=np.where(scale>1e-12,scale,1.)
    d=len(scale)+1;V=np.lib.format.open_memmap(DEST/(name+'_basis.npy'),mode='w+',dtype=np.float64,shape=(n,d))
    for beg in range(0,n,16384):
        b=np.c_[np.asarray(z0[beg:beg+16384]),hh[beg:beg+16384]] if free else hh[beg:beg+16384]
        V[beg:beg+len(b),:-1]=b/scale;V[beg:beg+len(b),-1]=1.
    V.flush();np.save(DEST/(name+'_scale.npy'),scale)
    margin=z0[np.arange(n),old]-np.max(np.where(np.eye(3,dtype=bool)[old],-np.inf,z0),axis=1)
    assert margin[P].min()>0
    epsilon=np.minimum(.001,margin/2)+np.minimum(1e-5,margin/4)
    problem=Problem(V,z0,P,epsilon,E,et,weights,name)
    startE,fe=problem.linear(False);startP,fp=problem.linear(True)
    report={'name':name,'representation':arm,'free_teacher_coefficients':free,'new_classifier_fits':1,
      'P_original_rows':int(pmass.sum()),'P_input_groups':len(P),'E_original_rows':int(full[E].sum()),'E_input_groups':len(E),
      'noP_feasibility':fe,'P_feasibility':fp,'source_sha256':sha(__file__),'data_role_M_S_all_used':True}
    save(DEST/(name+'_feasibility.json'),report)
    if startP is None:
        startP,slack=problem.linear(True,True);report['slack_optimization']=slack
        if startP is not None:problem.slack_cap=float(weights@startP[problem.n:])+reg['slack_lexicographic_tolerance']
    if startP is None:
        report.update(status='aborted_no_verified_start',seconds=time.monotonic()-start);save(DEST/(name+'_fit.json'),report);ledger_end(li,'aborted',reason=report['status']);return
    model,training=problem.train(startP,used,cc[used]);report['training']=training
    if model is None:
        report.update(status='aborted_readout_solver',seconds=time.monotonic()-start);save(DEST/(name+'_fit.json'),report);ledger_end(li,'aborted',reason=report['status']);return
    np.savez_compressed(DEST/(name+'_model.npz'),theta=model,scale=scale,cuts=np.asarray(problem.cuts,dtype=np.int64),
       E=E,truth=et,weights=weights,slack_cap=np.array(np.nan if problem.slack_cap is None else problem.slack_cap),
       uses_slack=np.array(problem.slack),representation=np.array(arm),free_teacher=np.array(free))
    pred=problem.scores(model).argmax(1).astype(np.int8);np.save(DEST/(name+'_all_prediction.npy'),pred)
    roles,cells=evaluate(r,y,fid,old,pred,fit,masks={'fit_full':fit,'inner':r.fold.eq(1).to_numpy()})
    cells.to_csv(DEST/(name+'_selection_classwise.csv'),index=False)
    basecc=raw_counts(fid,y,r.fold.eq(1).to_numpy(),n);guard=compare(cm_from_counts(basecc,old),np.array(roles['inner']['cm']),True)
    asa=cells[(cells.role=='inner')&(cells.route=='asa')]
    eligible=roles['fit_full']['negative_flips']==0 and roles['fit_full']['positive_flips']>0 and roles['inner']['negative_flips']==0 and guard['eligible'] and asa.repairs.sum()>0
    fullcheck=problem.scan(model,True);assert fullcheck['max_full_violation']<=TOL and fullcheck['protected_changed_inputs']==0
    report.update(status='completed',roles=roles,inner_ASA={'repairs':int(asa.repairs.sum()),'regressions':int(asa.regressions.sum()),
        'errors':int((asa.support-asa.new_correct).sum()),'baseline_errors':int((asa.support-asa.old_correct).sum())},
        inner_metric_guard=guard,primary_eligible=bool(eligible),full_constraint_check=fullcheck,seconds=time.monotonic()-start,
        model_sha256=sha(DEST/(name+'_model.npz')),risk_converged=bool(training.get('converged',False)))
    save(DEST/(name+'_fit.json'),report);ledger_end(li,'completed',eligible=bool(eligible),seconds=report['seconds'])
    emit(stage='readout_complete',name=name,fit_errors=roles['fit_full']['errors'],inner_errors=roles['inner']['errors'],inner_NF=roles['inner']['negative_flips'],asa=report['inner_ASA'],eligible=bool(eligible))


def select():
    reg=check();reports=[read(DEST/(n+'_fit.json')) for n in reg['arms']]
    good=[r for r in reports if r.get('primary_eligible')]
    chosen=min(good,key=lambda a:(a['inner_ASA']['errors'],a['roles']['inner']['errors'],a['name'])) if good else None
    save(DEST/'selection.json',{'primary_selected':None if chosen is None else chosen['name'],'selected_before_C_H':True,
      'primary_statuses':{a['name']:a['status'] for a in reports},'support_controls_still_execute':True,
      'source_sha256':sha(__file__),'reason':'ASA_and_class_guards_passed' if chosen else 'no_ASA_candidate_passed_primary_guards'})
    # Fixed-state diagnostics only after candidate selection is on disk.
    r,y,fid,z0,old,sel,fit=data()
    for a in reports:
        if a['status']!='completed':continue
        new=np.load(DEST/(a['name']+'_all_prediction.npy'));roles,cells=evaluate(r,y,fid,old,new,fit)
        save(DEST/(a['name']+'_full_diagnosis.json'),roles);cells.to_csv(DEST/(a['name']+'_classwise.csv'),index=False)
    emit(stage='selection_frozen',winner=None if chosen is None else chosen['name'],support_controls_will_execute=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['register','hidden','readout','select']);ap.add_argument('--arm');args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=4):
        if args.stage=='readout':readout(args.arm)
        else:{'register':register,'hidden':encode_hidden,'select':select}[args.stage]()
