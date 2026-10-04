"""Only two failed fixed endpoints; cached qualified d, zero new gradients."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,importlib.metadata,json,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import context,Counter,configure,CurrentInputBoundary,tensor_hash,risk,probabilities,restore,stats
from v159_float64_repeat_policy_v2 import repeat_values
from v160_fixed_endpoint_diagnostic_v3 import probe,store_scope,joint_check
PRIOR=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
DIAG=ROOT/'artifacts/v160_fixed_endpoint_diagnostic_20261002'
QUAL=ROOT/'artifacts/v160_saved_vector_numeric_polish_qualification_20261002'
OUT=ROOT/'artifacts/v160_cached_polished_direction_finite_probe_20261002'
PLAN=ROOT/'training/review_policy/v160_cached_polished_direction_finite_probe_contract.json'
PROTOCOL='V160-two-cached-polished-fixed-endpoint-finite-probes-v1'
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read_endpoint_rows(role):return pd.read_parquet(DIAG/f'role{role}/baseline_class1/OOF_original_rows.parquet')
def require():
    seal=read(OUT/'run_seal.json');plan=read(PLAN)
    assert seal['protocol']==plan['protocol']==PROTOCOL and seal['allowed_entries']==['training/v160_cached_polished_direction_finite_probe.py']
    assert seal['plan_sha256']==sha(PLAN);check_bindings(seal['source_sha256'])
    assert sys.version==seal['python_version'] and {n:importlib.metadata.version(n) for n in seal['package_versions']}==seal['package_versions']
    assert plan['new_caps']==dict(heads=836,features=836,class_gradients=0,margin_gradients=0,finite_proposals=40,fits=0,permanent_updates=0)
    return plan

def run(role):
    plan=require();assert role in [1,2];configure();spec=next(s for s in plan['roles'] if s['role']==role)
    folder=OUT/f'role{role}';assert not folder.exists();folder.mkdir()
    ctx=context(role)
    frozen_rows=read_endpoint_rows(role)
    assert np.array_equal(ctx['OOF_rows'].row_position,frozen_rows.row_position) and np.array_equal(ctx['OOF_rows'].truth,frozen_rows.truth)
    endpoint_correct=(frozen_rows.pred.eq(frozen_rows.truth)&frozen_rows.pure_current_input).to_numpy()
    # Preserve the fixed endpoint's existing correct rows, including its repairs.
    ctx['OOF_rows']=ctx['OOF_rows'].copy()
    ctx['OOF_rows']['protected_correct']=ctx['OOF_rows'].protected_correct.to_numpy()|endpoint_correct
    ctx['OOF_rows'].loc[ctx['OOF_rows'].protected_correct].to_parquet(folder/'fixed_endpoint_correct_protection_rows.parquet',index=False)
    model=CurrentInputBoundary().cuda();model.load_state_dict(torch.load(PRIOR/f'fold{role}_B/endpoint.pt',map_location='cpu',weights_only=True)['state'])
    initial=tensor_hash(model.state_dict());assert initial==spec['endpoint_parameter_sha256']
    base=tuple(p.detach().clone() for p in model.parameters());counter=Counter(model,spec['head_cap'],folder/'calls.jsonl',0)
    direction=np.load(ROOT/spec['qualified_direction']);cert=read(ROOT/spec['qualified_certificate']);assert cert['status']=='numeric_polish_certified_for_finite_probe_only' and direction.shape==(1060832,) and np.isfinite(direction).all() and float(np.abs(direction).max())==1.
    save(folder/'started.json',dict(role=role,initial_parameter_sha256=initial,run_seal_sha256=sha(OUT/'run_seal.json'),qualified_direction_sha256=sha(ROOT/spec['qualified_direction']),new_gradients=0,new_fits=0,permanent_updates=0))
    accepted=False;proposals=0;stop='execution_exception_stop';failure=None
    try:
        rv,qo,lo,_=risk(model,ctx,'OOF',ctx['ids']);qd,ld=probabilities(model,ctx,'deployment',np.arange(22546),True)
        np.save(folder/'baseline_risk.npy',rv);store_scope(folder/'baseline',ctx,'OOF',qo,lo);store_scope(folder/'baseline',ctx,'deployment',qd,ld)
        old_oo=np.load(DIAG/f'role{role}/baseline_class1/OOF_q.npy');old_de=np.load(DIAG/f'role{role}/baseline_deployment/deployment_q.npy')
        same=dict(OOF=repeat_values(qo[ctx['ids']],old_oo[ctx['ids']],'probability'),deployment=repeat_values(qd,old_de,'probability'),risk=repeat_values(rv,np.load(DIAG/f'role{role}/baseline_class1/risk.npy'),'risk'),OOF_logq=repeat_values(lo[ctx['ids']],np.load(DIAG/f'role{role}/baseline_class1/OOF_logq.npy')[ctx['ids']],'log_probability'),deployment_logq=repeat_values(ld,np.load(DIAG/f'role{role}/baseline_deployment/deployment_logq.npy'),'log_probability'))
        save(folder/'zero_step_repeat_review.json',same);assert all(r['passed'] for r in same.values())
        assert stats(ctx,qo,'OOF')['protected_regressions']==0 and stats(ctx,qd,'deployment')['mastered'] and joint_check(ctx,qd,ld)['passed']
        stop='fixed_polished_direction_finite_window_failed_stop'
        for j in range(20):
            target=folder/f'probe{j}';target.mkdir();proposals+=1
            result,_=probe(model,base,ctx,target,direction,2.**(-j),rv,cert['class_slopes'],counter)
            record_endpoint_progress(target,role)
            if result['accepted']:accepted=True;stop='polished_finite_probe_pass_not_fit';break
    except Exception as e:
        failure=dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());save(folder/'execution_failure.json',failure)
    finally:
        restore(model,base);restored=tensor_hash(model.state_dict());assert restored==initial
        restored_risk,rq,rl,_=risk(model,ctx,'OOF',ctx['ids']);rd,rdl=probabilities(model,ctx,'deployment',np.arange(22546),True)
        np.save(folder/'restored_risk.npy',restored_risk);store_scope(folder/'restored',ctx,'OOF',rq,rl);store_scope(folder/'restored',ctx,'deployment',rd,rdl)
        joint=joint_check(ctx,rd,rdl);assert joint['passed'];counts=counter.counts();counter.close()
        summary=dict(status=stop,role=role,finite_probe_pass=accepted,counts=counts,finite_proposals=proposals,new_class_gradients=0,new_margin_gradients=0,new_fits=0,permanent_updates=0,initial_parameter_sha256=initial,restored_parameter_sha256=restored,restored_joint_TRAIN_retention=joint,exception=failure,quality_acceptance=False)
        save(folder/'diagnostic.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    require()
    if failure:raise RuntimeError('Recorded execution failure; no automatic restart')

def record_endpoint_progress(folder,role):
    summary={}
    for scope in ['OOF','deployment']:
        r=pd.read_parquet(folder/f'{scope}_original_rows.parquet')
        parent=DIAG/f'role{role}'/('baseline_class1' if scope=='OOF' else 'baseline_deployment')
        b=pd.read_parquet(parent/f'{scope}_original_rows.parquet')
        assert np.array_equal(r[['row_position','truth']].to_numpy(),b[['row_position','truth']].to_numpy())
        old_wrong=b.pred.ne(b.truth).to_numpy();new_wrong=r.pred.ne(r.truth).to_numpy();pure=b.pure_current_input.to_numpy()
        truth=b.truth.to_numpy(np.int64);rival=b.pred.to_numpy(np.int64);index=np.arange(len(b))
        old_lp=b[['logp0','logp1','logp2']].to_numpy();new_lp=r[['logp0','logp1','logp2']].to_numpy()
        before=(old_lp[index,truth]-old_lp[index,rival])[old_wrong];after=(new_lp[index,truth]-new_lp[index,rival])[old_wrong];delta=after-before
        ledger=b.loc[old_wrong,['row_position','root','local','truth','pred','pure_current_input']].copy();ledger.rename(columns={'pred':'fixed_endpoint_rival'},inplace=True)
        ledger['baseline_truth_rival_margin']=before;ledger['candidate_truth_rival_margin']=after;ledger['margin_change']=delta
        ledger.to_parquet(folder/f'{scope}_endpoint_error_margin_progress.parquet',index=False)
        summary[scope]=dict(classification_changes_vs_fixed_endpoint=int(r.pred.ne(b.pred).sum()),repairs_vs_fixed_endpoint=int((old_wrong&~new_wrong).sum()),new_errors_vs_fixed_endpoint=int((~old_wrong&new_wrong).sum()),pure_repairs_vs_fixed_endpoint=int((old_wrong&~new_wrong&pure).sum()),pure_new_errors_vs_fixed_endpoint=int((~old_wrong&new_wrong&pure).sum()),old_error_rows=int(old_wrong.sum()),old_error_margin_improved_rows=int((delta>0).sum()),median_old_error_margin_before=float(np.median(before)) if len(before) else None,median_old_error_margin_change=float(np.median(delta)) if len(delta) else None,diagnostic_margin_counts_not_quality_acceptance=True)
    save(folder/'fixed_endpoint_progress.json',summary)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--role',type=int,choices=[1,2],required=True);args=ap.parse_args();run(args.role)
