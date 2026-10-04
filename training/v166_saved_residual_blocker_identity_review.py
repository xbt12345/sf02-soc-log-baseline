"""Saved actual residual blocker identities; no derivative or head calls."""
import json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from experiment_review import ROOT,read,sha,check_bindings
from v160_margin_normal import input_identity

TRIAL=ROOT/'artifacts/v166_coverage_first_diagnostic_20261002'
OUT=ROOT/'artifacts/v166_saved_residual_blocker_identity_review_20261002'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/diagnostic.json').exists() for r in range(3));OUT.mkdir();seal=read(TRIAL/'run_seal.json');check_bindings(seal['source_sha256']);plan=read(ROOT/seal['plan_path'])
    xfile=ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz';goldfile=ROOT/'data/official/train.parquet';paths={Path(__file__).resolve(),ROOT/'training/v160_margin_normal.py',xfile,goldfile}|{p for p in TRIAL.rglob('*') if p.is_file()}
    for role in range(3):paths.update(ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{scope}_probabilities.npy' for scope in ['OOF','deployment'])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));x=load_npz(xfile).tocsr();x.sort_indices();gold=pd.read_parquet(goldfile,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();results=[]
    for spec in plan['roles']:
        role=spec['role'];folder=TRIAL/f'role{role}';diag=read(folder/'diagnostic.json');refs=read(folder/'joint_restoration/active_normal_references.json');baseline=pd.read_parquet(folder/'baseline/OOF_original_rows.parquet');ids_oof=np.sort(baseline.local.unique());known_margins=[]
        if diag['candidate'] is None:results.append(dict(role=role,actual_finite_candidate_present=False,measured_functions=len(refs),not_global_infeasibility_claim=True));continue
        for identity,ref in refs.items():
            meta=ref['metadata'];logs=np.load(folder/f"treatment/{meta['scope']}_logq.npy");values=np.load(folder/f"treatment/{meta['scope']}_q.npy");local=meta['local'];known_margins.append(dict(identity=identity,scope=meta['scope'],local=local,truth=meta['truth'],rival=meta['rival'],margin=float(logs[local,meta['truth']]-logs[local,meta['rival']]),argmax_correct=bool(values[local].argmax()==meta['truth'])))
        blockers=pd.read_parquet(folder/'treatment/actual_blocking_original_rows.parquet');assert np.array_equal(blockers.truth,gold[blockers.row_position]);functions={};opinions={scope:np.load(ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{scope}_probabilities.npy',mmap_mode='r')[:,:16] for scope in ['OOF','deployment']}
        for (scope,local,truth,rival),group in blockers.groupby(['scope','local','truth','rival'],sort=True):
            ids=ids_oof if scope=='OOF' else np.arange(22546);pos=int(np.searchsorted(ids,local));assert ids[pos]==local;chunk=ids[pos//2048*2048:pos//2048*2048+2048];identity=input_identity(role,scope,chunk,x[chunk],np.asarray(opinions[scope][chunk],np.float64),pos%2048,int(truth),int(rival));q=np.load(folder/f'treatment/{scope}_q.npy');lp=np.load(folder/f'treatment/{scope}_logq.npy');assert q[local].argmax()!=truth
            functions[identity]=dict(scope=scope,local=int(local),truth=int(truth),rival=int(rival),original_rows=len(group),known_measured=identity in refs,actual_bad_margin=float(lp[local,truth]-lp[local,rival]),row_positions=group.row_position.astype(int).tolist(),roots=group.root.astype(str).unique().tolist())
        fresh=set(functions)-set(refs);known_bad=set(functions)&set(refs);new_union=len(set(refs)|set(functions));item=dict(role=role,actual_finite_candidate_present=True,actual_candidate_accepted=diag['candidate']['accepted'],blocking_original_rows=len(blockers),blocking_functions=len(functions),previously_measured_blocking_functions=len(known_bad),fresh_unmeasured_blocking_functions=len(fresh),measured_functions=len(refs),observed_union_if_added=new_union,registered_capacity=25,observed_union_exceeds_registered_capacity=new_union>25,all_measured_functions_actual_argmax_correct=all(v['argmax_correct'] for v in known_margins),all_measured_margins=known_margins,blocking_function_records=functions,fresh_input_identities=sorted(fresh),known_bad_input_identities=sorted(known_bad),no_derivatives_measured_here=True,no_new_capacity_or_execution_permission=True,no_global_infeasibility_or_sufficient_cause_claim=True);save(OUT/f'role{role}.json',item);results.append(item)
    check_bindings(bindings);save(OUT/'review.json',dict(status='V166_saved_actual_residual_blockers_complete_input_identities_verified',roles=results,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,source_sha256=bindings));print(json.dumps(dict(status='V166_saved_blocker_identity_review_passed',roles=[{k:r.get(k) for k in ['role','blocking_original_rows','previously_measured_blocking_functions','fresh_unmeasured_blocking_functions','observed_union_if_added','all_measured_functions_actual_argmax_correct']} for r in results],official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
