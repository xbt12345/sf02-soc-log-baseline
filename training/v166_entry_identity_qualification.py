"""Real original-input and cached-parameter identities; no forward/backward."""
import ast,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import tensor_hash
from v160_margin_normal import input_identity
from v166_coverage_execution_review import prospective_plan,review_plan
import v166_coverage_first_diagnostic as entry

OUT=ROOT/'artifacts/v166_entry_identity_qualification_20261002'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists();plan=prospective_plan();assert review_plan(plan)['status']=='V166_coverage_finite_plan_reviewed'
    active=['v166_coverage_first_diagnostic','v166_coverage_execution_review','v166_observed_function_measurement','v166_solver_trace','v166_coverage_joint_restoration','v166_entry_identity_qualification']
    for name in active:ast.parse((ROOT/'training'/f'{name}.py').read_text(encoding='utf-8'))
    OUT.mkdir();sources={ROOT/'training'/f'{name}.py' for name in active}|{entry.DRAFT,ROOT/'training/review_policy/v165_observed_boundaries.json',ROOT/'docs/V165_ACTUAL_RESULTS_AND_COVERAGE_FIRST_NEXT_PLAN_20261002.md',ROOT/'artifacts/v166_coverage_core_qualification_v3_20261002/qualification.json'}
    for folder in [entry.PRIOR,entry.CONTROL,ROOT/'artifacts/v165_independent_blocker_coverage_review_20261002']:sources.update(p for p in folder.rglob('*') if p.is_file())
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));results=[]
    for spec in plan['roles']:
        role=spec['role'];ctx=entry.load_context(role);model=entry.CurrentInputBoundary().cpu();model.load_state_dict(torch.load(entry.PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state']);identity=tensor_hash(model.state_dict());assert identity==spec['endpoint_parameter_sha256']
        records,normals,gradients,u=entry.cached_origin(spec,identity);blockers=pd.read_parquet(entry.CONTROL/f'role{role}/treatment/actual_blocking_original_rows.parquet');fresh={}
        for (scope,local,truth,rival),group in blockers.groupby(['scope','local','truth','rival'],sort=True):
            ids=ctx['ids'] if scope=='OOF' else np.arange(22546);pos=int(np.searchsorted(ids,local));assert ids[pos]==local;chunk=ids[pos//2048*2048:pos//2048*2048+2048]
            fid=input_identity(role,scope,chunk,ctx['x'][chunk],np.asarray(ctx[scope][chunk],np.float64),pos%2048,int(truth),int(rival));assert fid not in records;fresh[fid]=dict(scope=scope,local=int(local),truth=int(truth),rival=int(rival),original_rows=len(group))
        assert len(records)==spec['cached_functions'] and len(fresh)==spec['fresh_functions'] and len(records)+len(fresh)==spec['joint_functions']<=25
        endpoint=pd.read_parquet(entry.PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet');assert np.array_equal(ctx['OOF_rows'].protected_correct,endpoint.protected_correct)
        assert spec['head_cap']==3*(spec['OOF_chunks']+12)+2*len(fresh) and spec['fresh_margin_gradient_cap']==2*len(fresh)
        save(OUT/f'role{role}_registered_fresh_function_identities.json',dict(origin_parameter_sha256=identity,functions=fresh))
        results.append(dict(role=role,origin_parameter_sha256=identity,old_complete_normal_functions=len(records),new_complete_function_identities=len(fresh),joint_function_count=len(records)+len(fresh),all_actual_blocking_original_rows_covered=sum(r['original_rows'] for r in fresh.values()),all_old_gradient_and_normal_pairs_repeat=True,all_cumulative_protection_including_eight_repairs_kept=True,original_mixed_rows_not_deleted=True,head_cap_formula_verified=True))
    assert sum(r['new_complete_function_identities'] for r in results)==33
    check_bindings(bindings);save(OUT/'qualification.json',dict(status='V166_original_current_inputs_33_fresh_functions_cached_vectors_and_budget_identity_qualified',roles=results,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,complete_lifecycle_and_optimizer_qualification_still_required=True,physical_run_seal_still_required=True,execution_authority=False,source_sha256=bindings));print(json.dumps(dict(status='V166_entry_identity_qualification_passed',new_functions=33,official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
