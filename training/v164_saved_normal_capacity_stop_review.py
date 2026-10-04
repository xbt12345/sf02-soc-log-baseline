"""Original saved blocker groups explain capacity stops, without new calls."""
import json
from pathlib import Path
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings

TRIAL=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
OUT=ROOT/'artifacts/v164_saved_normal_capacity_stop_review_20261002'

def save(path,value):
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists(); OUT.mkdir(); sources={Path(__file__).resolve(),ROOT/'training/v164_short_supervised_trajectory.py',ROOT/'training/v160_fixed_endpoint_diagnostic_v3.py'}; inputs=[]
    for role in range(3):
        folder=TRIAL/f'role{role}'; fit=read(folder/'fit.json'); assert fit['status']=='margin_normal_cap_stop'
        point=folder/f"parameter_point{fit['permanent_updates']}"
        proposals=list(point.glob('restoration*/finite_probe/probe.json'))
        last=max(proposals,key=lambda p:int(p.parent.parent.name.removeprefix('restoration'))).parent
        blockerpath=last/'actual_blocking_original_rows.parquet'
        metas=list((point/'normals').glob('*/input_binding.json'))
        sources.update([folder/'fit.json',point/'parameter_identity.json',last/'probe.json',blockerpath,*metas]); inputs.append((role,point,last,blockerpath,metas))
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0)); results=[]
    for role,point,last,blockerpath,metas in inputs:
        identity=read(point/'parameter_identity.json')['parameter_sha256']; records=[read(p) for p in metas]
        assert all(r['base_parameter_sha256']==identity for r in records)
        known={(r['scope'],r['local'],r['truth'],r['rival']) for r in records}; assert len(known)==len(records)
        frame=pd.read_parquet(blockerpath); functions={tuple(values) for values in frame[['scope','local','truth','rival']].drop_duplicates().itertuples(index=False,name=None)}
        pending=functions-known; union=known|functions; assert len(union)>24
        report=read(last/'probe.json'); assert not report['accepted']
        classes={str(int(cls)):int((frame.truth==cls).sum()) for cls in sorted(frame.truth.unique())}
        result=dict(role=role,last_parameter_point_sha256=identity,last_rejected_probe=last.relative_to(ROOT).as_posix(),actual_blocking_original_rows=len(frame),blocking_original_class_counts=classes,measured_function_count=len(known),last_probe_blocking_functions=len(functions),already_measured_current_blocking_functions=len(functions&known),pending_unmeasured_functions=len(pending),required_function_union=len(union),registered_capacity=24,measured_plus_pending_exceeds_registered_capacity=True,pending_derivatives_not_executed=True,scope_local_truth_rival_tuple_is_bijective_function_key_at_fixed_role_input_and_chunk=True,not_a_direction_infeasibility_or_capacity_impossibility_proof=True,not_candidate_acceptance_or_classification_mastery=True)
        save(OUT/f'role{role}.json',result);results.append(result)
    check_bindings(bindings); save(OUT/'review.json',dict(status='all_three_actual_stops_are_registered_normal_union_capacity_truncations',roles=results,official_calls=0,parameter_updates=0,source_sha256=bindings)); print(json.dumps(dict(status='V164_saved_capacity_stop_review_passed',roles=results),ensure_ascii=False))

if __name__=='__main__':main()
