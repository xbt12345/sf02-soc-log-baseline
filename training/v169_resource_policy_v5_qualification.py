"""Exact resource boundaries, registration identity and unsealed refusal."""
import copy,json
from pathlib import Path
from unittest.mock import patch
from experiment_review import ROOT,read,sha,check_bindings
import v169_pair_execution_review_v5 as policy

OUT=ROOT/'artifacts/v169_resource_policy_v5_qualification_20261002'

def main():
    assert not OUT.exists()
    resources=policy.registered_resources();ram=resources['minimum_free_RAM_bytes'];commit=resources['minimum_free_commit_bytes']
    assert ram==5637144576 and commit==6710886400
    policy.check_memory(resources,ram,commit)
    refused=[]
    for name,physical,available_commit in [('physical_one_byte_short',ram-1,commit),('commit_one_byte_short',ram,commit-1)]:
        try:policy.check_memory(resources,physical,available_commit)
        except RuntimeError:refused.append(name)
        else:raise AssertionError(name)
    for name,key,value in [('unregistered_lower_RAM','minimum_free_RAM_bytes',ram-1),('unregistered_old_RAM','minimum_free_RAM_bytes',6*1024**3),('unregistered_commit','minimum_free_commit_bytes',commit-1),('unregistered_disk','minimum_free_disk_start_bytes',resources['minimum_free_disk_start_bytes']-1)]:
        bad={**resources,key:value}
        try:policy.check_memory(bad,10*1024**3,10*1024**3)
        except ValueError:refused.append(name)
        else:raise AssertionError(name)
    plan_path=ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v4.json';plan=read(plan_path)
    policy.review_plan(plan)
    actual_read=policy.read
    for name,change in [('resource_review_lineage_mismatch',dict(previous_resource_review_sha256='0'*64)),('resource_review_entry_mismatch',dict(entry_sha256='0'*64))]:
        def fake_read(path):
            value=actual_read(path)
            return {**value,**change} if Path(path).resolve()==(ROOT/policy.RESOURCE).resolve() else value
        with patch.object(policy,'read',fake_read):
            try:policy.review_plan(plan)
            except ValueError:refused.append(name)
            else:raise AssertionError(name)
    bad=copy.deepcopy(plan);bad['source_sha256'].pop(policy.RESOURCE)
    try:policy.review_plan(bad)
    except ValueError:refused.append('missing_bound_resource_review')
    else:raise AssertionError('missing_bound_resource_review')
    with patch.object(policy,'BASIS_SHA256','0'*64):
        try:policy.check_memory(resources,ram,commit)
        except ValueError:refused.append('immutable_resource_basis_SHA_mismatch')
        else:raise AssertionError('immutable_resource_basis_SHA_mismatch')
    from v169_prior_pair_training_entry_v14 import require
    try:require('initial')
    except RuntimeError as error:assert 'not sealed' in str(error)
    else:raise AssertionError('unsealed actual entry')
    sources=[Path(__file__).resolve(),ROOT/policy.ENTRY,ROOT/'training/v169_pair_execution_review_v5.py',ROOT/policy.BASIS,ROOT/policy.RESOURCE,plan_path]
    report=dict(status='V169_resource_policy_v5_exact_boundary_unregistered_threshold_review_identity_and_unsealed_refusals_passed',resources=resources,exact_registered_RAM_and_commit_equalities_pass=True,negative_cases_refused=refused,unsealed_actual_entry_refused_before_official_calls=True,original6GiB_evidence_preserved=True,scientific_numerical_thresholds_unchanged=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    check_bindings(report['source_sha256']);OUT.mkdir();(OUT/'qualification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],negative_cases=len(refused),official_calls=0)))

if __name__=='__main__':main()
