"""Separate physical and commit failure cases; no seal or model execution."""
import json
from pathlib import Path
from experiment_review import ROOT,sha
from v169_pair_execution_review_v4 import check_memory

def main():
    out=ROOT/'artifacts/v169_physical_and_commit_policy_qualification_20261002';assert not out.exists()
    resources=dict(minimum_free_RAM_bytes=6*1024**3,minimum_free_commit_bytes=int(6.25*1024**3));check_memory(resources,resources['minimum_free_RAM_bytes'],resources['minimum_free_commit_bytes'])
    refused=[]
    for name,physical,commit in [('physical_below_boundary',resources['minimum_free_RAM_bytes']-1,10*1024**3),('commit_below_boundary',10*1024**3,resources['minimum_free_commit_bytes']-1)]:
        try:check_memory(resources,physical,commit)
        except RuntimeError:refused.append(name)
        else:raise AssertionError('Expected separate prerequisite refusal')
    from v169_prior_pair_training_entry_v13 import require
    try:require('initial')
    except RuntimeError as e:assert 'not sealed' in str(e)
    else:raise AssertionError('Unsealed entry must refuse before any model call')
    sources=[Path(__file__).resolve(),ROOT/'training/v169_pair_execution_review_v4.py',ROOT/'training/v169_prior_pair_training_entry_v13.py']
    report=dict(status='V169_separate_physical_RAM_and_available_commit_boundary_policy_qualified',physical_RAM_preserved6GiB=True,available_commit_minimum_bytes=resources['minimum_free_commit_bytes'],refused_cases=refused,exact_registered_boundaries_pass=True,unsealed_actual_entry_refused=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    out.mkdir();(out/'qualification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],official_calls=0)))

if __name__=='__main__':main()
