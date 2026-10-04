"""Register actual V169 only after independent source/resource preseal review."""
import argparse,importlib.metadata,json,shutil,sys
from pathlib import Path
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v169_pair_execution_review_v3 import review_plan
from v169_pair_execution_review_v2 import MemoryStatus
import ctypes
import v169_prior_pair_training_entry_v12 as entry

BUNDLE=ROOT/'artifacts/v169_full_preseal_bundle_v2_20261002/bundle.json'
CANDIDATE=ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v2.json'

def main(root_review_path):
    assert not entry.OUT.exists() and not entry.PLAN.exists()
    root_path=Path(root_review_path).resolve(strict=True);assert root_path.is_relative_to(ROOT/'artifacts');root=read(root_path)
    assert root['supports_physical_seal'] is True and root['execution_authority'] is False and root['reviewed_entry_sha256']==sha(Path(entry.__file__))
    assert all(root[k]==0 for k in ['official_heads','official_features','official_derivatives','fits','permanent_updates']);check_bindings(root['source_sha256'])
    bundle=read(BUNDLE);check_bindings(bundle['source_sha256']);assert bundle['entry_sha256']==root['reviewed_entry_sha256'] and bundle['candidate_contract_sha256']==sha(CANDIDATE)
    plan=read(CANDIDATE);review_plan(plan);assert not plan['execution_authority'] and not plan['new_fit_permission']
    state=MemoryStatus();state.length=ctypes.sizeof(state);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state))
    free=shutil.disk_usage(ROOT).free;gpu_free,gpu_total=torch.cuda.mem_get_info()
    assert free>=plan['resources']['minimum_free_disk_start_bytes'] and state.avail_phys>=plan['resources']['minimum_free_RAM_bytes'] and gpu_free>=plan['resources']['minimum_free_GPU_bytes']
    bindings=dict(bundle['source_sha256']);bindings.update(root['source_sha256'])
    for path in [Path(__file__).resolve(),BUNDLE,CANDIDATE,root_path]:bindings[path.relative_to(ROOT).as_posix()]=sha(path)
    plan.update(status='V169_new_paired_training_contract_registered_before_official_calls',execution_authority=True,new_fit_permission=True,independent_preseal_review=root_path.relative_to(ROOT).as_posix(),independent_preseal_review_sha256=sha(root_path),dependency_bundle=BUNDLE.relative_to(ROOT).as_posix(),dependency_bundle_sha256=sha(BUNDLE),preseal_resource_snapshot=dict(free_disk_bytes=free,free_RAM_bytes=state.avail_phys,free_GPU_bytes=gpu_free,total_GPU_bytes=gpu_total))
    entry.OUT.mkdir();entry.PLAN.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');bindings[entry.PLAN.relative_to(ROOT).as_posix()]=sha(entry.PLAN)
    seal=dict(status='V169_new_physical_seal_before_official_training',protocol=entry.PROTOCOL,trainer_path=Path(entry.__file__).resolve().relative_to(ROOT).as_posix(),plan_path=entry.PLAN.relative_to(ROOT).as_posix(),plan_sha256=sha(entry.PLAN),root_review_path=root_path.relative_to(ROOT).as_posix(),root_review_sha256=sha(root_path),execution_dependency_closure_complete_reviewed=True,python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings)
    (entry.OUT/'run_seal.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');entry.require('initial')
    entry.save(entry.OUT/'registration.json',dict(status='V169_physically_sealed_before_official_calls',physical_sources=len(bindings),entry_sha256=sha(Path(entry.__file__)),new_caps=plan['new_caps'],prior_actual_costs=plan['prior_actual_costs'],resources=plan['resources'],official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,run_seal_sha256=sha(entry.OUT/'run_seal.json')))
    print(json.dumps(dict(status='V169_prior_pair_physically_registered',physical_sources=len(bindings),official_calls=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root-review',required=True);main(parser.parse_args().root_review)
