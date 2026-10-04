"""Verify frozen study identity and record actual completion and failure boundaries."""
import json,subprocess,sys
from pathlib import Path
import joblib
import run_v54 as v
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/v54_method_change_20260914'
OUT=ROOT/'evidence/2026-09-14/v54_delivery'

def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 entries={'README.md','docs/TRAINING_PLAN.md','training/README.md'};prior={}
 for ver,path in [('v51','evidence/2026-09-14/v51_review/delivery.json'),('v52','evidence/2026-09-14/v52_research/delivery.json'),('v53','evidence/2026-09-14/v53_delivery/delivery.json')]:
  count=0
  for p,h in v.read(ROOT/path)['bindings'].items():
   if p in entries:continue
   assert v.sha(ROOT/p)==h,(ver,p);count+=1
  prior[ver]={'frozen_files_unchanged':count,'excluded_mutable_entries':sorted(entries)}
 cfg=v.read(RUN/'configuration.json');assert v.sha(ROOT/'training/run_v54.py')==v.sha(RUN/'run_v54.py')
 assert v.sha(RUN/'configuration.json')==v.read(RUN/'preregistered.json')['configuration_sha256']
 for p,h in cfg['source_bindings'].items():assert v.sha(ROOT/p)==h,p
 for p,h in cfg['local_bindings'].items():assert v.sha(RUN/p)==h,p
 source_train=v.sha(ROOT/'data/official/train.parquet');assert source_train=='6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
 svc=0;mlp=0;completed={}
 for protocol in v.PROTOCOLS:
  completed[protocol]={}
  for family in ['kernel','neural']:
   folder=RUN/(protocol+'_'+family)
   for receipt in ['fit_complete.json','evaluation_complete.json']:
    for p,h in v.read(folder/receipt)['bindings'].items():assert v.sha(folder/p)==h,(protocol,p)
   inners=list(folder.glob('inner_*.joblib'));finals=list(folder.glob('final_*.joblib'))
   assert len(inners)==3
   if family=='kernel':svc+=len(inners)+len(finals)
   else:
    assert len(finals)==1;bundle=joblib.load(finals[0]);assert len(bundle['models'])==2
    assert [m.random_state for m in bundle['models']]==cfg['neural_final_seeds']
    mlp+=1+len(bundle['models'])
   completed[protocol][family]={'fit_receipt_verified':True,'evaluation_receipt_verified':True,'selection':v.read(folder/'selection.json')}
 assert svc==19 and mlp==12
 evidence=['v54_cold_groups','v54_verification_kernel','v54_verification_neural','v54_review','v54_failure_controls'];checked=0
 for name in evidence:
  folder=ROOT/'evidence/2026-09-14'/name
  for p,h in v.read(folder/'receipt.json')['files'].items():assert v.sha(folder/p)==h,(name,p);checked+=1
 tests=subprocess.run([sys.executable,str(ROOT/'training/test_v54.py')],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
 assert tests.returncode==0,tests.stdout+tests.stderr
 v.save(OUT/'test_execution.json',{'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr,'source_sha256':v.sha(ROOT/'training/test_v54.py')})
 summary=v.read(ROOT/'evidence/2026-09-14/v54_review/summary.json');assert not any(g['all_passed'] for g in summary['gates'].values())
 verification={family:v.read(ROOT/'evidence/2026-09-14'/('v54_verification_'+family)/'verification.json') for family in ['kernel','neural']}
 assert all(q['all_checks_passed'] for q in verification.values())
 raw=v.read(ROOT/'evidence/2026-09-14/v54_review/raw_inference_verification.json');assert raw['all_checks_passed'] and len(raw['models'])==11
 v.save(OUT/'final_review.json',{'study_execution_and_identity_verified':True,'prior_frozen_results':prior,
  'source_training_sha256':source_train,'completed_SVC_estimators':svc,'completed_MLP_optimization_runs':mlp,'inner_MLP_checkpoints_are_not_separate_fits':True,
  'protocol_completion':completed,'evidence_files_checked':checked,'score_rows_independently_replayed':sum(q['score_rows_independently_replayed'] for q in verification.values()),
  'raw_original_cases_per_final_bundle':706,'raw_final_bundles_checked':11,'new_candidate_quality_passed':False,'production_promoted':False,'ASA_solved':False,
  'official_training_only':True,'new_dependencies_installed':False,'platform_used':False,
  'execution_note':'After independent fold1/fold2 completion, the old serial dispatcher reached existing fold1 and was rejected before any new fit/write. All eight protocol-family fits and evaluations are complete. Some workers used 2 and others 4 BLAS threads; model/data settings unchanged.',
  'cleanup_note':'Deletion of this run __pycache__ was blocked by automatic approval policy; cache retained and excluded from study bindings. No alternative deletion attempted.',
  'future_not_executed':'Group DRO and stronger-regularization factorial with behavior-group inner validation; no claim that it can resolve identical-observation label conflicts.'})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 paths={ROOT/p for p in entries};paths.add(ROOT/'docs/V54_METHOD_CHANGE.md')
 for folder in [RUN]+[ROOT/'evidence/2026-09-14'/n for n in evidence]+[OUT]:
  paths.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 paths.update((ROOT/'training').glob('*v54*.py'))
 bindings={p.relative_to(ROOT).as_posix():v.sha(p) for p in sorted(paths)}
 v.save(OUT/'delivery.json',{'count':len(bindings),'bindings':bindings,'scope':'Completed research files and current entry documents; no production or transfer acceptance.'})
 for p,h in v.read(OUT/'delivery.json')['bindings'].items():assert v.sha(ROOT/p)==h,p
 print(json.dumps({'delivered_files':len(bindings),'prior_frozen_results':prior,'SVC_fits':svc,'MLP_runs':mlp,'quality_passed':False,'ASA_solved':False}),flush=True)
if __name__=='__main__':main()
