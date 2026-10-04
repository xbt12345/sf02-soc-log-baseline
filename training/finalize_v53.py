"""Bind the delivered ASA study and confirm prior frozen results stayed unchanged."""
import json,subprocess,sys
from pathlib import Path
from verify_v53_asa import read,sha
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/2026-09-14/v53_delivery'
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 entries={'README.md','docs/TRAINING_PLAN.md','training/README.md'};prior={}
 for version,path in [('v51','evidence/2026-09-14/v51_review/delivery.json'),('v52','evidence/2026-09-14/v52_research/delivery.json')]:
  checked=0
  for p,h in read(ROOT/path)['bindings'].items():
   if p in entries:continue
   assert sha(ROOT/p)==h,(version,p);checked+=1
  prior[version]={'frozen_files_unchanged':checked,'excluded_mutable_entries':sorted(entries)}
 evidence=['v53_raw_audit','v53_review','v53_joint_verification','v53_findings'];bound=0
 for name in evidence:
  folder=ROOT/'evidence/2026-09-14'/name
  for p,h in read(folder/'receipt.json')['files'].items():assert sha(folder/p)==h,(name,p);bound+=1
 run=ROOT/'artifacts/v53_asa_factorial_r2_20260914';c=read(run/'configuration.json')
 assert sha(ROOT/'training/run_v53_asa.py')==sha(run/'run_v53_asa.py')
 for p,h in c['local_bindings'].items():assert sha(run/p)==h,p
 for p in c['protocols']:
  for f,h in read(run/p/'complete.json')['bindings'].items():assert sha(run/p/f)==h,(p,f)
 joint=ROOT/'artifacts/v53_complete_joint_20260914'
 assert sha(ROOT/'training/probe_v53_joint.py')==sha(joint/'probe_v53_joint.py')
 for p in c['protocols']:
  for f,h in read(joint/p/'complete.json')['files'].items():assert sha(joint/p/f)==h,(p,f)
 tests=subprocess.run([sys.executable,str(ROOT/'training/test_v53_inputs.py')],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
 assert tests.returncode==0,tests.stdout+tests.stderr
 save('test_execution.json',{'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr,'test_source_sha256':sha(ROOT/'training/test_v53_inputs.py'),'frozen_runner_matches_working_source':True})
 facts=read(ROOT/'evidence/2026-09-14/v53_findings/summary.json');gates=read(ROOT/'evidence/2026-09-14/v53_review/summary.json')['continuation_gates']
 assert not any(v['all_passed'] for v in gates.values());assert not any(v['all_passed'] for v in facts['joint_continuation_gates'].values())
 save('final_review.json',{'evidence_verified':True,'prior_frozen_results':prior,'evidence_files_checked':bound,
  'completed_EBM_models':16,'R2_new_EBM_fits':14,'R1_completed_controls_reused':2,'incomplete_dense_interaction_attempts':2,'fit_side_joint_count_models':4,
  'all_candidate_quality_gates_passed':False,'production_promoted':False,'ASA_solved':False,'official_training_data_only':True,'platform_used':False,
  'tests':'5 input/coarsening/label-mass tests plus independent original-file, learned-table, count, split and raw-variant verification',
  'new_dependency':'interpret-core 0.7.8; installed without updating existing dependencies',
  'supported_progress':'N_PAIR repaired 1168 of the 1532 seen-pure-same errors, with 0 new errors in that diagnostic slice. Overall candidate gates still fail.',
  'remaining':'Conflicting reliable observations; sparse and unseen combinations; class regression and pressure suspicious collapse; insufficient normal network counterexamples.',
  'future_not_executed':'Hierarchical support shrinkage and fit-side inner selection of complete/partial information parameters.'})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 paths=set(ROOT/p for p in entries);paths.add(ROOT/'docs/V53_ASA_EXECUTION.md')
 for folder in [ROOT/'artifacts/v53_asa_factorial_20260914',run,joint]+[ROOT/'evidence/2026-09-14'/n for n in evidence]+[OUT]:
  paths.update(p for p in folder.rglob('*') if p.is_file())
 paths.update((ROOT/'training').glob('*v53*.py'))
 bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
 save('delivery.json',{'count':len(bindings),'bindings':bindings,'scope':'Immutable study files and current entry documents as of delivery; no quality or transfer guarantee.'})
 for p,h in read(OUT/'delivery.json')['bindings'].items():assert sha(ROOT/p)==h,p
 print(json.dumps({'delivered_files':len(bindings),'prior_frozen_results':prior,'new_candidate_promoted':False,'ASA_solved':False}),flush=True)
if __name__=='__main__':main()
