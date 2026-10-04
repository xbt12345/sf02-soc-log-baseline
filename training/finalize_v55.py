"""Bind the completed fit-side experiment without claiming outer acceptance."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/v55_risk_validation_20260914';OUT=ROOT/'evidence/2026-09-14/v55_delivery'
sys.path.insert(0,str(RUN));import run_v55 as n;import run_v54 as v

def main():
 assert not OUT.exists();OUT.mkdir(parents=True);cfg,r,x=n.load(ROOT,RUN);entries={'README.md','docs/TRAINING_PLAN.md','training/README.md'};prior={}
 for version,file in [('v51','v51_review'),('v53','v53_delivery'),('v54','v54_delivery')]:
  count=0
  for p,h in v.read(ROOT/'evidence/2026-09-14'/file/'delivery.json')['bindings'].items():
   if p in entries:continue
   assert v.sha(ROOT/p)==h,(version,p);count+=1
  prior[version]=count
 assert v.sha(ROOT/'training/run_v55.py')==v.sha(RUN/'run_v55.py');done=0
 for split in n.SPLITS:
  for candidate in cfg['specs']:
   folder=RUN/(split+'_'+candidate)
   for p,h in v.read(folder/'complete.json')['bindings'].items():assert v.sha(folder/p)==h
   assert len(list(folder.glob('epoch_*.joblib')))==3;done+=1
 assert done==16
 supplement=ROOT/'artifacts/v55_calibration_control_r2_20260914';suppcfg=v.read(supplement/'configuration.json');assert v.sha(supplement/'configuration.json')==v.read(supplement/'preregistered.json')['sha256']
 for base,key in [(ROOT,'source_bindings'),(supplement,'local_bindings')]:
  for p,h in suppcfg[key].items():assert v.sha(base/p)==h
 for split in n.SPLITS:
  for p,h in v.read(supplement/split/'complete.json')['bindings'].items():assert v.sha(supplement/split/p)==h
 evidence=['v55_review','v55_verification','v55_diagnosis','v55_calibration_verification'];checked=0
 for name in evidence:
  folder=ROOT/'evidence/2026-09-14'/name
  for p,h in v.read(folder/'receipt.json')['files'].items():assert v.sha(folder/p)==h;checked+=1
 review=v.read(ROOT/'evidence/2026-09-14/v55_review/summary.json');verification=v.read(ROOT/'evidence/2026-09-14/v55_verification/verification.json')
 assert review['outer_candidate'] is None,'An eligible candidate needs outer execution before finalizing.'
 assert verification['all_checks_passed'] and len(verification['runs'])==16
 calibration=v.read(ROOT/'evidence/2026-09-14/v55_calibration_verification/verification.json');assert calibration['all_checks_passed'] and calibration['actual_fits']==2
 tests=subprocess.run([sys.executable,str(ROOT/'training/test_v55.py')],capture_output=True,text=True,encoding='utf-8',cwd=ROOT);assert tests.returncode==0,tests.stderr
 v.save(OUT/'test_execution.json',{'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr,'source_sha256':v.sha(ROOT/'training/test_v55.py')})
 originalsha=v.sha(ROOT/'data/official/train.parquet');assert originalsha=='6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
 v.save(OUT/'final_review.json',{'fit_side_experiment_completed':True,'actual_training_runs':done+2,'main_training_runs':done,'main_checkpoints':48,'supplemental_training_runs':2,'supplemental_final_models':2,'source_rows':len(r),'prior_frozen_files_unchanged':prior,'excluded_mutable_entries':sorted(entries),
  'original_training_sha256':originalsha,'verification':verification,'calibration_verification':calibration,'evidence_files_checked':checked,'inner_gate_passed':False,'outer_models_fit':0,'outer_quality_evaluation_executed':False,'promoted':False,'ASA_solved':False,'platform_used':False,'new_dependencies':False,
  'scope':'Preregistered continuation gate rejected every new candidate. Completed fit-side objective and validation experiment plus two post-hoc separated-calibration controls; no claim about improved outer quality. Supplemental hypothesis does not override the main gate. Both chosen thresholds increase full-input errors versus their same-model argmax controls.'})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());paths={ROOT/p for p in entries};paths.add(ROOT/'docs/V55_RISK_VALIDATION.md')
 for folder in [RUN,supplement,ROOT/'artifacts/v55_calibration_control_20260914']+[ROOT/'evidence/2026-09-14'/e for e in evidence]+[OUT]:paths.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 paths.update((ROOT/'training').glob('*v55*.py'));bindings={p.relative_to(ROOT).as_posix():v.sha(p) for p in sorted(paths)};v.save(OUT/'delivery.json',{'count':len(bindings),'bindings':bindings})
 for p,h in v.read(OUT/'delivery.json')['bindings'].items():assert v.sha(ROOT/p)==h
 print(json.dumps({'files':len(bindings),'completed_runs':done+2,'main_runs':done,'supplemental_runs':2,'prior_preserved':prior,'inner_gate_passed':False,'outer_fits':0}),flush=True)
if __name__=='__main__':main()
