"""Finalize the completed local experiment, preserving earlier frozen evidence."""
import json,subprocess,sys
from pathlib import Path
import run_v56 as t
import run_v54 as v
import v56_shared_control as s
ROOT=t.ROOT;OUT=ROOT/'evidence/2026-09-14/v56_delivery';ENTRY={'README.md','training/README.md','docs/TRAINING_PLAN.md'}

def main():
 assert not OUT.exists();OUT.mkdir(parents=True);c,r,x=t.load();sc,_,_=s.load();prior={}
 for version,name in [('v51','v51_review'),('v53','v53_delivery'),('v54','v54_delivery'),('v55','v55_delivery')]:
  count=0
  for p,h in v.read(ROOT/'evidence/2026-09-14'/name/'delivery.json')['bindings'].items():
   if p in ENTRY:continue
   assert v.sha(ROOT/p)==h,(version,p);count+=1
  prior[version]=count
 evidence=['v56_review','v56_verification','v56_diagnosis','v56_information_controls','v56_shared_review','v56_shared_verification'];checked=0
 for name in evidence:
  folder=ROOT/'evidence/2026-09-14'/name
  for p,h in v.read(folder/'receipt.json')['files'].items():assert v.sha(folder/p)==h;checked+=1
 fits=0
 for run in [t.RUN,s.RUN]:
  for protocol in c['protocols']:
   folder=run/protocol;receipt=v.read(folder/'complete.json')
   for p,h in receipt['bindings'].items():assert v.sha(folder/p)==h
   fits+=receipt.get('new_base_fits',0)+receipt.get('residual_fits',receipt.get('new_residual_fits',0))
 assert fits==38
 main=v.read(ROOT/'evidence/2026-09-14/v56_review/summary.json');shared=v.read(ROOT/'evidence/2026-09-14/v56_shared_review/summary.json');assert not main['eligible_next_stage'] and not shared['passed']
 for name in ['v56_verification','v56_shared_verification']:assert v.read(ROOT/'evidence/2026-09-14'/name/'verification.json')['all_checks_passed']
 tests=[]
 for name in ['test_v56.py','test_v56_shared.py']:
  z=subprocess.run([sys.executable,str(ROOT/'training'/name)],cwd=ROOT,text=True,capture_output=True,encoding='utf-8');assert z.returncode==0,z.stderr;tests.append({'source':name,'sha256':v.sha(ROOT/'training'/name),'returncode':z.returncode,'stdout':z.stdout,'stderr':z.stderr})
 v.save(OUT/'tests.json',tests);trainsha=v.sha(ROOT/'data/official/train.parquet');assert trainsha=='6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
 regressions=[]
 for protocol,dd in shared['protocols'].items():
  for view,z in dd['metrics'].items():
   base=main['protocols'][protocol]['models']['base'][view]['asa'];new=z['asa']
   for k,label in [(1,'malicious'),(2,'suspicious')]:
    change=new['class_errors_B_M_S'][k]-base['class_errors_B_M_S'][k]
    if change>0:regressions.append({'protocol':protocol,'view':view,'class':label,'base_errors':base['class_errors_B_M_S'][k],'new_errors':new['class_errors_B_M_S'][k],'increase':change})
 v.save(OUT/'final_review.json',{'actual_new_fits':fits,'new_base_fits':2,'new_residual_fits':36,'reused_base_models':2,'source_rows':len(r),'original_training_sha256':trainsha,'prior_files_unchanged':prior,'mutable_entries_excluded':sorted(ENTRY),'evidence_files_checked':checked,'posthoc_per_cell_regressions':regressions,
 'main_gates':main['gates'],'supplemental_gates':shared['gates'],'passed':False,'promoted':False,'original_pressure_evaluation_executed':False,'formal_submission_changed':False,'platform_used':False,'new_dependencies':False,'scope':'Completed 26-fit main study and 12-fit structural correction; real local gains, unresolved class tradeoffs and missing-condition regressions. No transfer acceptance. Posthoc per-cell regression audit supplements, not replaces, frozen gates.'})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());paths={ROOT/p for p in ENTRY};paths.add(ROOT/'docs/V56_SUPPORT_POOLING.md')
 for folder in [t.RUN,s.RUN,OUT]+[ROOT/'evidence/2026-09-14'/e for e in evidence]:paths.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 paths.update((ROOT/'training').glob('*v56*.py'));bindings={p.relative_to(ROOT).as_posix():v.sha(p) for p in sorted(paths)};v.save(OUT/'delivery.json',{'count':len(bindings),'bindings':bindings})
 for p,h in bindings.items():assert v.sha(ROOT/p)==h
 print(json.dumps({'files':len(bindings),'new_fits':fits,'prior_preserved':prior,'passed':False,'promoted':False}),flush=True)
if __name__=='__main__':main()
