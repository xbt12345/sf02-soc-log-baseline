"""Hard wall-clock watchdog for the nonpromotable, TRAIN-only diagnostics."""
import os
import subprocess
import sys
import time
from v138_runtime import ROOT,OUT,read,save,sha,require_run_seal


def main():
    require_run_seal(ROOT/'training/v138_train.py')
    target=OUT/'probe_watchdog_receipt.json'
    if target.exists():raise FileExistsError(target)
    # The watchdog counts module load and source checks inside its 600s limit.
    results=[]
    for f in range(3):
        r=read(OUT/f'fold{f}_H_L/fit.json')
        if r['endpoint_stats']['pure_M_errors']+r['endpoint_stats']['pure_S_errors']==0:
            results.append({'fold':f,'executed':False,'reason':'No pure endpoint error'});continue
        folder=OUT/f'probe_fold{f}'
        log=OUT/f'probe_fold{f}_console.log'
        if folder.exists() or log.exists():raise FileExistsError('No repeated diagnostic solver')
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
        start=time.monotonic()
        with log.open('w',encoding='utf-8') as stream:
            try:
                run=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'training/v138_feasibility.py'),'--fold',str(f)],cwd=ROOT,
                    env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                code=run.returncode;timedout=False
            except subprocess.TimeoutExpired:code=None;timedout=True
        elapsed=time.monotonic()-start
        if timedout or code!=0:
            folder.mkdir(exist_ok=True)
            if not (folder/'receipt.json').exists():
                save(folder/'receipt.json',{'status':'supervised_diagnostic_incomplete','fold':f,'certificate':False,'round2_eligible':False,
                    'termination':'hard_watchdog_time_limit' if timedout else 'technical_failure','seconds_including_imports':elapsed,
                    'HELD_labels_used':0,'candidate_promotable':False,'limits':'No full-population certificate. Not proof of impossibility.'})
        results.append({'fold':f,'executed':True,'exit_code':code,'hard_watchdog_timeout':timedout,'seconds':elapsed,'receipt_sha256':sha(folder/'receipt.json')})
        save(OUT/'probe_watchdog_progress.json',results)
        print(results[-1],flush=True)
    save(target,{'status':'conditional_diagnostics_finished','results':results,'primary_fits_added':0,'optimizer_updates_added':0,
        'supervised_diagnostic_folds':sum(x['executed'] for x in results),'source_sha256':sha(__file__)})


if __name__=='__main__':main()
