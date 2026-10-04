"""Cloud execution, resumable completed fits and a single results archive."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def execute(arguments,log,env):
    print('STAGE: '+arguments[2],flush=True)
    process=subprocess.Popen(arguments,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,encoding='utf-8',errors='replace',env=env,cwd=str(Path(__file__).parent))
    messages=queue.Queue()
    def read():
        for line in process.stdout:messages.put(line)
        messages.put(None)
    threading.Thread(target=read,daemon=True).start()
    try:
        while True:
            try:line=messages.get(timeout=20)
            except queue.Empty:
                line='RUNNING: current stage is still working; no result yet.\n'
            if line is None:break
            print(line,end='',flush=True);log.write(line);log.flush()
        if process.wait()!=0:raise RuntimeError('Stage failed: '+arguments[1])
    except BaseException:
        if process.poll() is None:
            process.terminate();process.wait()
        raise


def pack(root,work,failed):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=work/(('v37_failure_' if failed else 'v37_results_')+stamp+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(root.glob('*')):
            if path.is_file() and path.suffix in ('.py','.json','.md'):z.write(path,'runtime/'+path.name)
        # Full source and model/prediction evidence; large reproducible prepared
        # corpus remains on platform, its byte identity is included in result.
        for path in sorted(work.rglob('*')):
            if not path.is_file() or path.suffix=='.zip' or 'prepared.parquet'==path.name:
                continue
            z.write(path,'work/'+str(path.relative_to(work)).replace('\\','/'))
    print(('RETURN_FAILURE_ZIP: ' if failed else 'RETURN_THIS_ZIP: ')+str(archive),flush=True)
    print('ZIP_SHA256: '+sha(archive),flush=True)
    return archive


def main(args):
    root=Path(__file__).resolve().parent;work=root/'work';work.mkdir(exist_ok=True)
    train=Path(args.train).resolve()
    if not train.is_file() or sha(train)!='6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742':
        raise ValueError('Official train missing or damaged: '+str(train))
    lock=work/'running.lock'
    # OS file lock is released by process death; no stale-lock deletion needed.
    held=lock.open('a+')
    if os.name=='posix':
        import fcntl
        try:fcntl.flock(held.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This bundle is already running')
    env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads))
    config=json.loads((root/'v37_batch.json').read_text(encoding='utf-8'))
    from v37_contract import CONTRACT
    if config!=CONTRACT:raise ValueError('Configuration differs from frozen contract')
    failed=False
    with (work/'execution.log').open('a',encoding='utf-8') as log:
        try:
            execute([sys.executable,'-m','unittest','discover','-s',str(root),'-p','test_v37*.py'],log,env)
            print('SOFTWARE_CHECKS_ONLY: model quality has not been established.',flush=True)
            candidates=sorted(work.glob('prepared_attempt*'))
            completed=[p for p in candidates if (p/'result.json').exists()]
            if len(completed)>1:raise ValueError('Ambiguous completed preparation')
            if completed:
                prepared=completed[0]
            else:
                prepared=work/('prepared_attempt'+str(len(candidates)+1).zfill(3))
                command=[sys.executable,'-u',str(root/'run_v37_prepare.py'),'--train',str(train),
                    '--output-dir',str(prepared),'--workers',str(args.workers)]
                if args.previous:command+=['--previous',str(Path(args.previous).resolve())]
                execute(command,log,env)
            audit=work/'independent_audit.json'
            if audit.exists():
                try:json.loads(audit.read_text(encoding='utf-8'))
                except json.JSONDecodeError:audit.rename(work/('incomplete_audit_'+str(time.time_ns())+'.json'))
            if not audit.exists():
                execute([sys.executable,'-u',str(root/'audit_v37_prepared.py'),'--train',str(train),
                         '--prepared',str(prepared),'--output',str(audit)],log,env)
            else:
                value=json.loads(audit.read_text(encoding='utf-8'))
                if not value['all_checks_passed']:raise ValueError('Existing audit failed')
                for filename,key in [('prepared.parquet','prepared_sha256'),('groups.parquet','groups_sha256'),('protocol.parquet','protocol_sha256')]:
                    if sha(prepared/filename)!=value[key]:raise ValueError('Prepared audit identity changed')
                if sha(work/'audit_cases.json')!=value['audit_cases_sha256']:raise ValueError('Raw audit cases changed')
            reference=root/'local_reference.json'
            if not reference.exists():raise ValueError('Required local input gate reference missing')
            if reference.exists():
                expected=json.loads(reference.read_text(encoding='utf-8'));actual=json.loads(audit.read_text(encoding='utf-8'))
                for key in ['canonical_input_views_sha256','canonical_groups_sha256','canonical_roles_sha256']:
                    if actual[key]!=expected[key]:raise ValueError('Local/cloud prepared content differs: '+key)
            results=work/'models'
            execute([sys.executable,'-u',str(root/'run_v37_train.py'),'--prepared',str(prepared),
                     '--output-dir',str(results),'--audit',str(audit),'--tasks',','.join(config['tasks']),'--views',','.join(config['views'])],log,env)
            replay=work/'model_replay.json'
            if replay.exists():
                replay=work/('model_replay_'+str(time.time_ns())+'.json')
            if not replay.exists():
                execute([sys.executable,'-u',str(root/'review_v37_round.py'),'--prepared',str(prepared),'--results',str(results),
                         '--output',str(replay),'--cases',str(work/'audit_cases.json'),'--train',str(train)],log,env)
            replay_data=json.loads(replay.read_text(encoding='utf-8'))
            expected=len(config['tasks'])*len(config['views'])
            if not replay_data['all_checks_passed'] or len(replay_data['models'])!=expected:
                raise ValueError('Model replay incomplete or failed')
            (work/'run_complete.json').write_text(json.dumps({'models':expected,'fits':expected*3,
                'replay_sha256':sha(replay),'official_input_sha256':sha(train),
                'model_quality_accepted':False,'blind_or_external_test':False,
                'stage_D':'await_C_review; no target-adaptive automatic training'},indent=2),encoding='utf-8')
        except BaseException:
            failed=True
            log.write(traceback.format_exc());log.flush()
            print(traceback.format_exc(),flush=True)
    if not failed:print('BATCH_EXECUTION_COMPLETE; quality requires reviewing the results.',flush=True)
    pack(root,work,failed)
    held.close()
    if failed:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True);p.add_argument('--previous')
    p.add_argument('--threads',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if not (1<=args.threads<=16 and 1<=args.workers<=4):raise ValueError('Resource limits: threads 1..16, workers 1..4')
    main(args)
