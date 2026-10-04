"""Download bounded public source snapshots for review; do not import them."""
import hashlib
import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/2026-09-13/v40_methods'


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SF02-source-review/1.0'}),timeout=25) as r:
        return r.read()


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'source_receipts.json').exists()
    specs=[('scikit-learn/scikit-learn','refs/tags/1.3.2',['COPYING','sklearn/feature_extraction/_dict_vectorizer.py','sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py','sklearn/linear_model/_logistic.py']),
           ('interpretml/interpret','refs/heads/develop',['LICENSE','python/interpret-core/setup.py','python/interpret-core/interpret/glassbox/_ebm/_ebm.py','python/interpret-core/interpret/utils/_measure_interactions.py'])]
    receipts=[]
    for repo,ref,files in specs:
        response=subprocess.run(['git','ls-remote','https://github.com/'+repo+'.git',ref,ref+'^{}'],capture_output=True,text=True,timeout=30)
        lines=response.stdout.strip().splitlines()
        commit=None
        errors=[]
        if response.returncode == 0 and lines:
            line=next((v for v in lines if v.endswith('^{}')),lines[0])
            commit=line.split()[0]
        else:
            errors.append({'method':'git_ls_remote','error':response.stderr.strip()})
            try:
                commit=json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/'+ref.split('/')[-1]))['sha']
            except Exception as e:
                errors.append({'method':'github_commit_api','error':str(e)})
        download_ref=commit or ref.split('/')[-1]
        folder=OUT/'sources'/repo.replace('/','__');folder.mkdir(parents=True,exist_ok=True)
        entry={'repo':repo,'requested_ref':ref,'commit':commit,'resolution_errors':errors,'files':[]}
        for path in files:
            url='https://raw.githubusercontent.com/'+repo+'/'+download_ref+'/'+path
            try:
                raw=fetch(url); dest=folder/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
                entry['files'].append({'path':path,'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
            except Exception as e:
                entry['files'].append({'path':path,'url':url,'error':str(e)})
        receipts.append(entry)
        print(json.dumps({'repo':repo,'commit':commit,'files':len(entry['files'])}),flush=True)
    metadata=json.loads(fetch('https://pypi.org/pypi/interpret-core/json'))
    package={'url':'https://pypi.org/pypi/interpret-core/json','version':metadata['info']['version'],
             'requires_python':metadata['info']['requires_python'],
             'wheels':[v['filename'] for v in metadata['urls'] if v['filename'].endswith('.whl')]}
    (OUT/'source_receipts.json').write_text(json.dumps({'retrieved_utc':datetime.now(timezone.utc).isoformat(),
        'repositories':receipts,'package_metadata':package,'code_executed_or_installed':False},indent=2),encoding='utf-8')


if __name__=='__main__': main()
