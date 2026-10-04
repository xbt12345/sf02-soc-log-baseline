"""Fetch small original implementation files for review, never execute them."""
import concurrent.futures,hashlib,json,urllib.request,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/2026-09-13/v39_methods'
REPOS={
 'PolinaKirichenko/deep_feature_reweighting':['README.md','dfr_evaluate_spurious.py','LICENSE'],
 'HazyResearch/correct-n-contrast':['README.md','contrastive_supervised_loader.py','contrastive_network.py','LICENSE'],
 'logpai/Drain3':['README.md','drain3/template_miner.py','LICENSE'],
 'ocsf/ocsf-schema':['README.md','events/network/network_activity.json','events/iam/authentication.json','objects/network_connection_info.json','objects/network_endpoint.json','LICENSE']}
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'SF02-method-source-review','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=30) as r:return r.read()
def one(repo,paths):
    head=subprocess.run(['git','ls-remote','https://github.com/'+repo+'.git','HEAD'],cwd=ROOT,text=True,capture_output=True,timeout=40,check=True)
    commit=head.stdout.split()[0]
    if len(commit)!=40:raise ValueError('Unexpected public HEAD')
    directory=OUT/'sources'/repo.replace('/','__');directory.mkdir(parents=True,exist_ok=True)
    result={'repository':repo,'commit':commit,'files':[],'not_found':[]}
    for name in paths:
        url='https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+name
        try:blob=get(url)
        except urllib.error.HTTPError as e:
            if e.code==404:result['not_found'].append(name);continue
            raise
        p=directory/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():raise FileExistsError('Do not overwrite source snapshot')
        p.write_bytes(blob)
        result['files'].append({'path':name,'local':str(p.relative_to(ROOT)),'url':url,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
    return result
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'source_receipts_git.json').exists():raise FileExistsError('Preserve receipt')
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(one,r,p):r for r,p in REPOS.items()}
        for f in concurrent.futures.as_completed(futures):
            try:results.append(f.result())
            except Exception as e:results.append({'repository':futures[f],'error':repr(e)})
    (OUT/'source_receipts_git.json').write_text(json.dumps({'scope':'Read-only public git HEAD and raw source retrieval; prior GitHub API rate-limit failures retained; no package install, source execution, or dataset download','results':results},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='files'} for r in results],ensure_ascii=False))
if __name__=='__main__':main()
