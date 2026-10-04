"""Repair the new review delivery schema; preserve initial failure provenance."""
import json
from v102_root_review import ROOT, DEST, sha
from v102_publish import save


def main():
    target=ROOT/'evidence/2026-09-28/v102_root_review/delivery.json'
    out=DEST/'publication_correction.json'
    assert not out.exists()
    save(out,{'initial_delivery_sha256':sha(target),'initial_local_MCP_tests':'9 run, 2 errors from absent validation_scope',
        'repair':'Add the required delivery validation_scope field; update no-fit status assertions and historical-replay wording. MCP server permissions and tool implementations unchanged.'})
    execute=DEST/'execution_summary.json'
    e=json.loads(execute.read_text());e['validation_scope']=e['scope'];save(execute,e)
    d=json.loads(target.read_text());d['validation_scope']=d['scope']
    paths=[p for p in DEST.iterdir() if p.is_file()]+list((ROOT/'training').glob('v102_*.py'))+[ROOT/d['current_review']]
    d['artifact_sha256']={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    save(target,d)
    for p in [ROOT/'README.md',ROOT/'docs/TRAINING_PLAN.md']:
        p.write_text(p.read_text(encoding='utf-8').replace('最近完整开发回放仍v7.9','最近完整开发回放仍为v7.9'),encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';c=json.loads(cp.read_text(encoding='utf-8'))
    c['project']['current_summary']=c['project']['current_summary'].replace('最近完整开发回放仍v7.9','最近完整开发回放仍为v7.9')
    for entry in c['documents']:
        if entry['id'].startswith('v102-') or entry['path'] in ['README.md','docs/TRAINING_PLAN.md']:
            entry['sha256']=sha(ROOT/entry['path'])
            entry['summary']=entry['summary'].replace('最近完整开发回放仍v7.9','最近完整开发回放仍为v7.9')
        else:assert sha(ROOT/entry['path'])==entry['sha256'],entry['id']
    save(cp,c)
    print(json.dumps({'corrected':True,'bound_files':len(paths),'delivery_sha256':sha(target)}))


if __name__=='__main__':main()
