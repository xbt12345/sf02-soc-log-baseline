"""Bind read-only diagnostics, source snapshots and revised research documents."""
import ast
import collections
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/2026-09-13/v40_methods/verification.json'
EVID=ROOT/'evidence/2026-09-13/v40_mechanisms'
BASE=ROOT/'artifacts/v39_local_r2_20260913'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    assert not OUT.exists()
    checks={}
    prep=read(BASE/'prepared/complete.json')
    checks['official_train_unchanged']=sha(ROOT/'data/official/train.parquet')==prep['official_sha256']
    checks['prepared_files_unchanged']=all(sha(BASE/'prepared'/n)==h for n,h in prep['files'].items())
    prior=read(ROOT/'evidence/2026-09-13/v39_execution/primary_review/primary_review.json')
    checks['nine_comparison_models_unchanged']=all(sha(ROOT/v['path']/'model.joblib')==v['model_sha256'] for v in prior['model_bindings'])
    report=read(EVID/'summary.json'); stress=read(EVID/'stress_support.json'); ablation=read(EVID/'missing_ablation.json')
    checks['new_diagnostic_scripts_bound']=sha(ROOT/'training/audit_v40_mechanisms.py')==report['script_sha256'] and sha(ROOT/'training/audit_v40_stress_support.py')==stress['script_sha256'] and sha(ROOT/'training/audit_v40_missing_ablation.py')==ablation['script_sha256']
    checks['stress_model_unchanged']=sha(BASE/'old_protocol_stress/model.joblib')==stress['model_sha256']==ablation['model_sha256']
    checks['four_frozen_model_replays_exact']=all(v['replay_max_difference']==0 for v in report['models']) and stress['prediction_replay_max_difference']==0
    totals=collections.Counter(); eval_rows=0; errors=0
    for f in report['folds']:
        d=pq.read_table(BASE/('primary/fold_%s/SEMANTIC/evaluation.parquet'%f['fold']),columns=['label_index','pred_label','route']).to_pandas()
        predicted=d.pred_label.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
        wrong=predicted!=d.label_index.to_numpy()
        eval_rows+=len(d);errors+=int(wrong.sum())
        pp=f['partitions']['all']['evaluation']
        checks['fold_%s_rows_and_errors'%f['fold']]=sum(v['rows'] for v in pp.values())==len(d) and sum(v['errors'] for v in pp.values())==int(wrong.sum())
        aa=f['partitions']['asa']['evaluation']
        checks['fold_%s_asa_errors'%f['fold']]=sum(v['errors'] for v in aa.values())==int((wrong&(d.route.to_numpy()=='asa')).sum())
        for name,v in aa.items():totals[name]+=v['errors']
    checks['full_oof_scope']=eval_rows==1378650 and errors==6153 and sum(totals.values())==6018
    d=pq.read_table(BASE/'old_protocol_stress/evaluation.parquet').to_pandas()
    p=d[['p_benign','p_malicious','p_suspicious']].to_numpy(); y=d.label_index.to_numpy(); wrong=p.argmax(1)!=y
    checks['stress_confident_errors_recount']=int((wrong&(p.max(1)>.95)).sum())==stress['wrong_predictions_over_0_95']==2528
    checks['ablation_changes_conserve_errors']=ablation['baseline']['errors']+sum(v['regressed']-v['fixed'] for v in ablation['routes'].values())==ablation['intervention']['errors']==2238
    checks['ablation_does_not_fix_auth']=ablation['routes']['authentication']['new_errors']==10 and all(np.argmax(v['probabilities'])==1 for v in ablation['authentication_probabilities_after_intervention'] if 'denied' in v['text'])
    # Independently verify the two semantic columns on every fitted projection.
    r=pq.read_table(BASE/'prepared/rows.parquet',columns=['projection_id','inner_role']).to_pandas()
    fitted=set(r.loc[(r.inner_role>=0)&(r.inner_role!=2),'projection_id'])
    proj=pq.read_table(BASE/'prepared/projections.parquet',columns=['projection_id','facts']).to_pandas()
    n=0; bad=0
    for item in proj.itertuples(index=False):
        if item.projection_id not in fitted:continue
        facts=json.loads(item.facts);n+=1
        bad+=int((facts.get('auth_result')=='failure')!=(facts.get('outcome')=='failure'))
    checks['duplicate_failure_columns_recount']=n==68952 and bad==0 and read(EVID/'duplicate_failure_check.json')['identical_in_stress_fit']
    sources=read(OUT.parent/'source_receipts.json')
    checks['eight_source_snapshot_hashes']=sum(len(v['files']) for v in sources['repositories'])==8 and all('sha256' in v and sha(OUT.parent/'sources'/repo['repo'].replace('/','__')/v['path'])==v['sha256'] for repo in sources['repositories'] for v in repo['files'])
    own_scripts=[ROOT/'training'/v for v in ['audit_v40_mechanisms.py','audit_v40_stress_support.py','audit_v40_missing_ablation.py','collect_v40_sources.py','verify_v40_research.py']]
    for pth in own_scripts:ast.parse(pth.read_text(encoding='utf-8'))
    checks['own_script_syntax']=True
    docs=[ROOT/'docs/V40_RESEARCH_REVIEW.md',ROOT/'docs/V40_NEXT_PLAN.md']
    broken=[]
    for pth in docs:
        for target in re.findall(r'\]\(([^)]+)\)',pth.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            dest=(pth.parent/target.split('#')[0]).resolve()
            if dest!=OUT.resolve() and not dest.exists():broken.append([str(pth),target])
    checks['new_document_local_links']=not broken
    bound_files=list(EVID.glob('*.json'))+[OUT.parent/'source_receipts.json']+docs+own_scripts+[ROOT/'README.md',ROOT/'training/README.md',ROOT/'docs/TRAINING_PLAN.md']
    result={'checks':checks,'all_checks_passed':all(checks.values()),'asa_oof_error_partition':dict(totals),
        'official_train_sha256':prep['official_sha256'],'files':{p.relative_to(ROOT).as_posix():sha(p) for p in bound_files},
        'broken_links':broken,'scope':'Frozen model/source/data identity, full saved prediction counts, diagnostic provenance and document links; not new classifier quality',
        'new_model_training_executed':False,'quality_accepted':False,'fresh_blind_test':False,'platform_used':False}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'all_checks_passed':result['all_checks_passed'],'checks':checks,'asa_oof_error_partition':dict(totals)},ensure_ascii=False),flush=True)
    assert result['all_checks_passed']


if __name__=='__main__':main()
