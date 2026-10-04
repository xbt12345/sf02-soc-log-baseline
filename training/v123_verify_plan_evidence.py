"""Independent counters for V123 support; verifies publication, not training."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v123_targeted_plan_20260929'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    target=OUT/'verification.json'
    if target.exists():raise FileExistsError(target)
    counts={}
    for name,path in [('v121','artifacts/v121_paired_batch_training_20260929/delivery.json'),
                      ('v122','artifacts/v122_evidence_review_20260929/verification.json')]:
        binding=read(ROOT/path)['artifact_sha256']
        for rel,h in binding.items():assert sha(ROOT/rel)==h,rel
        counts[name]=len(binding)
    summary=read(OUT/'support_summary.json')
    for rel,h in summary['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in read(OUT/'audit_outputs.json').items():assert sha(OUT/rel)==h,rel
    plan=read(ROOT/'training/review_policy/v123_targeted_plan.json')
    for rel,h in plan['evidence_sha256'].items():assert sha(ROOT/rel)==h,rel
    d=pd.read_parquet(OUT/'support_ladder.parquet')
    prior=pd.read_parquet(ROOT/'artifacts/v122_evidence_review_20260929/row_diagnosis.parquet')
    for c in prior.columns:assert d[c].equals(prior[c]),c
    assert len(d)==112807 and not d.row_position.duplicated().any()
    facts=[json.loads(v) for v in d.facts_json]
    fold=d.fold.to_numpy();y=d.truth.to_numpy();roots=d.root.to_numpy()
    # Deliberately use tuple keys and explicit counters rather than audit groupby.
    expected={}
    levels={'exact':None,'destination':{'src_port_fixed','src_port_range'},
            'behavior':{'action','outcome','transport_protocol','src_role','dst_role'}}
    for level,fields in levels.items():
        kk=[]
        for f in facts:
            if level=='exact':v=f
            elif level=='destination':v={k:v for k,v in f.items() if k not in fields}
            else:v={k:f.get(k) for k in fields}
            kk.append(tuple(sorted(v.items())))
        for cls,label in [(1,'M'),(2,'S')]:
            nr=np.zeros(len(d),dtype=np.int64);ng=nr.copy()
            for held in range(3):
                ct=Counter();rr=defaultdict(set)
                for i in np.flatnonzero((fold!=held)&(y==cls)):
                    ct[kk[i]]+=1;rr[kk[i]].add(int(roots[i]))
                for i in np.flatnonzero(fold==held):
                    nr[i]=ct[kk[i]];ng[i]=len(rr[kk[i]])
            assert np.array_equal(nr,d[f'{level}_{label}_rows'])
            assert np.array_equal(ng,d[f'{level}_{label}_roots'])
            expected[level,label]=nr,ng
    q=(y==2)&d.A_all_seven_wrong.to_numpy()
    both=(expected['destination','M'][1]>=2)&(expected['destination','S'][1]>=2)
    visible=[]
    for f in facts:
        if f.get('transport_protocol')=='icmp':
            visible.append(all(isinstance(f.get(k),int) and 0<=f[k]<256 for k in ['icmp_type','icmp_code']))
        else:
            v=f.get('dst_port_fixed')
            visible.append(f.get('transport_protocol') in ['tcp','udp'] and isinstance(v,int) and 0<=v<65536)
    visible=np.array(visible)
    observed=Counter()
    for i in np.flatnonzero(q):
        if both[i]:bucket='known_parameter_two_roots_per_class' if visible[i] else 'unknown_parameter_pooled_support'
        elif expected['destination','S'][0][i]>0:bucket='same_class_support_but_insufficient_dual_root_coverage'
        else:bucket='no_same_class_at_destination_resolution'
        assert bucket==d.diagnostic_bucket.iloc[i]
        observed[bucket]+=1
    assert dict(observed)==plan['diagnostic_population']['buckets']
    assert sum(observed.values())==2236
    assert ((expected['behavior','M'][1][q]>=2)&(expected['behavior','S'][1][q]>=2)).all()
    # Real counterexamples to two tempting but false support claims.
    false_known=int((q&both&~visible).sum())
    exact_absent=(expected['exact','M'][0]==0)&(expected['exact','S'][0]==0)
    coarse_present=(expected['behavior','M'][0]>0)&(expected['behavior','S'][0]>0)
    assert false_known==186
    assert int((q&exact_absent&coarse_present).sum())==1842
    cat=read(ROOT/'mcp_readonly/catalog.json')
    for e in cat['documents']:assert sha(ROOT/e['path'])==e['sha256'],e['id']
    assert cat['project']['authoritative_direction_id']=='v123-review'
    assert cat['project']['authoritative_delivery_id']=='v121-delivery'
    assert plan['new_fits_this_turn']==plan['optimizer_steps_this_turn']==0
    assert plan['phase1']['constants']['primary_fits']==6
    assert plan['phase2']['no_eligible_factor_fits']==0
    counts['catalog']=len(cat['documents'])
    files=[p for p in OUT.iterdir() if p.is_file()]
    files += [ROOT/p for p in ['docs/V123_TARGETED_REMEDIATION_AND_TRAINING_PLAN.md',
        'training/review_policy/v123_targeted_plan.json','training/review_policy/v123_risk_actions.json',
        'training/v123_support_ladder_audit.py','training/v123_verify_plan_evidence.py']]
    result={'status':'v123_plan_evidence_verified_not_runtime_or_quality','new_fits':0,
        'latest_actual_training':'V121','current_direction':'V123','quality_acceptance':False,
        'historical_and_catalog_checks':counts,'independent_rows_verified':len(d),
        'support_levels_verified':3,'buckets':dict(observed),
        'real_counterexamples':{'unknown_is_not_known':false_known,'no_exact_match_does_not_mean_no_coarse_support':1842},
        'training_runtime_implemented':False,'model_promoted':False,
        'artifact_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files}}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
