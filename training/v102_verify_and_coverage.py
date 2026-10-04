"""Independent graph/metric verification and descriptive official-data support audit."""
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import f1_score, confusion_matrix
from v102_root_review import ROOT, OLD, DEST, behavior, sha


def graph_roots(frame, normalized):
    components = np.unique(frame.component)
    links=[]
    for f,key in [(frame[['component','fid']],'fid'),(normalized[['component','N1_group']],'N1_group')]:
        pairs=f.drop_duplicates()
        lead=pairs.groupby(key).component.transform('min').to_numpy()
        links.append((np.searchsorted(components,pairs.component),np.searchsorted(components,lead)))
    a=np.concatenate([x[0] for x in links]); b=np.concatenate([x[1] for x in links])
    graph=sparse.coo_matrix((np.ones(len(a),np.int8),(a,b)),shape=(len(components),len(components))).tocsr()
    n,labels=connected_components(graph,directed=False)
    minimum=np.full(n,np.iinfo(np.int64).max,np.int64)
    np.minimum.at(minimum,labels,components)
    return minimum[labels[np.searchsorted(components,frame.component)]]


def main():
    assert not (DEST/'verification.json').exists()
    receipt=json.loads((DEST/'receipt.json').read_text())
    checks={'input_hashes':all(sha(ROOT/p)==h for p,h in receipt['inputs'].items()),
        'output_hashes':all(sha(DEST/p)==h for p,h in receipt['outputs'].items()),
        'script_hash':sha(ROOT/'training/v102_root_review.py')==receipt['script_sha256']}
    source=pd.read_parquet(OLD/'source_rows.parquet')
    n1=pd.read_parquet(OLD/'N1_ASA_group_map.parquet')
    sa=source[source.is_ASA].merge(n1[['row_position','N1_group']],on='row_position',validate='1:1')
    roots=graph_roots(source,sa)
    ledger=pd.read_parquet(DEST/'source_error_and_support_ledger.parquet')
    expected=pd.Series(roots,index=source.row_position).loc[ledger.row_position].to_numpy()
    checks['independent_scipy_connected_components']=bool(np.array_equal(expected,ledger.merged_root))
    audit=json.loads((DEST/'audit.json').read_text())
    for name, metric in audit['models'].items():
        pred=ledger[name];y=ledger.label_index
        checks[name+'_metrics']=bool(abs(f1_score(y,pred,labels=[1,2],average='macro')-metric['MS_F1'])<1e-12 and int((y!=pred).sum())==metric['errors'])
        recalls=ledger.assign(correct=pred==y).groupby(['merged_root','label_index']).correct.mean()
        checks[name+'_root_recall']=all(abs(recalls.xs(c,level='label_index').mean()-metric['class_root_macro_recall'][str(c)])<1e-12 for c in (1,2))
    s=ledger[ledger.label_index==2].copy();s['correct']=s.R0_MAG_200==2
    roots_s=s.groupby(['merged_root','fold']).agg(rows=('row_position','size'),correct=('correct','sum'),old_components=('component','nunique'))
    roots_s.sort_values('rows',ascending=False).to_csv(DEST/'S_group_coverage.csv')
    errors=ledger[ledger.R0_MAG_200!=ledger.label_index]
    detail={'S_roots':len(roots_s),'S_zero_correct_roots':int((roots_s.correct==0).sum()),
        'S_all_correct_roots':int((roots_s.correct==roots_s.rows).sum()),
        'largest_S_root_rows':int(roots_s.rows.max()),
        'largest_S_root_share':float(roots_s.rows.max()/len(s)),
        'candidate_wrong_by_class_and_support':errors.groupby(['label_index','support_bucket']).size().rename('errors').reset_index().to_dict('records')}
    (DEST/'coverage_detail.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding='utf-8')
    # Use ALL official labels only for a declared descriptive coverage audit.
    # No prediction fitting, checkpoint/threshold selection, or new blind-test claim.
    rowpath=ROOT/'artifacts/v75_four_arm_20260921_r2/rows.parquet'
    r=pd.read_parquet(rowpath,columns=['row_position','component','projection_id','route','label_index'])
    r['fid']=np.load(ROOT/'artifacts/v79_execution_20260927/row_feature_id.npy',mmap_mode='r')
    allasa=r[r.route=='asa'].merge(n1[['row_position','N1_group']],on='row_position',validate='1:1')
    allroots=graph_roots(r,allasa)
    allasa['official_merged_root']=allroots[allasa.row_position.to_numpy()]
    projections=pd.read_parquet(ROOT/'artifacts/v75_four_arm_20260921_r2/projections.parquet',columns=['facts'])
    keys={int(p):behavior(json.loads(projections.facts.iat[int(p)])) for p in allasa.projection_id.unique()}
    allasa['behavior']=allasa.projection_id.map(keys)
    tally=allasa[allasa.behavior.notna()].groupby(['behavior','label_index']).agg(rows=('row_position','size'),roots=('official_merged_root','nunique')).reset_index()
    tally.to_csv(DEST/'all_official_behavior_support.csv',index=False)
    rootsets=allasa[allasa.behavior.notna()].groupby(['behavior','label_index']).official_merged_root.agg(set).to_dict()
    lookup=allasa.set_index('row_position').official_merged_root
    zero=ledger[ledger.support_bucket=='zero_same_class'].copy()
    zero['other_official_same_class_roots']=[len(rootsets.get((b,int(c)),set())-{int(lookup.loc[pos])}) for pos,b,c in zip(zero.row_position,zero.behavior,zero.label_index)]
    zero[['row_position','label_index','behavior','other_official_same_class_roots']].to_parquet(DEST/'source_zero_support_official_coverage.parquet',index=False)
    counts=[]
    for c,g in zero.groupby('label_index'):
        for bucket,take in [('no_other_official_root',g.other_official_same_class_roots==0),('one_other_official_root',g.other_official_same_class_roots==1),('two_or_more_other_official_roots',g.other_official_same_class_roots>=2)]:
            h=g[take]
            counts.append({'class':int(c),'coverage':bucket,'rows':len(h),'candidate_errors':int((h.R0_MAG_200!=h.label_index).sum())})
    full={'status':'descriptive_full_official_coverage_not_validation','official_rows':len(r),'ASA_rows':len(allasa),
        'source_ASA_rows':len(ledger),'source_ASA_fraction':len(ledger)/len(allasa),
        'ASA_classes':allasa.label_index.value_counts().sort_index().to_dict(),
        'official_labelled_population_merged_roots':int(len(np.unique(allroots))),
        'source_zero_support_coverage_elsewhere_in_official':counts,
        'scope':'All official labels including repeatedly observed old development roles used ONLY to count behavior support. This informs the plan, not an unbiased performance estimate. Expanded coverage is not demonstrated prediction improvement.'}
    (DEST/'full_official_coverage.json').write_text(json.dumps(full,ensure_ascii=False,indent=2),encoding='utf-8')
    checks['all_source_rows_present_in_official']=bool(set(ledger.row_position).issubset(set(allasa.row_position)))
    result={'all_checks_passed':all(checks.values()),'checks':checks,'new_classifier_fits':0,
        'scope':'Independent graph and metric verification; not a full model-checkpoint replay or quality acceptance.',
        'script_sha256':sha(__file__),'new_outputs':{p.name:sha(p) for p in DEST.iterdir() if p.is_file() and p.name not in receipt['outputs']}}
    (DEST/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'verification':result['all_checks_passed'],'detail':detail,'full_coverage':full},ensure_ascii=False))


if __name__=='__main__':main()
