"""Recompute model/source errors and exact cached-input ambiguity; no model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from v61_common import read, save, sha, FIELDS
from audit_v61_inputs import minimum_errors

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v62_capacity_20260915'


def codes(values):
    # Dictionary equality is exact; no approximate embedding or hash-distance joins.
    mapping={}; out=[]
    for value in values:
        if value not in mapping: mapping[value]=len(mapping)
        out.append(mapping[value])
    return np.asarray(out,dtype=np.int32)


def cached_keys(arrays):
    text=codes(x.tobytes() for x in arrays['text'])
    facts=codes(x.tobytes() for x in arrays['facts'])
    stat=codes(x.tobytes() for x in arrays['stats'])
    keys=[]
    for i,ns in enumerate(arrays['neighbors']):
        states=tuple(sorted((int(facts[j]),arrays['relation'][i,k].tobytes()) for k,j in enumerate(ns) if j>=0))
        keys.append((int(text[i]),int(facts[i]),int(stat[i]),states))
    return codes(keys)


def prediction_rows(arm,role,rows):
    df=pd.read_parquet(OUT/arm/(role+'.parquet')).sort_values('row_position').reset_index(drop=True)
    expected=rows[rows.role.eq(role)].sort_values('row_position').reset_index(drop=True)
    assert df[['row_position','label','group']].equals(expected[['row_position','label','group']])
    p=df[['p_B','p_M','p_S']].to_numpy()
    assert np.isfinite(p).all() and (p>=0).all() and np.abs(p.sum(1)-1).max()<2e-6
    df['pred']=p.argmax(1);df['correct']=df.pred.eq(df.label)
    return df.merge(expected[['row_position','key']],on='row_position',validate='one_to_one')


def per_source(df):
    return df.groupby(['label','group']).agg(rows=('correct','size'),recall=('correct','mean'),correct=('correct','sum'))


def paired(base,cand,repeats=4000):
    a=per_source(base);b=per_source(cand);assert a.index.equals(b.index)
    delta=(b.recall-a.recall)
    rng=np.random.default_rng(20260919);result={};draws=[]
    for label in [1,2]:
        d=delta.loc[label].to_numpy();sample=[]
        for _ in range(repeats):sample.append(float(rng.choice(d,size=len(d),replace=True).mean()))
        sample=np.asarray(sample);draws.append(sample)
        result[str(label)]={'mean_gain':float(d.mean()),'paired_source_95_interval':np.quantile(sample,[.025,.975]).tolist(),
                           'sources':len(d),'improved_sources':int((d>0).sum()),'worsened_sources':int((d<0).sum())}
    both=(draws[0]+draws[1])/2
    result['M_S_balanced']={'mean_gain':float((delta.loc[1].mean()+delta.loc[2].mean())/2),
                           'paired_source_95_interval':np.quantile(both,[.025,.975]).tolist()}
    result['method']='4000 paired source draws stratified by true class; development diagnostic, not multiplicity-corrected blind inference'
    return result


def main():
    receipt=read(OUT/'cache_receipt.json')
    for name,digest in receipt['bindings'].items():assert sha(OUT/name)==digest
    rows=pd.read_parquet(OUT/'rows.parquet');assert rows.role.ne('evaluation').all()
    arrays=dict(np.load(OUT/'cache.npz'));rows['key']=cached_keys(arrays)
    rows.to_parquet(OUT/'cached_view_keys.parquet',index=False)
    report={'scope':'Previously observed fit/selection only; no old outer evaluation or new label assertion',
            'exact_cached_view':'Stored float32 text/fact vectors and stats plus unordered capped neighbor fact vectors and relation flags',
            'hash_collision_assumption':False,'by_role':{},'arms':{}}
    for role in ['fit','selection']:
        part=rows[rows.role.eq(role)]
        asa=part[part.label.gt(0)].copy();asa['weight']=0.
        for c in [1,2]:
            mask=asa.label.eq(c);sizes=asa[mask].groupby('group').size()
            asa.loc[mask,'weight']=.5/len(sizes)/asa.loc[mask,'group'].map(sizes)
        weighted=asa.groupby(['key','label']).weight.sum().unstack(fill_value=0)
        report['by_role'][role]={'cached_view_conflicts':minimum_errors(part.key,part.label),
              'retrospective_source_balanced_upper_bound':float(weighted.max(axis=1).sum()),
              'upper_bound_is_post_label_finite_sample_lookup_not_learnable_generalization':True,
              'classes':{str(c):{'rows':len(g),'sources':g.group.nunique(),
                  'source_row_quantiles':g.groupby('group').size().quantile([0,.5,.9,1]).to_dict()}
                  for c,g in part.groupby('label')}}
    arms=['meanmax','attention']
    for arm in ['class_balanced','source_balanced']:
        if (OUT/arm/'result.json').exists():arms.append(arm)
    all_predictions={}
    for arm in arms:
        report['arms'][arm]={}
        for role in ['fit','selection']:
            df=prediction_rows(arm,role,rows);all_predictions[(arm,role)]=df
            src=per_source(df);src.reset_index().to_csv(OUT/arm/(role+'_sources.csv'),index=False)
            class_result={}
            mixed=df.groupby('key').label.nunique();mixed=set(mixed[mixed>1].index)
            for label in [1,2]:
                g=df[df.label.eq(label)];s=src.loc[label];wrong=g[~g.correct]
                class_result[str(label)]={'rows':len(g),'sources':len(s),'row_recall':float(g.correct.mean()),
                    'mean_source_recall':float(s.recall.mean()),'zero_recall_sources':int(s.recall.eq(0).sum()),
                    'perfect_recall_sources':int(s.recall.eq(1).sum()),'errors':len(wrong),
                    'errors_at_mixed_cached_views':int(wrong.key.isin(mixed).sum()),
                    'errors_at_role_label_pure_cached_views':int((~wrong.key.isin(mixed)).sum()),
                    'zero_recall_source_sizes':s[s.recall.eq(0)].rows.value_counts().sort_index().to_dict()}
            report['arms'][arm][role]=class_result
    report['comparisons']={arm:paired(all_predictions[('meanmax','selection')],all_predictions[(arm,'selection')]) for arm in arms if arm!='meanmax'}
    base=read(OUT/'meanmax/result.json');report['continuation_gates']={}
    for arm in arms[1:]:
        result=read(OUT/arm/'result.json');delta=report['comparisons'][arm]
        checks={'source_balanced_gain_at_least_1pp':delta['M_S_balanced']['mean_gain']>=.01,
            'S_source_gain_at_least_2pp':delta['2']['mean_gain']>=.02,
            'M_source_drop_at_most_1pp':delta['1']['mean_gain']>=-.01,
            'positive_source_interval':delta['M_S_balanced']['paired_source_95_interval'][0]>0}
        if arm!='attention':
            a=base['selected_selection']['ASA'];b=result['selected_selection']['ASA']
            checks.update(row_macro_drop_at_most_1pp=b['macro_f1_M_S']-a['macro_f1_M_S']>=-.01,
                row_M_recall_drop_at_most_1pp=b['recall_B_M_S'][1]-a['recall_B_M_S'][1]>=-.01,
                row_S_recall_drop_at_most_1pp=b['recall_B_M_S'][2]-a['recall_B_M_S'][2]>=-.01)
        report['continuation_gates'][arm]={'checks':checks,'all_passed':all(checks.values())}
    # What does the fit corpus say for exactly the same frozen representation?
    fit=rows[rows.role.eq('fit')];counts=fit.groupby(['key','label']).size().unstack(fill_value=0)
    for c in [0,1,2]:
        if c not in counts:counts[c]=0
    df=all_predictions[('meanmax','selection')].merge(counts.add_prefix('fit_label_'),left_on='key',right_index=True,how='left')
    df['coverage']='unseen_exact_view'
    seen=df['fit_label_1'].notna();support=df[['fit_label_0','fit_label_1','fit_label_2']].fillna(0).gt(0)
    df.loc[seen&support.sum(axis=1).eq(1),'coverage']='seen_single_label'
    df.loc[seen&support.sum(axis=1).gt(1),'coverage']='seen_conflicting_labels'
    df.to_parquet(OUT/'selection_view_coverage.parquet',index=False)
    report['selection_cached_view_coverage']=df.groupby(['label','coverage']).agg(rows=('correct','size'),errors=('correct',lambda x:int((~x).sum())),sources=('group','nunique')).reset_index().to_dict('records')
    # Semantic coverage is deliberately coarser than exact representation identity.
    raw=pd.read_parquet(ROOT/'artifacts/v61_source_factorial_20260914_r2/records.parquet')
    raw=raw.merge(rows[['row_position','role']],on=['row_position','role'],validate='one_to_one')
    semantic=['action','outcome','transport_protocol','src_role','dst_role']
    coverage=raw.groupby(semantic+['role','label']).agg(rows=('row_position','size'),sources=('group','nunique')).reset_index()
    coverage.to_csv(OUT/'semantic_coverage.csv',index=False)
    save(OUT/'capacity_analysis.json',report)
    print({'by_role':report['by_role'],'arms':report['arms'],'comparisons':report['comparisons'],'coverage':report['selection_cached_view_coverage']})


if __name__=='__main__':main()
