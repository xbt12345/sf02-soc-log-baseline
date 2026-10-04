"""Trace behavior coverage and competing labels without assigning attack truth."""
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from v61_common import read,save,sha,FIELDS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v62_capacity_20260915'
DATA=ROOT/'artifacts/v61_source_factorial_20260914_r2'


def main():
    rows=pd.read_parquet(OUT/'cached_view_keys.parquet')
    raw=pd.read_parquet(DATA/'records.parquet').merge(rows[['row_position','role','key']],on=['row_position','role'],validate='one_to_one')
    assert raw.role.ne('evaluation').all()
    cache=dict(np.load(OUT/'cache.npz'))
    n=pd.DataFrame({'row_position':rows.row_position,'neighbor_count':(cache['neighbors']>=0).sum(1)})
    raw=raw.merge(n,on='row_position',validate='one_to_one')
    semantic=['action','outcome','transport_protocol','src_role','dst_role']
    summaries=[];changes=[];diagnoses={}
    for role in ['fit','selection']:
        a=pd.read_parquet(OUT/'meanmax'/(role+'.parquet'))
        b=pd.read_parquet(OUT/'source_balanced'/(role+'.parquet'))
        a['pred_a']=a[['p_B','p_M','p_S']].to_numpy().argmax(1)
        b['pred_b']=b[['p_B','p_M','p_S']].to_numpy().argmax(1)
        d=raw[raw.role.eq(role)].merge(a[['row_position','pred_a']],on='row_position',validate='one_to_one').merge(b[['row_position','pred_b']],on='row_position',validate='one_to_one')
        d['correct_a']=d.label.eq(d.pred_a);d['correct_b']=d.label.eq(d.pred_b)
        d['fixed']=~d.correct_a&d.correct_b;d['broken']=d.correct_a&~d.correct_b
        group=d.groupby(semantic+['label']).agg(rows=('label','size'),sources=('group','nunique'),correct_baseline=('correct_a','sum'),correct_source_balanced=('correct_b','sum'),fixed=('fixed','sum'),broken=('broken','sum')).reset_index()
        group['role']=role;summaries.append(group)
        mixed=d.groupby('key').label.nunique();mixed=set(mixed[mixed>1].index)
        s=d[d.label.eq(2)&~d.correct_a]
        diagnoses[role]={'S_errors':len(s),'S_errors_empty_context':int(s.neighbor_count.eq(0).sum()),
            'S_errors_nonconflicting_view':int((~s.key.isin(mixed)).sum()),
            'S_errors_nonconflicting_and_empty_context':int(((~s.key.isin(mixed))&s.neighbor_count.eq(0)).sum())}
        c=d[d.fixed|d.broken].copy()
        c=c[['row_position','role','label','group','text']+FIELDS+['neighbor_count','correct_a','correct_b']]
        changes.append(c)
    pd.concat(summaries).to_csv(OUT/'behavior_errors.csv',index=False)
    pd.concat(changes).to_parquet(OUT/'objective_changes.parquet',index=False)
    # Exact fit-side label supports for identical normalized single-event inputs.
    single=pd.factorize(pd.MultiIndex.from_frame(raw[['text']+FIELDS]),sort=False)[0]
    raw['single']=single
    fit=raw[raw.role.eq('fit')];dev=raw[raw.role.eq('selection')]
    support=fit.groupby(['single','label']).agg(rows=('group','size'),sources=('group','nunique')).unstack(fill_value=0)
    support.columns=['fit_'+str(a)+'_'+str(b) for a,b in support.columns]
    joined=dev.merge(support,left_on='single',right_index=True,how='left')
    joined.to_parquet(OUT/'selection_single_event_support.parquet',index=False)
    # Positive and negative classes in the same coarse behavior bucket are retained.
    # This is a support audit, not an assertion that such events share attack intent.
    coarse=fit.groupby(semantic+['label']).group.nunique().unstack(fill_value=0)
    coarse.columns=['fit_sources_'+str(c) for c in coarse.columns]
    coarse.to_csv(OUT/'fit_behavior_support.csv')
    official=next((ROOT/'docs/official').glob('*.docx'))
    xml=ET.fromstring(zipfile.ZipFile(official).read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paras=[''.join(x.text or '' for x in p.findall('.//w:t',ns)) for p in xml.findall('.//w:p',ns)]
    start=next(i for i,p in enumerate(paras) if '基于SOC日志网络安全威胁检测算法设计与实现' in p)
    # The following title marks the next problem. Keep full paragraphs of this one.
    end=next(i for i in range(start+1,len(paras)) if paras[i].startswith('（三）'))
    assert start<end and 'SF-2026-03' not in '\n'.join(paras[start:end])
    (OUT/'official_task_excerpt.txt').write_text('\n'.join(paras[start:end]),encoding='utf-8')
    save(OUT/'coverage_audit.json',{'scope':'Official labels unchanged; behavior and source supports are observations, not adjudication rules',
        'S_error_context_diagnosis':diagnoses,
        'official_document_sha256':sha(official),'official_document':str(official.relative_to(ROOT)),
        'official_excerpt_sha256':sha(OUT/'official_task_excerpt.txt'),
        'no_new_outer_statistics_or_model_predictions':True,
        'limitations':['Coarse behavior overlap is not equivalent to complete conditional support.',
            'Exact-view novelty is not proof that all behavioral semantics are novel.',
            'Source groups use anonymized identifiers and do not prove physical actor/session identity.',
            'Empty context means no other eligible distinct state after target exclusion, not absence of real activity.']})
    print(diagnoses)
    print(pd.concat(summaries).to_string(index=False))


if __name__=='__main__':main()
