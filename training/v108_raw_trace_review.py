"""Trace V107 errors back to official messages; no fitting or label invention."""
import json
import re
from collections import Counter
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from run_v75 import ROOT, OUT, save, sha
from v108_root_evidence_audit import DEST, PREV


def main():
    assert not (DEST/'raw_trace_review.json').exists()
    d = pd.read_parquet(DEST/'support_and_error_ledger.parquet')
    facts = pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts
    fm = {int(p):json.loads(facts.iat[int(p)]) for p in d.projection_id.unique()}
    positions = d.row_position.to_numpy()
    raw = []; offset=0
    official = ROOT/'data/official/train.parquet'
    reg = json.loads((PREV/'registration.json').read_text())
    assert sha(official) == reg['input_sha256']['data/official/train.parquet']
    for batch in pq.ParquetFile(official).iter_batches(batch_size=32768,columns=['event_id','message_sanitized']):
        a=np.searchsorted(positions,offset);b=np.searchsorted(positions,offset+len(batch))
        if a!=b:
            loc=positions[a:b]-offset
            part=batch.take(loc).to_pandas()
            part['row_position']=positions[a:b];raw.append(part)
        offset+=len(batch)
    r=pd.concat(raw,ignore_index=True)
    assert np.array_equal(r.row_position,positions)
    d['raw_message']=r.message_sanitized
    d['facts_json']=[facts.iat[int(p)] for p in d.projection_id]
    ep=re.compile(r'\b(?P<role>src|dst)\s+[^\s:]+:[^\s/]+/(?P<token>[^\s]+)')
    stats=Counter();trace=[]
    for i,row in enumerate(d.itertuples()):
        fs=fm[int(row.projection_id)]
        if fs.get('transport_protocol') not in ('tcp','udp'): continue
        tokens={m['role']:m['token'] for m in ep.finditer(row.raw_message)}
        for role in ('src','dst'):
            tok=tokens.get(role);val=fs.get(role+'_port_fixed')
            if val != 65536: continue
            kind='redacted' if tok and 'CRED-' in tok else 'plain_numeric_in_range' if tok and tok.isdecimal() and 0<=int(tok)<=65535 else 'other_or_missing'
            stats[(role,kind)]+=1
            if row.truth==2 and row.N1_TabM25==2 and row.N2_TabM25!=2:
                stats[('regressed_S_'+role,kind)]+=1
            if kind!='redacted': trace.append({'row_position':int(row.row_position),'role':role,'token':tok,'kind':kind})
    # Deterministic top-mass paired case studies, not hand-picked correctness.
    neighbors=pd.read_parquet(DEST/'regressed_S_nearest_input_neighbors.parquet')
    pairs=pd.read_parquet(PREV/'N2_text_pairs.parquet').drop_duplicates('local').set_index('local')
    cases=[]
    for n in neighbors[neighbors.view=='N2'].sort_values(['S_rows','local'],ascending=[False,True]).head(20).itertuples():
        case={'query_local':int(n.local),'heldout_fold':int(n.fold),'regressed_S_rows':int(n.S_rows)}
        for label,local,cl in [('query',n.local,2),('nearest_fit_M',n.nearest_M_local,1),('nearest_fit_S',n.nearest_S_local,2)]:
            rows=d[(d.local==local)&(d.truth==cl)]
            rows=rows[rows.fold==n.fold] if label=='query' else rows[rows.fold!=n.fold]
            assert len(rows)
            row=rows.sort_values('row_position').iloc[0]
            case[label]={'row_position':int(row.row_position),'root':int(row.root),'fold':int(row.fold),'truth':int(row.truth),
                'raw_message':row.raw_message,'facts':json.loads(row.facts_json),
                'N1_text':pairs.loc[int(local),'before'],'N2_text':pairs.loc[int(local),'after']}
        cases.append(case)
    d[['row_position','local','root','fold','truth','raw_message','facts_json','N1_TabM25','N2_TabM25']].to_parquet(DEST/'ASA_raw_fact_prediction_trace.parquet',index=False)
    save(DEST/'top20_regression_case_triples.json',cases)
    save(DEST/'unknown_port_exceptions.json',trace)
    s=d[d.truth==2].groupby('root').agg(rows=('truth','size'),N1_correct=('N1_TabM25',lambda p:int((p==2).sum())),N2_correct=('N2_TabM25',lambda p:int((p==2).sum()))).sort_values('rows',ascending=False)
    top=s.index[:3];take=(d.truth==2)&~d.root.isin(top)
    doc=next((ROOT/'docs/official').glob('附件2*.docx'))
    with ZipFile(doc) as z:xml=ET.fromstring(z.read('word/document.xml'))
    ps=[''.join(x.itertext()) for x in xml.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')]
    paragraphs=[s for s in ps if any(t in s for t in ('评价可重点考察','原始标签分为','本题面向真实企业安全运营场景'))]
    result={'status':'raw_trace_completed_no_fit','classifier_fits':0,'source_sha256':sha(__file__),
        'official_train_sha256':sha(official),'official_question_sha256':sha(doc),
        'ASA_rows_traced':len(d),'unknown_port_role_counts':{f'{a}:{b}':n for (a,b),n in stats.items()},
        'unknown_port_plain_numeric_in_range':sum(v for (k,c),v in stats.items() if k in ('src','dst') and c=='plain_numeric_in_range'),
        'top3_S_groups':s.head(3).reset_index().to_dict('records'),
        'S_excluding_top3_groups':{'rows':int(take.sum()),'N1_correct':int((d.loc[take,'N1_TabM25']==2).sum()),'N2_correct':int((d.loc[take,'N2_TabM25']==2).sum())},
        'question_relevant_paragraphs':paragraphs,
        'limits':['65536 is the parser unknown-port sentinel, never a valid observed TCP/UDP port.',
            'Redacted raw ports cannot be recovered from this file; visible prefixes survive N1/N2 text but are not complete numeric port values.',
            'Case triples are diagnostic, use labels for neighbor selection, and are not inference or training rules.',
            'Top3-excluded scores are retrospective concentration diagnostics, not a replacement benchmark.']}
    save(DEST/'raw_trace_review.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
