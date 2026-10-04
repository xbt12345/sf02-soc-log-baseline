"""Retrospective observable carrier coverage; no new feature training or relabeling."""
import collections
import json
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from run_v75 import ROOT, OUT, BATCH, save, sha
from v78_boundary import DEST


def family(s):
    if all(x in s for x in ['"originId"','"categories"','"destination"']):return 'origin_categories_destination'
    if all(x in s for x in ['"quarantineFolder"','"phish']):return 'mail_quarantine_phish_fields'
    if 'CEF:' in s and 'sql-injection' in s.lower():return 'cef_sql_injection_payload'
    return 'other'


def main():
    train=pd.read_parquet(OUT/'rows.parquet',columns=['event_id','route','label_index'])
    ans=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet').set_index('event_id').label_binary
    pred=pd.read_parquet(DEST/'frozen_development/predictions.parquet',columns=['event_id','route','O_lbfgs_pred'])
    stats=[];samples=[]
    for dataset,filename in [('train','train.parquet'),('development','valid_input.parquet')]:
        counts=collections.Counter();offset=0
        for b in pq.ParquetFile(ROOT/'data/official'/filename).iter_batches(batch_size=8192,columns=['event_id','message_sanitized'],use_threads=False):
            df=b.to_pandas();meta=(train if dataset=='train' else pred).iloc[offset:offset+len(df)]
            np.testing.assert_array_equal(df.event_id,meta.event_id)
            labels=meta.label_index.map({0:'benign',1:'malicious',2:'suspicious'}).tolist() if dataset=='train' else ans.reindex(df.event_id).tolist()
            for j,(eid,s,route,label) in enumerate(zip(df.event_id,df.message_sanitized.fillna(''),meta.route,labels)):
                carrier=family(s);counts[(carrier,route,label)]+=1
                if route=='unsupported' and label=='malicious':
                    samples.append({'dataset':dataset,'row_position':offset+j,'event_id':eid,'carrier':carrier,
                        'pred':None if dataset=='train' else int(meta.O_lbfgs_pred.iloc[j]),'raw_sha256':__import__('hashlib').sha256(s.encode()).hexdigest()})
            offset+=len(df)
        stats.extend({'dataset':dataset,'carrier':k[0],'route':k[1],'label':k[2],'rows':n} for k,n in counts.items())
    pd.DataFrame(stats).to_csv(DEST/'carrier_support.csv',index=False)
    pd.DataFrame(samples).to_parquet(DEST/'unsupported_carrier_ledger.parquet',index=False)
    report={'new_fits':0,'source_sha256':sha(__file__),'scope':'Post-result DEVELOPMENT error explanation. Carrier uses observable key presence, not labels. Not a trained feature, not a general parser, not exhaustive semantic coverage.',
        'important':pd.DataFrame(stats).query("carrier != 'other'").to_dict('records'),
        'unsupported_M':pd.DataFrame(samples).groupby(['dataset','carrier'],dropna=False).size().rename('rows').reset_index().to_dict('records')}
    save(DEST/'carrier_audit.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
