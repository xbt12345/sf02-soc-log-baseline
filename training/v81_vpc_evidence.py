"""Audit within-record interval availability, without fitting or guessing timestamps."""
import json
import re
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from run_v75 import ROOT, OUT, save, sha, adapter
from v79_execute import rows

DEST=ROOT/'artifacts/v81_diagnosis_20260927'


def finite(s):
    return int(s) if re.fullmatch(r'[0-9]+',s) else None


def main():
    path=DEST/'vpc_interval_audit.json'
    if path.exists():raise FileExistsError(path)
    r=rows();v=adapter();collected=[]
    target=pd.read_parquet(ROOT/'artifacts/v79_execution_20260927/development_predictions.parquet',columns=['event_id','route'])
    for split,positions,filename in [('train',np.flatnonzero(r.route.eq('vpc_v2').to_numpy()),'train.parquet'),('development',np.flatnonzero(target.route.eq('vpc_v2').to_numpy()),'valid_input.parquet')]:
        offset=0
        for b in pq.ParquetFile(ROOT/'data/official'/filename).iter_batches(batch_size=8192,columns=['event_id','message_sanitized'],use_threads=False):
            ix=positions[np.searchsorted(positions,offset):np.searchsorted(positions,offset+len(b))]
            for pos in ix:
                local=int(pos)-offset;s=b.column(1)[local].as_py();a=s.split();assert len(a)==14 and a[0]=='2'
                start,end=finite(a[10]),finite(a[11]);duration=end-start if start is not None and end is not None else None
                packets,bytecount=finite(a[8]),finite(a[9]);parsed=v.prepare_record({'message_sanitized':s})
                assert parsed['route']=='vpc_v2'
                f=parsed['facts'];canon=json.dumps(f,sort_keys=True,separators=(',',':'))
                collected.append({'split':split,'row_position':int(pos),'event_id':b.column(0)[local].as_py(),
                    'fold':int(r.fold.iat[pos]) if split=='train' else None,'label':int(r.label_index.iat[pos]) if split=='train' else None,
                    'start_is_literal_integer':start is not None,'end_is_literal_integer':end is not None,
                    'duration_seconds_candidate':duration,'packets':packets,'bytes':bytecount,'action':a[12],
                    'parser_has_duration': 'duration_seconds' in f,'facts_hash':hashlib.sha256(canon.encode()).hexdigest(),
                    'raw_sha256':hashlib.sha256(s.encode()).hexdigest()})
            offset+=len(b)
    data=pd.DataFrame(collected)
    ans=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet');labels=ans.set_index('event_id').label_binary.map({'benign':0,'malicious':1,'suspicious':2})
    m=data.split.eq('development');data.loc[m,'label']=data.loc[m,'event_id'].map(labels);assert data.label.notna().all();data.label=data.label.astype('int8')
    stats=[]
    for (split,label),g in data.groupby(['split','label']):
        duration=g.duration_seconds_candidate
        good=duration.notna() & duration.ge(0)
        stats.append({'split':split,'class':int(label),'rows':len(g),'literal_nonnegative_intervals':int(good.sum()),
                      'negative_intervals':int(duration.lt(0).sum()),'zero_intervals':int(duration.eq(0).sum()),
                      'duration_in_current_parser':int(g.parser_has_duration.sum()),
                      'duration_quantiles':{str(k):float(z) for k,z in duration[good].quantile([0,.25,.5,.75,1]).items()} if good.any() else None,
                      'actions':{str(k):int(z) for k,z in g.action.value_counts().items()}})
    # An association with development labels is audit material; not a threshold, pseudo-label or selection signal.
    data.to_parquet(DEST/'vpc_interval_records.parquet',index=False)
    result={'new_fits':0,'source_sha256':sha(__file__),'cells':stats,'records_sha256':sha(DEST/'vpc_interval_records.parquet'),
            'scope':'Literal within-record numeric intervals only. Standard VPC syntax does not establish that anonymization preserved real intervals. No class rule or fitted rate feature. Negative, redacted and nonnumeric intervals must never be guessed.'}
    save(path,result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
