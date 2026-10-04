from pathlib import Path
import hashlib
import json
import sys
import gc
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def counts(s, n=25):
    return {str(k): int(v) for k,v in s.fillna('<NULL>').value_counts(dropna=False).head(n).items()}

def main():
    summaries = {}
    populations = {}
    for role in ['train','valid_input']:
        path = ROOT / 'data/official' / (role+'.parquet')
        df = pd.read_parquet(path)
        if role == 'valid_input':
            ans = pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
            assert not ans.event_id.duplicated().any()
            assert set(df.event_id) == set(ans.event_id)
            df = df.merge(ans, on='event_id', how='left', validate='one_to_one')
            del ans
        y = df.label_binary
        summary = {'file':str(path.relative_to(ROOT)), 'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(), 'rows':len(df), 'columns':list(df), 'labels':counts(y), 'id_duplicates':int(df.event_id.duplicated().sum()),'fields':{}, 'cross_tabs':{},'samples':{}}
        for col in df:
            s = df[col]
            empty = s.isna() | s.astype(str).str.strip().eq('')
            summary['fields'][col] = {'dtype':str(s.dtype),'null':int(s.isna().sum()),'empty_or_null':int(empty.sum()),'unique':int(s.nunique(dropna=True)),'top':counts(s,12)}
        ts = pd.to_datetime(df.timestamp,unit='s',utc=True,errors='coerce')
        summary['time'] = {'min':str(ts.min()),'max':str(ts.max()),'unique':int(ts.nunique()),'invalid':int(ts.isna().sum()),'decreasing_adjacent':int((df.timestamp.diff()<0).sum()),'months':counts(ts.dt.strftime('%Y-%m'),100)}
        summary['cross_tabs']['month_label'] = pd.crosstab(ts.dt.strftime('%Y-%m'),y).to_dict(orient='index')
        for col in ['pipeline','vendor_name','product_name','src_host','dst_host']:
            ct = pd.crosstab(df[col].fillna('<NULL>'),y)
            ct['total']=ct.sum(axis=1)
            ct=ct.sort_values('total',ascending=False)
            summary['cross_tabs'][col] = ct.head(60).to_dict(orient='index')
        msg = df.message_sanitized.fillna('')
        summary['message_length'] = msg.str.len().describe(percentiles=[.5,.9,.99]).to_dict()
        summary['cross_tabs']['message_empty_label'] = pd.crosstab(msg.str.strip().eq(''),y).to_dict(orient='index')
        for label in sorted(y.unique()):
            part=df[y==label]
            summary['samples'][label] = part.sample(min(10,len(part)),random_state=42).to_dict(orient='records')
        feature_cols=[c for c in df if c not in ['event_id','label_binary']]
        full_hash=pd.util.hash_pandas_object(df[feature_cols].fillna(''),index=False).to_numpy()
        no_time_cols=[c for c in feature_cols if c!='timestamp']
        no_time_hash=pd.util.hash_pandas_object(df[no_time_cols].fillna(''),index=False).to_numpy()
        summary['duplicates']={}
        for key,h in [('all_features',full_hash),('without_timestamp',no_time_hash)]:
            groups=pd.DataFrame({'h':h,'label':y}).groupby('h').label.agg(['size','nunique'])
            summary['duplicates'][key]={'unique_groups':len(groups),'duplicate_extra_rows':len(df)-len(groups),'conflicting_groups':int((groups['nunique']>1).sum()),'rows_in_conflicting_groups':int(groups.loc[groups['nunique']>1,'size'].sum())}
        populations[role]={'event_id':set(df.event_id),'full_hash':set(full_hash),'no_time_hash':set(no_time_hash),'categorical':{c:set(df[c].fillna('').astype(str)) for c in ['pipeline','src_ip','dst_ip','src_host','dst_host','username','product_name','vendor_name']}}
        if role=='valid_input':
            previous=populations['train']
            summary['overlap_with_train']={'event_id_count':len(set(df.event_id)&previous['event_id']),'all_features_rows':int(np.isin(full_hash,list(previous['full_hash'])).sum()),'without_timestamp_rows':int(np.isin(no_time_hash,list(previous['no_time_hash'])).sum()),'fields':{}}
            for c,known in previous['categorical'].items():
                s=df[c].fillna('').astype(str)
                summary['overlap_with_train']['fields'][c]={'seen_row_fraction':float(s.isin(known).mean()),'seen_nonempty_row_fraction':float(s[s.ne('')].isin(known).mean()),'shared_values':len(set(s)&known)}
        summaries[role]=summary
        dump(role+'_profile.json',summary)
        print(json.dumps({'role':role,'rows':len(df),'labels':summary['labels'],'time':summary['time'],'duplicates':summary['duplicates'],'overlap':summary.get('overlap_with_train')},ensure_ascii=False),flush=True)
        del df, msg, ts, full_hash, no_time_hash, groups, part, y
        gc.collect()
    dump('data_summary.json',summaries)

if __name__=='__main__':
    main()
