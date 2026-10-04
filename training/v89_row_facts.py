"""Keep partial facts at original-row granularity, avoiding projection-consensus information loss."""
import functools,json,time
import numpy as np,pandas as pd,pyarrow.parquet as pq
from v89_common import ROOT,OUT,DEST,read,save,sha,emit
from v79_execute import rows
from v89_partial_facts import parse


def main():
    r=rows();mapping=np.empty(len(r),np.int32);dictionary=[];codes={};offset=0;start=time.monotonic()
    @functools.lru_cache(maxsize=2048)
    def key(raw,route):
        a=parse(raw,route)
        return json.dumps({'facts':a['facts'],'states':a['states']},sort_keys=True,separators=(',',':'))
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=16384,columns=['message_sanitized'],use_threads=False):
        raw=batch.to_pandas().message_sanitized;routes=r.route.iloc[offset:offset+len(raw)]
        for j,(text,route) in enumerate(zip(raw,routes)):
            s=key(text,str(route))
            if s not in codes:codes[s]=len(dictionary);dictionary.append(s)
            mapping[offset+j]=codes[s]
        offset+=len(raw)
        if offset%262144==0:emit(stage='row_facts',rows=offset,dictionary=len(dictionary),seconds=round(time.monotonic()-start,1))
    assert offset==len(r);np.save(DEST/'row_fact_code.npy',mapping)
    pd.DataFrame({'fact_code':np.arange(len(dictionary)),'observation_json':dictionary}).to_parquet(DEST/'row_fact_dictionary.parquet',index=False)
    hard=read(DEST/'hard_m_facts.json')
    for a in hard:
        decoded=json.loads(dictionary[mapping[a['row_position']]])
        assert decoded['facts']==a['extracted']['facts'] and decoded['states']==a['extracted']['states']
    save(DEST/'row_facts_receipt.json',{'status':'complete','original_rows':len(r),'fact_values_preserved_per_row':True,
        'unique_observation_sets':len(dictionary),'new_fits':0,'source_sha256':sha(__file__),
        'row_mapping_sha256':sha(DEST/'row_fact_code.npy'),'dictionary_sha256':sha(DEST/'row_fact_dictionary.parquet'),
        'seconds':time.monotonic()-start,'scope':'Known literal facts and unknown/conflict states preserved per original row; raw source remains authority. No use of labels for extraction, and no assertion all raw semantics are parsed.'})


if __name__=='__main__':main()
