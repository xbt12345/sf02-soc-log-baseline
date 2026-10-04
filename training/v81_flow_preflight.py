"""Apply additive observation candidate to every audited real VPC record."""
import json
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from run_v75 import ROOT, save, sha
from v81_flow_observations import derive_vpc_observations


def main():
    dest=ROOT/'artifacts/v81_diagnosis_20260927'
    if (dest/'flow_preflight.json').exists():raise FileExistsError('completed preflight')
    data=pd.read_parquet(dest/'vpc_interval_records.parquet');output=[];counts=Counter();shift_checks=0
    for split,filename in [('train','train.parquet'),('development','valid_input.parquet')]:
        selected=data[data.split.eq(split)].sort_values('row_position').set_index('row_position');positions=selected.index.to_numpy();offset=0
        for batch in pq.ParquetFile(ROOT/'data/official'/filename).iter_batches(batch_size=8192,columns=['event_id','message_sanitized'],use_threads=False):
            ix=positions[np.searchsorted(positions,offset):np.searchsorted(positions,offset+len(batch))]
            for pos in ix:
                i=int(pos)-offset;s=batch.column(1)[i].as_py();r=selected.loc[pos]
                assert batch.column(0)[i].as_py()==r.event_id
                a=derive_vpc_observations(s,'vpc_v2');tokens=s.split()
                observed=a['values'].get('observed_interval_seconds')
                if pd.notna(r.duration_seconds_candidate) and r.duration_seconds_candidate>=0:
                    assert observed==r.duration_seconds_candidate
                    shifted=tokens.copy();shifted[10]=str(int(tokens[10])+10000000);shifted[11]=str(int(tokens[11])+10000000)
                    assert derive_vpc_observations(' '.join(shifted),'vpc_v2')['values']==a['values'];shift_checks+=1
                else:assert observed is None
                for key in a['values']:counts[(split,int(r.label),key)]+=1
                counts[(split,int(r.label),'log_status:'+tokens[13])]+=1
                output.append({'split':split,'row_position':int(pos),'event_id':r.event_id,'raw_sha256':r.raw_sha256,
                               'values_json':json.dumps(a['values'],sort_keys=True),'states_json':json.dumps(a['states'],sort_keys=True),
                               'spans_json':json.dumps(a['spans'],sort_keys=True)})
            offset+=len(batch)
    assert len(output)==len(data)
    pd.DataFrame(output).to_parquet(dest/'flow_observation_candidate.parquet',index=False)
    result={'rows':len(output),'absolute_shift_invariance_checks':shift_checks,'raw_input_changed':False,'candidate_used_by_classifier':False,
            'counts':[{'split':s,'class':c,'field':f,'rows':n} for (s,c,f),n in sorted(counts.items())],
            'source_sha256':sha(__file__),'derivation_source_sha256':sha(ROOT/'training/v81_flow_observations.py'),
            'candidate_sha256':sha(dest/'flow_observation_candidate.parquet'),
            'scope':'Coverage and arithmetic/invariance checks on official records. No guarantee of preserved real timing, new label information or classification gain.'}
    save(dest/'flow_preflight.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
