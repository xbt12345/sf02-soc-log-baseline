"""Error examples and allowed-fit evidence support; no training/relabeling."""
from pathlib import Path
import sys,json,collections
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'training'))
from run_v36_train import string_codes

def save(n,v): (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def selected(path,positions,columns):
    result={}; offset=0;positions=np.array(sorted(set(positions)))
    for b in pq.ParquetFile(path).iter_batches(batch_size=4096,columns=columns,use_threads=False):
        lo,hi=np.searchsorted(positions,[offset,offset+len(b)])
        if hi>lo:
            for p,r in zip(positions[lo:hi],b.take(pa.array(positions[lo:hi]-offset)).to_pylist()):result[int(p)]=r
        offset+=len(b)
    return result

def main():
    cloud=ROOT/'artifacts/v36_cloud_20260912T191112Z/work';prepared=ROOT/'artifacts/v36_prepared_r13_20260912/prepared.parquet'
    gt=pq.read_table(cloud/'prepared_attempt001/groups.parquet');groups=gt['union_group'].to_numpy();y=gt['label_index'].to_numpy().astype(int)
    prot=pq.read_table(cloud/'prepared_attempt001/protocol.parquet');allpos=np.arange(len(y))
    facts,fc=string_codes(prepared,allpos,'facts');fd=[json.loads(v) for v in facts]
    routes,rc=string_codes(prepared,allpos,'route');asa=np.array([v=='asa' for v in routes])[rc]
    counts=np.bincount(fc*3+y,minlength=len(facts)*3).reshape(-1,3)
    summary={};selections=[]
    for task in ['source_ad','source_duo','source_waf','asa_hard','known_dev']:
        roles=prot[task].to_numpy();fit=(roles==0)|(roles==1);ev=roles==3
        cat=collections.defaultdict(lambda:np.zeros(3,dtype=int));relation=collections.defaultdict(lambda:np.zeros(3,dtype=int))
        fit_counts=np.bincount(fc[fit]*3+y[fit],minlength=len(facts)*3).reshape(-1,3)
        for i,info in enumerate(fd):
            if not fit_counts[i].any():continue
            key=json.dumps({k:info.get(k) for k in ['category','action','outcome','protocol']},sort_keys=True)
            cat[key]+=fit_counts[i]
            key=json.dumps({k:info.get(k) for k in ['protocol','src_role','dst_role']},sort_keys=True)
            relation[key]+=fit_counts[i]
        item={'fit_category_action_outcome_protocol':{k:v.tolist() for k,v in cat.items()},'fit_protocol_role_relationship':{k:v.tolist() for k,v in relation.items() if 'null' not in k}}
        if task=='asa_hard':
            for side,m in [('fit_asa',fit&asa),('evaluation_asa',ev&asa)]:
                fs=np.bincount(fc[m]*3+y[m],minlength=len(facts)*3).reshape(-1,3);coll=(fs>0).sum(1)>1
                item[side]={'rows':int(m.sum()),'class_counts':fs.sum(0).tolist(),'mixed_fact_groups':int(coll.sum()),'mixed_rows':int(fs[coll].sum()),'minimum_errors_fixed_facts':int((fs.sum(1)-fs.max(1)).sum())}
        summary[task]=item
        for variant in ['B0','B2','B2_W']:
            folder=cloud/'models'/(task+'_'+variant);t=pq.read_table(folder/'evaluation.parquet');pos=t['row_position'].to_numpy();ey=t['label_index'].to_numpy();pred=t['prediction'].to_numpy();risk=1-t['p_benign'].to_numpy()
            a=json.loads((folder/'report.json').read_text());th=a['risk'][1]['threshold']
            masks={'wrong_suspicious':(ey==2)&(pred!=2),'normal_alarm':(ey==0)&(risk>=th),'correct_suspicious':(ey==2)&(pred==2)}
            if task=='known_dev':
                products=selected(prepared,pos[(ey==2)&(pred!=2)],['product'])
                wanted=[p for p,v in products.items() if v['product']!='ASA Firewall']
                masks['non_asa_wrong_suspicious']=np.isin(pos,wanted)
            for typ,m in masks.items():
                ix=np.flatnonzero(m);seen=set();chosen=[]
                for j in ix[np.argsort(-risk[ix])]:
                    if groups[pos[j]] in seen:continue
                    seen.add(groups[pos[j]]);chosen.append(int(j))
                    if len(chosen)==4:break
                for j in chosen:selections.append({'model':folder.name,'kind':typ,'row_position':int(pos[j]),'label':int(ey[j]),'prediction':int(pred[j]),'probabilities':[float(t[n][j].as_py()) for n in ['p_benign','p_malicious','p_suspicious']]})
    positions=[v['row_position'] for v in selections]
    prepared_examples=selected(prepared,positions,['b0','b1','facts','product','route','asa_template'])
    raw=selected(ROOT/'data/official/train.parquet',positions,['message_sanitized','event_id'])
    for r in selections:r.update(prepared_examples[r['row_position']]);r.update(raw[r['row_position']])
    save('fit_evidence_support.json',summary);save('targeted_error_examples.json',selections)
    for task,v in summary.items():
        if task=='asa_hard':print(json.dumps({k:x for k,x in v.items() if k not in ['fit_category_action_outcome_protocol']},ensure_ascii=False),flush=True)
    print('EXAMPLES',len(selections),flush=True)
if __name__=='__main__':main()
