"""Raw-record pressure replay of saved models; no label use in model prediction."""
import json,re
import numpy as np,pandas as pd,pyarrow.parquet as pq,joblib,torch
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from v89_common import ROOT,OUT,DEST,OLD,PRIOR,read,save,sha
from run_v75 import adapter
from v75_views import view,byte_matrix
from v75_corrective import stable
from v75_metadata import encode
from v89_partial_facts import parse
from v89_partial_expression import dictionary_features


def main():
    examples=read(DEST/'actual_R0_stress.json');wanted={a['row_position'] for a in examples};samples={};offset=0
    for b in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=16384,use_threads=False):
        take=sorted(i for i in wanted if offset<=i<offset+b.num_rows)
        if take:
            df=b.to_pandas()
            for i in take:samples[i]=df.iloc[i-offset].to_dict()
        offset+=b.num_rows
        if len(samples)==len(wanted):break
    parser=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');teacher=joblib.load(OLD/'teacher.joblib')
    def make(row):
        rec=parser.prepare_record(row);fx=normalize(enc.transform([rec['facts']]).astype(np.float32),norm='l2')
        mx,_,_=encode([row['src_port']],[rec['facts'].get('src_port_fixed',65536)])
        return sparse.hstack([byte_matrix([stable(view(row['message_sanitized'])[0])]),fx,mx],format='csr')
    hh={}
    gelu=lambda t:.5*t*(1+np.tanh(np.sqrt(2/np.pi)*(t+.044715*t**3)))
    for arm in ['A0','A3']:
        a=torch.load(PRIOR/f'fold1/{arm}_epoch020.pt',map_location='cpu',weights_only=True)['model'];hh[arm]=[a[k].double().numpy() for k in ['first.weight','second.weight','second.bias']]
    models={n:np.load(DEST/(n+'_model.npz')) for n in ['R00','R01','R10','R11','F00'] if (DEST/(n+'_model.npz')).exists()}
    def predict(row,route,name):
        x=make(row);z=np.asarray(x@teacher['coef'])+teacher['intercept'];m=models[name]
        if name=='F00':
            a=parse(row['message_sanitized'],route);v=dictionary_features([json.dumps({k:a[k] for k in ['facts','states']})]).toarray()/m['scale']
        else:
            arm=str(m['representation']);w1,w2,b=hh[arm];h=gelu(gelu(np.asarray(x@w1.T))@w2.T+b)
            v=np.c_[z,h] if bool(m['free_teacher']) else h;v=np.c_[v/m['scale'],np.ones(1)]
        scores=z+v@m['theta'][:3*v.shape[1]].reshape(3,v.shape[1]).T
        return int(scores.argmax(1)[0]),scores[0]
    details=[]
    for e in examples:
        i=e['row_position'];row=samples[i];rename=row.copy();rename['message_sanitized']=re.sub(r'((?:USER|HOST|CRED|ORG)-)(\d+)',lambda m:m[1]+str(int(m[2])+7654321),row['message_sanitized'] or '')
        blank=row.copy();blank['product_name']=None;blank['vendor_name']=None
        clock=row.copy();raw=row['message_sanitized'];_,ledger=view(raw);parts=[]
        for a,b,kind in ledger['spans']:
            segment=raw[a:b]
            if kind=='absolute_clock':segment=re.sub(r'\b(\d{2}):(\d{2}):(\d{2})\b',lambda m:f'{(int(m[1])+1)%24:02}:{m[2]}:{m[3]}',segment)
            parts.append(segment)
        clock['message_sanitized']=''.join(parts) if raw is not None else None
        for name in models:
            orig,score=predict(row,e['route'],name);saved=np.load(DEST/(name+'_all_prediction.npy'),mmap_mode='r')
            if name=='F00':gid=np.load(DEST/'F00_row_group.npy',mmap_mode='r')[i]
            else:gid=np.load(ROOT/'artifacts/v79_execution_20260927/row_feature_id.npy',mmap_mode='r')[i]
            assert orig==saved[gid],(i,name,'raw_original_replay')
            for kind,var in [('identity',rename),('product_empty',blank),('recognized_clock',clock)]:
                pred,s=predict(var,e['route'],name);details.append({'row_position':i,'route':e['route'],'model':name,'pressure':kind,'original_prediction':orig,'variant_prediction':pred,'prediction_invariant':pred==orig,'score_max_abs_change':float(np.abs(s-score).max())})
    counts={}
    for name in models:
        counts[name]={k:sum(not a['prediction_invariant'] for a in details if a['model']==name and a['pressure']==k) for k in ['identity','product_empty','recognized_clock']}
    save(DEST/'stress_model_replay.json',{'status':'complete','source_sha256':sha(__file__),'original_examples':len(examples),'original_raw_model_replay_matches':True,'decision_changes':counts,'details':details,'new_fits':0,'scope':'25 observed format/class examples, renamed marker identities, product blank, recognized absolute clock edits. Limited pressure checks, no realistic external/temporal guarantee. Original R0 identity matrix counterexample retained even if argmax unchanged.'})
    print(json.dumps({'stress_replayed':len(examples),'decision_changes':counts},ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):main()
