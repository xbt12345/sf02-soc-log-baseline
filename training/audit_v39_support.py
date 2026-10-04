"""Additional support and representation checks; no fit, no split mutation."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v38_local_r1_20260913'
sys.path.insert(0,str(RUN/'frozen_training_runtime'))
import joblib,numpy as np,pandas as pd,pyarrow.parquet as pq
from run_v38_train import negative_control
OUT=ROOT/'evidence/2026-09-13/v39_review'
def save(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    if (OUT/'support_supplement.json').exists():raise FileExistsError('Preserve prior run')
    d=pq.read_table(RUN/'prepared/rows.parquet').to_pandas()
    p=pq.read_table(RUN/'prepared/projections.parquet').to_pandas()
    facts=[json.loads(v) for v in p.C_BOTH];mx,names=negative_control(facts)
    codes,_=pd.factorize(np.asarray([bytes(v.astype('u1')) for v in mx.toarray()],dtype=object),sort=False)
    d['mask']=codes[d.projection_id.to_numpy()]
    old=pq.read_table(ROOT/'artifacts/v37_prepared_r13_20260913/prepared.parquet',columns=['raw_hash']).to_pandas()
    d['raw_id']=pd.factorize(old.raw_hash,sort=False)[0]
    support=[]
    for scope,m in [('official_all',np.ones(len(d),bool)),('v38_allowed_dev',d.inner_role>=0),('v38_final_fit',d.inner_role.isin([0,1]))]:
        for (route,label),s in d.loc[m].groupby(['route','label_index']):
            support.append({'scope':scope,'route':route,'label':int(label),'rows':len(s),
                'legacy_components':int(s.union_group.nunique()),'raw_message_keys':int(s.raw_id.nunique()),
                'raw_keys_not_independent_incident_count':True})
    save('route_support.json',support)
    pairs=[]
    for (route,mask),s in d[d.inner_role.isin([0,1])].groupby(['route','mask']):
        c=s.groupby('label_index').agg(rows=('row_position','size'),raw_keys=('raw_id','nunique'),legacy_components=('union_group','nunique')).reset_index()
        if len(c)>1:pairs.append({'route':route,'mask':int(mask),'class_support':c.to_dict('records')})
    save('within_family_contrasts.json',pairs)
    b=joblib.load(RUN/'equal_classifiers_attempt2/C_BOTH/model.joblib')
    x=b['fact_encoder'].transform(facts);fitids=np.unique(d.loc[d.inner_role.isin([0,1]),'projection_id'])
    known=(np.asarray(abs(x[fitids]).sum(0)).ravel()>0)
    ntext=b['model'].coef_.shape[1]-x.shape[1];cold=[]
    for name in ['credential_check=invalid','response=missing','outcome=failure','icmp_code:bit3']:
        j=b['fact_encoder'].feature_names.index(name)
        cold.append({'feature':name,'fit_nonzero_projection_count':int(x[fitids,j].count_nonzero()),'coef':b['model'].coef_[:,ntext+j].tolist()})
    # Read vectorizer vocabulary via its actual exposed object, not guessed state.
    enc_attrs={k:type(v).__name__ for k,v in vars(b['text_encoder']).items()}
    save('support_supplement.json',{'scope':'Observed support and fixed model coefficients; no quality claim',
        'feature_support':cold,'text_encoder_attributes':enc_attrs,'mixed_route_masks':len(pairs),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({'feature_support':cold,'mixed_route_masks':len(pairs)},ensure_ascii=False))
if __name__=='__main__':main()
