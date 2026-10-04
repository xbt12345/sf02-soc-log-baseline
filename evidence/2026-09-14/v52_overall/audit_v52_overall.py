"""Read-only model/data diagnosis; writes new v52 evidence, never model selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/2026-09-14/v52_overall'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def metrics(d):
    return {'rows':len(d), 'errors':int(d.error.sum()),
            'confusion_B_M_S':confusion_matrix(d.label_index, d.pred, labels=[0,1,2]).tolist()}

def main():
    assert not OUT.exists(), 'Never overwrite completed or incomplete evidence.'
    OUT.mkdir(parents=True)
    paths=[ROOT/'artifacts/v48_information_repair_r2_20260914'/n for n in ['rows.parquet','projections.parquet']]
    paths += [ROOT/f'artifacts/v51_fact_residual_20260914/fold_{f}/evaluation.parquet' for f in range(3)]
    r=pd.read_parquet(paths[0]);p=pd.read_parquet(paths[1])
    d=pd.concat([pd.read_parquet(x) for x in paths[2:]]).sort_values('row_position').reset_index(drop=True)
    r=r.sort_values('row_position').reset_index(drop=True)
    for k in ['row_position','event_id','label_index','projection_id','route','body_group','fold']:
        if k in d: np.testing.assert_array_equal(r[k],d[k])
        else: d[k]=r[k]
    d['pred']=d[['p_benign','p_malicious','p_suspicious']].to_numpy().argmax(1)
    d['error']=d.pred!=d.label_index
    # The key covers the entire retained text and fact mapping, not just 13 selected fields.
    keys=[json.dumps([str(t),json.loads(f)],sort_keys=True,separators=(',',':')) for t,f in zip(p.text,p.facts)]
    allkeys, codes=np.unique(np.asarray(keys,dtype=object),return_inverse=True)
    d['input_key']=codes[d.projection_id.to_numpy()]
    a=d[d.route=='asa'].copy().reset_index(drop=True)
    assert not p.iloc[a.projection_id].text.fillna('').str.len().any()
    assert a.groupby('body_group').input_key.nunique().max()==1
    assert a.groupby('body_group').fold.nunique().max()==1
    a['support_category']='';a['fit_M']=0;a['fit_S']=0
    floors=[]
    for fold in range(3):
        tr=a[a.fold!=fold];ev=a.fold==fold
        count=pd.crosstab(tr.input_key,tr.label_index).reindex(columns=[1,2],fill_value=0)
        n=count.reindex(a.loc[ev,'input_key'],fill_value=0).to_numpy()
        a.loc[ev,'fit_M']=n[:,0];a.loc[ev,'fit_S']=n[:,1]
        y=a.loc[ev,'label_index'].to_numpy()
        unseen=n.sum(1)==0;mixed=(n>0).all(1);pure=(n>0).sum(1)==1
        match=n[np.arange(len(n)),y-1]>0
        cat=np.where(unseen,'unseen_input',np.where(mixed,'seen_mixed_labels',np.where(match,'seen_pure_same_label','seen_pure_opposite_label')))
        a.loc[ev,'support_category']=cat
        # New fact values vs new combinations, confined to ASA fit support.
        fitfacts=[json.loads(p.iloc[i].facts) for i in tr.projection_id.unique()]
        atoms={(k,json.dumps(v,sort_keys=True)) for f in fitfacts for k,v in f.items()}
        cold={i:any((k,json.dumps(v,sort_keys=True)) not in atoms for k,v in json.loads(p.iloc[i].facts).items()) for i in a.loc[ev,'projection_id'].unique()}
        for idx in a.index[ev & (a.support_category=='unseen_input')]:
            a.loc[idx,'support_category']='unseen_fact_value' if cold[a.loc[idx,'projection_id']] else 'unseen_combination_known_values'
        tab=pd.crosstab(a.loc[ev,'input_key'],a.loc[ev,'label_index']).reindex(columns=[1,2],fill_value=0)
        floor=int(tab.min(axis=1).sum());floors.append({'fold':fold,'finite_evaluation_same_input_min_errors':floor,'mixed_input_groups':int(((tab>0).sum(1)>1).sum())})
    groups=[]
    for (fold,key),g in a.groupby(['fold','input_key']):
        if not g.error.any(): continue
        groups.append({'fold':int(fold),'input_key':int(key),'rows':len(g),'errors':int(g.error.sum()),
            'labels_M_S':[int((g.label_index==j).sum()) for j in [1,2]],'body_groups':int(g.body_group.nunique()),
            'support_categories':g.support_category.value_counts().to_dict(),
            'facts':json.loads(p.iloc[int(g.projection_id.iloc[0])].facts)})
    groups.sort(key=lambda x:x['errors'],reverse=True)
    small=d[(d.route!='asa')&d.error].copy()
    positions=set(small.row_position.astype(int));raw={};offset=0
    rawfile=ROOT/'data/official/train.parquet'
    for batch in pq.ParquetFile(rawfile).iter_batches(batch_size=16384,columns=['message_sanitized']):
        want=sorted(x-offset for x in positions if offset<=x<offset+batch.num_rows)
        if want:
            col=batch.column(0)
            for i in want: raw[offset+i]=col[i].as_py()
        offset+=batch.num_rows
    assert len(raw)==len(positions)
    cases=[]
    for (route,pid),g in small.groupby(['route','projection_id']):
        cases.append({'route':route,'rows':len(g),'labels_B_M_S':[int((g.label_index==j).sum()) for j in range(3)],
            'predictions_B_M_S':[int((g.pred==j).sum()) for j in range(3)],'body_groups':int(g.body_group.nunique()),
            'retained_text':p.iloc[int(pid)].text,'facts':json.loads(p.iloc[int(pid)].facts),
            'raw_examples':[{'row_position':int(pos),'message_sanitized':raw[int(pos)]} for pos in g.row_position.head(3)]})
    summary={'status':'overall_diagnosis_no_new_model','scope':'Previously inspected official development folds; no relabeling, new training, selection or blind validation.',
        'overall':metrics(d),'macro_f1':float(f1_score(d.label_index,d.pred,labels=[0,1,2],average='macro')),
        'by_route':{str(k):metrics(g) for k,g in d.groupby('route')},'ASA_error_fraction':float(a.error.sum()/d.error.sum()),
        'ASA_support_partition':{str(k):metrics(g) for k,g in a.groupby('support_category')},
        'ASA_finite_input_floor_by_fold':floors,'ASA_unique_full_retained_inputs':int(a.input_key.nunique()),
        'ASA_finite_input_floor_total':sum(x['finite_evaluation_same_input_min_errors'] for x in floors),
        'ASA_top_input_error_counts':[g['errors'] for g in groups[:10]],
        'limit':'Finite evaluation input collisions are not proven mislabels, raw-data irreducibility, guaranteed learnable gains or real-world Bayes error.',
        'bindings':{x.relative_to(ROOT).as_posix():sha(x) for x in paths+[rawfile,Path(__file__)]}}
    a.to_parquet(OUT/'asa_row_diagnosis.parquet',index=False)
    small.to_parquet(OUT/'non_asa_error_rows.parquet',index=False)
    save('summary.json',summary);save('asa_error_inputs.json',groups);save('non_asa_raw_review.json',cases)
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save('receipt.json',{'files':{x.name:sha(x) for x in OUT.iterdir() if x.is_file()},'new_model_fits':0})
    print(json.dumps({k:summary[k] for k in ['overall','ASA_error_fraction','ASA_support_partition','ASA_finite_input_floor_total','ASA_top_input_error_counts']},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
