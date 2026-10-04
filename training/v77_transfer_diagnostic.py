"""Frozen-model 2x2 dataset/refit audit. No fit, tuning, or source mutation."""
import argparse
import functools
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, NAMES, BATCH, read, save, sha, adapter, metrics
from v75_views import view, byte_matrix
from v75_corrective import stable
from v75_metadata import encode

DEST = ROOT / 'evidence/2026-09-22/v77_attribution'
MODELS = {'D_inner': OUT/'four_arm/D.joblib', 'D_full': OUT/'full/FULL.joblib',
          'F_inner': OUT/'corrective/F.joblib', 'F_full': OUT/'F_diagnostic/model.joblib'}


def infer():
    if DEST.exists():
        raise FileExistsError(DEST)
    # Bind and verify source artifacts before running new predictions.
    delivery = read(ROOT/'evidence/2026-09-21/v75_four_arm/delivery.json')
    for name, h in delivery['artifact_sha256'].items():
        assert sha(ROOT/name) == h, name
    DEST.mkdir(parents=True)
    inp = ROOT/'data/official/valid_input.parquet'
    from v73_inference import INPUT_COLUMNS
    pf = pq.ParquetFile(inp)
    assert set(pf.schema_arrow.names) == set(INPUT_COLUMNS)
    models = {k: joblib.load(p) for k, p in MODELS.items()}
    v = adapter()
    enc = joblib.load(OUT/'facts_encoder.joblib')
    refs = {k: pd.read_parquet(OUT/p/'predictions.parquet') for k,p in
            [('D_full','official_replay'),('F_full','F_diagnostic')]}
    contract = {'new_fits': 0, 'answers_read_during_inference': False,
        'model_sha256': {k: sha(p) for k,p in MODELS.items()},
        'input_sha256': sha(inp), 'source_sha256': sha(__file__),
        'purpose': 'Same-input frozen inner/full model comparison, inspected development only; no threshold or checkpoint selection.',
        'reference_sha256': {k: sha(OUT/p/'predictions.parquet') for k,p in
                            [('D_full','official_replay'),('F_full','F_diagnostic')]}}
    save(DEST/'contract.json', contract)
    parse = functools.lru_cache(maxsize=2048)(lambda s: v.prepare_record({'message_sanitized': s}))
    offset = 0
    writer = None
    max_delta = {k: 0.0 for k in refs}
    start = time.monotonic()
    with threadpool_limits(limits=4):
        for batch in pf.iter_batches(batch_size=BATCH, use_threads=False):
            df = batch.to_pandas()
            inv, unique = pd.factorize(df.message_sanitized.fillna('').astype(str), sort=False)
            parsed = [parse(s) for s in unique]
            facts = [p['facts'] for p in parsed]
            texts = [view(s)[0] for s in unique]
            fx = normalize(enc.transform(facts).astype(np.float32), norm='l2', copy=False)
            ports = np.array([f.get('src_port_fixed',65536) for f in facts])[inv]
            extra,_,_ = encode(df.src_port.tolist(), ports)
            data = {'row_position': np.arange(offset,offset+len(df)), 'event_id': df.event_id.to_numpy(),
                    'route': np.array([p['route'] for p in parsed])[inv]}
            for arm in ['D', 'F']:
                tx = texts if arm == 'D' else [stable(s) for s in texts]
                x = sparse.hstack([byte_matrix(tx),fx],format='csr',dtype=np.float32)[inv]
                x = sparse.hstack([x,extra],format='csr',dtype=np.float32)
                for role in ['inner','full']:
                    key = arm+'_'+role
                    q = models[key].predict_proba(x)
                    logits = models[key].decision_function(x)
                    pred = q.argmax(1).astype(np.int8)
                    assert np.isfinite(q).all() and np.isfinite(logits).all()
                    np.testing.assert_array_equal(pred,logits.argmax(1))
                    np.testing.assert_allclose(q.sum(1),1,atol=1e-5)
                    if key in refs:
                        ref = refs[key].iloc[offset:offset+len(df)]
                        np.testing.assert_array_equal(ref.event_id,df.event_id)
                        np.testing.assert_array_equal(ref.route,data['route'])
                        np.testing.assert_array_equal(ref.pred,pred)
                        before = ref[['p0','p1','p2']].to_numpy()
                        np.testing.assert_allclose(before,q,atol=1e-7,rtol=1e-6)
                        max_delta[key] = max(max_delta[key],float(np.abs(before-q).max()))
                    data[key+'_pred'] = pred
                    # Actual decoder ranking and separate raw-logit diagnostic.
                    data[key+'_ms_score'] = q[:,1]/np.maximum(q[:,1]+q[:,2],1e-300)
                    data[key+'_ms_logit_margin'] = logits[:,1]-logits[:,2]
                    for i in range(3): data[key+'_p'+str(i)] = q[:,i]
            t = pa.Table.from_pydict(data)
            if writer is None: writer = pq.ParquetWriter(DEST/'predictions.parquet',t.schema,compression='zstd')
            writer.write_table(t)
            offset += len(df)
            if offset % 262144 == 0:
                print({'stage':'frozen_inference','rows':offset,'seconds':round(time.monotonic()-start,1)},flush=True)
    writer.close()
    assert offset == pf.metadata.num_rows
    save(DEST/'inference_receipt.json',{'new_fits':0,'rows':offset,'answers_read':False,
        'full_model_labels_match_prior':True,'max_probability_delta':max_delta,
        'probability_argmax_equals_logit_argmax':True,'seconds':time.monotonic()-start,
        'predictions_sha256':sha(DEST/'predictions.parquet'),'contract_sha256':sha(DEST/'contract.json')})
    print({'stage':'inference_complete','rows':offset,'seconds':round(time.monotonic()-start,1)},flush=True)


def compare(y,a,b):
    return {'rows':len(y),'inner_errors':int((a!=y).sum()),'full_errors':int((b!=y).sum()),
            'fixed_by_refit':int(((a!=y)&(b==y)).sum()),'broken_by_refit':int(((a==y)&(b!=y)).sum()),
            'label_changes':int((a!=b).sum())}


def score():
    rec = read(DEST/'inference_receipt.json')
    assert sha(DEST/'predictions.parquet') == rec['predictions_sha256']
    assert sha(DEST/'contract.json') == rec['contract_sha256']
    p = pd.read_parquet(DEST/'predictions.parquet')
    answers_path = ROOT/'data/official/valid_answer_private.parquet'
    ans = pd.read_parquet(answers_path)
    assert p.event_id.is_unique and ans.event_id.is_unique
    d = p.merge(ans,on='event_id',validate='one_to_one',how='left',sort=False)
    assert len(d) == 2014052 and d.label_binary.isin(NAMES).all()
    y = d.label_binary.map(dict(zip(NAMES,range(3)))).to_numpy()
    cells=[];route_pairs=[];rank=[];totals={}
    for model in MODELS:
        pred=d[model+'_pred'].to_numpy()
        totals[model]=metrics(np.bincount(y*3+pred,minlength=9).reshape(3,3))
    for route,z in d.groupby('route'):
        ix=z.index.to_numpy(); yy=y[ix]
        for arm in ['D','F']:
            a=z[arm+'_inner_pred'].to_numpy();b=z[arm+'_full_pred'].to_numpy()
            route_pairs.append({'route':route,'arm':arm,**compare(yy,a,b)})
            for label in np.unique(yy):
                take=yy==label
                cells.append({'route':route,'arm':arm,'label':NAMES[label],**compare(yy[take],a[take],b[take])})
        ms=yy!=0
        for model in MODELS:
            scores=z[model+'_ms_score'].to_numpy()[ms];labels=(yy[ms]==1).astype(int)
            both=len(np.unique(labels))==2
            rank.append({'route':route,'model':model,'M_rows':int(labels.sum()),'S_rows':int(len(labels)-labels.sum()),
                'M_vs_S_auroc':float(roc_auc_score(labels,scores)) if both else None,
                'M_average_precision':float(average_precision_score(labels,scores)) if both else None,
                'M_prevalence_in_MS':float(labels.mean()) if len(labels) else None,
                'M_vs_S_logit_margin_auroc':float(roc_auc_score(labels,z[model+'_ms_logit_margin'].to_numpy()[ms])) if both else None})
    pd.DataFrame(cells).to_csv(DEST/'paired_target_cells.csv',index=False)
    pd.DataFrame(rank).to_csv(DEST/'target_ranking.csv',index=False)
    # Full D has seen these holdout rows: this corner is refit behavior, never validation.
    r=pd.read_parquet(OUT/'rows.parquet')
    inner=pd.read_parquet(OUT/'four_arm/predictions.parquet')
    full=pd.read_parquet(OUT/'full/predictions.parquet')
    np.testing.assert_array_equal(r.row_position,inner.row_position)
    np.testing.assert_array_equal(r.row_position,full.row_position)
    hold=r.is_validation.to_numpy();yt=r.label_index.to_numpy()
    a=inner.D_pred.to_numpy();b=full.FULL_pred.to_numpy()
    train_cells=[]
    for (route,label),z in r[hold].groupby(['route','label_index']):
        ix=z.index.to_numpy()
        train_cells.append({'route':route,'label':NAMES[label],**compare(yt[ix],a[ix],b[ix])})
    pd.DataFrame(train_cells).to_csv(DEST/'former_holdout_refit_cells.csv',index=False)
    fw=read(OUT/'corrective/fit_weights.json');allw=read(OUT/'F_diagnostic/fit_support.json')
    mapping={(x['route'],x['label']):x for x in allw}
    weights=[]
    for w in fw:
        q=mapping[(w['route'],w['label'])]
        weights.append({'route':w['route'],'label':w['label'],'inner_rows':w['original_fit_rows'],
            'full_rows':q['original_rows'],'inner_weight':w['weight'],'full_weight':q['weight'],
            'full_to_inner_weight':q['weight']/w['weight']})
    pd.DataFrame(weights).to_csv(DEST/'refit_weight_changes.csv',index=False)
    result={'new_fits':0,'quality_acceptance':False,'scope':'Already inspected answer DEVELOPMENT diagnostics. Frozen models; no threshold fitting, no model promotion.',
        'totals_on_same_target_rows':totals,'target_paired_by_route':route_pairs,
        'D_former_holdout_comparison':compare(yt[hold],a[hold],b[hold]),
        'former_holdout_warning':'Full model fitted these rows; full-model cell is resubstitution, NOT heldout/generalization.',
        'causal_limit':'Refit contrast holds target records and representation fixed, but jointly changes training membership, update count/order and, for F, derived weights. It is not a pure training-size effect.',
        'ranking_limit':'Within-route M/S ranking conditions on true M/S and excludes benign; not complete three-class quality or evidence calibration transfers.',
        'source_sha256':sha(__file__),'private_answers_sha256':sha(answers_path),
        'outputs_sha256':{q.name:sha(q) for q in DEST.iterdir() if q.is_file() and q.name!='diagnosis.json'}}
    save(DEST/'diagnosis.json',result)
    print({k:{f:v[f] for f in ['errors','recall','macro_f1','normal_false_alerts']} for k,v in totals.items()})
    print(pd.DataFrame(route_pairs).query("route in ['asa','unsupported','vpc_v2']").to_string(index=False))
    print(pd.DataFrame(rank).query("route in ['asa','unsupported','vpc_v2']").to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['infer','score']);a=p.parse_args()
    {'infer':infer,'score':score}[a.stage]()
