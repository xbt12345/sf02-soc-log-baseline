"""Independent score replay, complete WAF/Duo wrapper coverage and controls."""
import argparse,collections,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
import v38_representation as rep
import v38_learning as learn
from run_v38_prepare import sha,save,EXPECTED
from run_v38_train import negative_control
from audit_v37_prepared import variants

def main(a):
    prepared=Path(a.prepared);models=Path(a.models);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    assert sha(a.train)==EXPECTED
    rows=pq.read_table(prepared/'rows.parquet').to_pandas();projs=pq.read_table(prepared/'projections.parquet').to_pandas()
    original_cases=json.loads((prepared/'audit_cases.json').read_text(encoding='utf-8'))
    selected=rows['product'].isin(['Duo','Barracuda WAF']).to_numpy(copy=True)
    selected[[c['row_position'] for c in original_cases]]=True
    cases=[];offset=0;counter=collections.Counter();failures=[]
    for b in pq.ParquetFile(a.train).iter_batches(batch_size=2048,columns=['event_id','message_sanitized'],use_threads=False):
        positions=np.flatnonzero(selected[offset:offset+len(b)])
        if len(positions):
            rawrows=b.to_pylist()
            for i in positions:
                pos=offset+int(i);row=rows.iloc[pos];raw=rawrows[i]['message_sanitized'] or ''
                assert rawrows[i]['event_id']==row.event_id
                direct=rep.prepare_message(raw);cached=projs.iloc[int(row.projection_id)]
                if direct['text']!=cached.text or rep.canonical(direct['facts'])!=cached.base_facts or rep.canonical(direct['ports'])!=cached.ports:
                    failures.append(['projection',pos])
                changes=variants(raw,row.route)
                for name,changed in changes:
                    if rep.prepare_message(changed)!=direct:failures.append(['wrapper',pos,name])
                    counter[name]+=1
                cases.append({'position':pos,'raw':raw,'changes':[v for _,v in changes]})
                counter['source:'+str(row['product'])]+=1
        offset+=len(b)
    assert counter['source:Barracuda WAF']==4130 and counter['source:Duo']==337
    save(out/'full_wrapper_audit.json',{'cases':len(cases),'checks':dict(counter),'failures':failures,'passed':not failures,
        'correction':'Preparation sampler used a pandas Series.product attribute which is a method, so its all-WAF/Duo branch did not execute. This supplement explicitly selects the product column and verifies all 4130 WAF and 337 Duo records. Projection construction and training rows were unaffected.'})
    if failures:raise RuntimeError('Full wrapper audit failed')
    # Complete original and transformed messages are predicted in the actual
    # classifier API; batches avoid holding expanded sparse matrices together.
    texts_raw=[];pair_orig=[];pair_changed=[]
    for c in cases:
        base=len(texts_raw);texts_raw.append(c['raw'])
        for raw in c['changes']:
            pair_orig.append(base);pair_changed.append(len(texts_raw));texts_raw.append(raw)
    evaluated=rows[rows.inner_role==2];uids,inv=np.unique(evaluated.projection_id.to_numpy(),return_inverse=True);pr=projs.iloc[uids]
    results=[];availability=json.loads((models/'availability_control.json').read_text(encoding='utf-8'))
    for view in rep.VIEWS:
        folder=models/view;b=joblib.load(folder/'model.joblib');saved=pq.read_table(folder/'evaluation.parquet').to_pandas()
        assert np.array_equal(saved.row_position.to_numpy(),evaluated.row_position.to_numpy())
        facts=[json.loads(v) for v in pr[view]]
        x=sparse.hstack([b['text_encoder'].transform(pr.text.tolist()),b['fact_encoder'].transform(facts)],format='csr')
        logits=x.dot(b['model'].coef_.T)+b['model'].intercept_
        exp=np.exp(logits-logits.max(1,keepdims=True));p=(exp/exp.sum(1,keepdims=True))[inv]
        expected=saved[['p_benign','p_malicious','p_suspicious']].to_numpy();y=saved.label_index.to_numpy();pred=p.argmax(1)
        decision=np.asarray(['benign','malicious','suspicious'])[pred]
        maxdiff=float(abs(p-expected).max());decisiondiff=int((decision!=saved.pred_label.to_numpy()).sum())
        assert maxdiff<=1e-10 and decisiondiff==0
        pp=[]
        for start in range(0,len(texts_raw),256):
            records=[rep.view_record(rep.prepare_message(r),view) for r in texts_raw[start:start+256]]
            prob,_=learn.classify(b,[r['text'] for r in records],[r['facts'] for r in records]);pp.append(prob)
        pp=np.concatenate(pp)
        wrapdiff=float(np.abs(pp[pair_orig]-pp[pair_changed]).max()) if pair_orig else 0.
        wrapdec=int((pp[pair_orig].argmax(1)!=pp[pair_changed].argmax(1)).sum()) if pair_orig else 0
        assert wrapdiff<=1e-12 and wrapdec==0
        # Negative control comparison, including class-conditional agreement.
        mf=[json.loads(v) for v in pr['C_BOTH']];mx,names=negative_control(mf)
        mc=joblib.load(models/'availability_control.joblib');assert names==mc['names']
        mask_pred=mc['model'].predict(mx)[inv]
        agreements=[float((pred[y==c]==mask_pred[y==c]).mean()) for c in range(3)]
        # Diagnostic randomization only: changes truth-relevant values and is
        # not a label-preserving counterfactual or a new quality evaluation.
        keys=pd.factorize(np.asarray([r.tobytes() for r in mx.toarray()],dtype=object),sort=False)[0][inv]
        groups=[np.flatnonzero(keys==k) for k in np.unique(keys)]
        shuffled=[]
        for seed in range(10):
            rng=np.random.default_rng(3800+seed);order=np.arange(len(y))
            for ids in groups:order[ids]=rng.permutation(ids)
            shuffled.append(learn.cm_metrics(y,pred[order])['macro_f1'])
        item={'view':view,'all_evaluation_rows':len(saved),'manual_softmax_max_difference':maxdiff,'saved_decision_differences':decisiondiff,
              'raw_original_messages':len(cases),'raw_inference_messages_with_variants':len(texts_raw),'wrapper_pairs':len(pair_orig),
              'wrapper_probability_max_difference':wrapdiff,'wrapper_decision_differences':wrapdec,
              'availability_prediction_agreement_per_true_class':agreements,
              'conditional_score_permutation_macro_f1':{'mean':float(np.mean(shuffled)),'min':float(min(shuffled)),'max':float(max(shuffled)),
                  'scope':'diagnostic association test within fact-availability patterns; not label-preserving transformations'},
              'recomputed':learn.cm_metrics(y,pred),'model_sha256':sha(folder/'model.joblib')}
        results.append(item);print(json.dumps({k:item[k] for k in ['view','all_evaluation_rows','wrapper_pairs','wrapper_decision_differences','availability_prediction_agreement_per_true_class']}),flush=True)
    save(out/'independent_replay.json',{'models':results,'all_replay_checks_passed':True,'scope':'Numerical/decision replay and specified wrapper invariance only; not model quality acceptance'})
    save(out/'quality_decision.json',{'quality_accepted':False,'platform_full_training_permitted_by_current_plan':False,
        'reasons':['Availability-only three-class control outperforms all three semantic models on this development audit.',
                   'All held Duo suspicious rows are benign; WAF suspicious rows assigned malicious.',
                   'ASA suspicious recall is 48/733 in this internal audit; exact ports offer no clear total gain.',
                   'Initial fit malicious support is two conservative groups; row counts do not establish transfer.'],
        'availability_macro_f1':availability['evaluation']['macro_f1'],'stage_D_decision':'Do not increase capacity or reweight yet: current evidence does not isolate class-prior suppression or nonlinear capacity as the primary failure. Follow the preregistered shortcut/support stop conditions.',
        'historical_outer_protocols_not_scored_this_round':True,'fresh_external_test':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--models',required=True);p.add_argument('--train',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
