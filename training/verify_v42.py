"""Independent saved-artifact replay, support reconstruction and scope attacks."""
import argparse
import collections
import gc
import json
import re
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pyarrow.parquet as pq


def main(a):
    start=time.perf_counter();root=Path(a.root).resolve();run=Path(a.run).resolve();out=Path(a.out).resolve();assert not out.exists()
    sys.path.insert(0,str(run/'frozen_training_runtime'))
    import v42_core as core
    from run_v42 import data,PROBS
    from run_v39_prepare import sha,save
    from run_v40_train import metrics
    from audit_v37_prepared import variants
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    stage=run/'prepared';receipt=read(stage/'complete.json');prep=Path(receipt['v39_prepared']);verified={}
    def check(p,h):
        assert sha(p)==h,str(p);verified[p.relative_to(root).as_posix()]=h
    check(root/'data/official/train.parquet',receipt['official_sha256'])
    check(prep/'complete.json',receipt['v39_prepared_receipt_sha256'])
    for folder,files in [(stage,receipt['files']),(run/'frozen_training_runtime',receipt['runtime_sources']),(prep,read(prep/'complete.json')['files'])]:
        for n,h in files.items():check(folder/n,h)
    for name,h in read(stage/'input_checks.json')['source_bindings'].items():check(root/name,h)
    selection=read(run/'primary_review/selection.json')
    check(run/'primary_review/primary_review.json',selection['primary_review_sha256'])
    check(stage/'configuration.json',selection['configuration_sha256'])
    check(run/'frozen_training_runtime/run_v42.py',selection['script_sha256'])
    folders=sorted((run/'primary').glob('fold_*/*/model.joblib'));assert len(folders)==6
    stresspaths=list((run/'old_protocol_stress').glob('model.joblib'))
    assert len(stresspaths)==int(selection['proceed_to_stress'])
    assert selection['selected_view']==('R' if selection['proceed_to_stress'] else None)
    rows,all_ids,texts,facts=data(prep)
    # Existing real cases plus every distinct newly eligible evaluation input.
    oldcases=root/'artifacts/v40_local_r1_20260913/raw_inference_cases.json';cases=read(oldcases);check(oldcases,sha(oldcases))
    existing={c['row_position'] for c in cases};wanted={}
    for path in folders+stresspaths:
        d=pq.read_table(path.parent/'evaluation.parquet',columns=['row_position','projection_id','context_eligible']).to_pandas()
        for r in d.loc[d.context_eligible].drop_duplicates('projection_id').itertuples():wanted[int(r.row_position)]=int(r.projection_id)
    eligible_positions=set(wanted);wanted={p:i for p,i in wanted.items() if p not in existing};off=0
    meta=pq.read_table(prep/'rows.parquet',columns=['row_position','projection_id','route']).to_pandas()
    for batch in pq.ParquetFile(root/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized'],use_threads=False):
        for pos in sorted(set(wanted).intersection(range(off,off+len(batch)))):
            cases.append({'row_position':pos,'projection_id':wanted[pos],'route':meta.route.iloc[pos],'raw':batch.column(0)[pos-off].as_py() or ''})
        off+=len(batch)
    pr=pq.read_table(prep/'projections.parquet',columns=['text','facts']).to_pandas()
    records=[];expected=[];counts=collections.Counter();representative=[];projected=[];originals=[]
    for i,item in enumerate(cases):
        raw=item['raw'];pid=item['projection_id'];base=core.prior.prepare_record({'message_sanitized':raw})
        assert base['text']==pr.text.iloc[pid] and core.prior.canonical(base['facts'])==pr.facts.iloc[pid]
        originals.append(base)
        per_case=[('original',{'message_sanitized':raw}),('outside_metadata_replaced',{'message_sanitized':raw,
            'timestamp':'2099-01-01T00:00:00Z','product_name':None,'vendor_name':'unseen_vendor','src_ip':'203.0.113.255',
            'dst_ip':'192.0.2.1','username':'unseen_identity','event_id':'replacement','pipeline':'replacement','src_port':65535,
            'src_host':'new_host','dst_host':'new_host','label_binary':'suspicious'})]
        if item['route']=='clock_probe':
            changes=[('absolute_clock',core.prior.TASK_BOUNDARY.sub(lambda m:m[1]+clock+m[3],core.prior.ISO_LITERAL.sub(clock,raw))) for clock in ['2099-01-01T00:00:00Z','2011-12-31T23:59:59Z']]
        else:
            changes=variants(raw,item['route'])
            if item['route']=='asa':changes.append(('address_interface',re.sub(r'dmz[-_]\d+','dmz-999',re.sub(r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)','203.0.113.99',raw))))
        per_case += [(n,{'message_sanitized':r}) for n,r in changes]
        for name,r in per_case:
            p=core.prior.prepare_record(r);assert p['text']==base['text'] and p['facts']==base['facts'],(item['row_position'],name)
            records.append(r);expected.append(i);counts[name]+=1;projected.append(p)
        if i<50 or item['row_position'] in eligible_positions or item['route']=='clock_probe':representative.extend(range(len(records)-len(per_case),len(records)))
    expected=np.array(expected);results=[];stress_review=None
    def predict(bundle,tt,ff):
        baseline=bundle['baseline'];bp=baseline['model'].predict_proba(core.old_parameters.matrix(baseline,tt,ff))
        if bundle['view']=='R':return bundle['offset'].predict(tt,ff,bp)[0],bp
        old=bundle['reference'];pp=old['model'].predict_proba(core.old_parameters.matrix(old,tt,ff));ctx,_=bundle['support'].encode(tt,ff)
        return core.gated_reference(bp,pp,ctx>=0),bp
    for path in folders+stresspaths:
        folder=path.parent;rec=read(folder/'complete.json')
        for n,k in [('model.joblib','model_sha256'),('evaluation.parquet','predictions_sha256'),('report.json','report_sha256'),('binding.json','binding_sha256')]:check(folder/n,rec[k])
        binding=read(folder/'binding.json');assert binding['runtime_sources']==receipt['runtime_sources'];check(stage/'complete.json',binding['prepared_sha256'])
        model=joblib.load(path);baseline_state=joblib.hash(model['baseline']);assert baseline_state==binding['baseline_joblib_state_hash']
        stress=binding['stress'];fold=binding['fold'];view=model['view']
        basefolder=prep.parent/('old_protocol_stress' if stress else 'primary/fold_%s/SEMANTIC'%fold)
        check(basefolder/'model.joblib',binding['reference_model_sha256'])
        base_loaded=joblib.load(basefolder/'model.joblib');assert joblib.hash(base_loaded)==baseline_state;del base_loaded
        fit=(rows.inner_role.to_numpy()!=2) if stress else (rows.fold.to_numpy()!=fold)
        assert int(fit.sum())==binding['fit_rows'] and int((~fit).sum())==binding['evaluation_rows']
        support=model['offset'].support if view=='R' else model['support']
        rebuilt=core.ContextSupport().fit(texts,facts,all_ids[fit],rows.body_group.to_numpy()[fit],3)
        assert rebuilt.context_keys==support.context_keys and rebuilt.support_lower_bound==support.support_lower_bound
        prob,bp=predict(model,texts,facts);ctx,_=support.encode(texts,facts);eligible=ctx>=0
        if view=='R':
            # Independently check the optimum on original rows, without a fit.
            delta=model['offset'].delta;assert (np.abs(delta)<=core.MAX_OFFSET+1e-12).all()
            yfit=rows.label_index.to_numpy()[fit];pid=all_ids[fit]
            use=(ctx[pid]>=0)&(yfit!=0)&(bp.argmax(1)[pid]!=0);pid=pid[use];yt=(yfit[use]==1).astype(float)
            g=ctx[pid];margin=core.conditional_margin(bp)[pid];n=len(delta)
            from scipy.special import expit
            grad=np.bincount(g,weights=expit(margin+delta[g])-yt,minlength=n)+10*delta
            pg=np.where(delta<=-core.MAX_OFFSET+1e-10,np.minimum(grad,0),np.where(delta>=core.MAX_OFFSET-1e-10,np.maximum(grad,0),grad))
            assert not n or np.abs(pg).max()<1e-7
            before=float(np.sum(np.logaddexp(0,margin)-yt*margin))
            after=float(np.sum(np.logaddexp(0,margin+delta[g])-yt*(margin+delta[g]))+5*np.dot(delta,delta))
            assert abs(before-model['offset'].optimization['objective_before'])<1e-7
            assert abs(after-model['offset'].optimization['objective_after'])<1e-7 and after<=before+1e-8
        else:
            original_p=joblib.load(root/('artifacts/v41_local_r1_20260913/primary/fold_%s/P/model.joblib'%fold))
            assert joblib.hash(original_p)==joblib.hash(model['reference']);del original_p
        assert np.array_equal(prob[~eligible],bp[~eligible]) and np.array_equal(prob[:,0],bp[:,0])
        assert np.array_equal(prob.argmax(1)==0,bp.argmax(1)==0)
        d=pq.read_table(folder/'evaluation.parquet').to_pandas();ids=all_ids[~fit]
        assert np.array_equal(d.row_position,rows.row_position.to_numpy()[~fit])
        assert np.array_equal(d.label_index,rows.label_index.to_numpy()[~fit])
        assert np.array_equal(d.context_eligible,eligible[ids])
        diff=float(np.max(np.abs(prob[ids]-d[PROBS].to_numpy())));assert diff<=1e-10
        assert metrics(d.label_index.to_numpy(dtype=int),prob[ids])==read(folder/'report.json')['evaluation']
        # Counterexamples remove a mandatory observation or create a novel full
        # combination. They are feature-level fallbacks, not new attack labels.
        attack_text=[];attack_facts=[];attack_counts=collections.Counter()
        for pid in np.flatnonzero(eligible):
            f=facts[pid];protocol=f['transport_protocol']
            fields=['action','outcome','transport_protocol','src_role','dst_role']+(['src_port_fixed','dst_port_fixed'] if protocol in ('tcp','udp') else ['icmp_type','icmp_code'])
            for field in fields:
                f2=dict(f);f2.pop(field);assert core.complete_context(texts[pid],f2)[0] is None
                attack_text.append(texts[pid]);attack_facts.append(f2);attack_counts['missing_'+field]+=1
            field='dst_port_fixed' if protocol in ('tcp','udp') else 'icmp_code';stop=65536 if protocol in ('tcp','udp') else 256
            for value in range(stop):
                f2=dict(f);f2[field]=value;key,_=core.complete_context(texts[pid],f2)
                if key not in support.support_lower_bound:break
            else:raise AssertionError('No unseen context counterexample')
            attack_text.append(texts[pid]);attack_facts.append(f2);attack_counts['unseen_complete_combination']+=1
        if attack_text:
            attack_prob,attack_base=predict(model,attack_text,attack_facts);attack_ctx,_=support.encode(attack_text,attack_facts)
            assert (attack_ctx<0).all() and np.array_equal(attack_prob,attack_base)
        orig=predict(model,[v['text'] for v in originals],[v['facts'] for v in originals])[0];rawdiff=0.;flips=0
        for lo in range(0,len(records),512):
            v=projected[lo:lo+512];p=predict(model,[x['text'] for x in v],[x['facts'] for x in v])[0];q=orig[expected[lo:lo+512]]
            rawdiff=max(rawdiff,float(np.abs(p-q).max()));flips+=int((p.argmax(1)!=q.argmax(1)).sum())
        api_diff=0.
        for lo in range(0,len(representative),512):
            ix=np.array(representative[lo:lo+512]);p=core.predict_records(model,[records[j] for j in ix])
            api_diff=max(api_diff,float(np.abs(p-orig[expected[ix]]).max()))
        assert rawdiff<=1e-10 and api_diff<=1e-10 and flips==0
        assert joblib.hash(model['baseline'])==baseline_state
        if stress:
            y=d.label_index.to_numpy();p=prob[ids];b=bp[ids];bc=b.argmax(1)==y;pc=p.argmax(1)==y;asa=d.route.to_numpy()=='asa'
            bm=metrics(y,b);pm=metrics(y,p);tails=[]
            for key,ix in d.groupby(['route','label_index']).indices.items():
                ix=np.array(ix)
                if len(ix)<=100 or d.iloc[ix].body_group.nunique()<=30:tails.append(not (bc[ix]&~pc[ix]).any())
            checks={'total_errors_nonworse':pm['errors']<=bm['errors'],'ASA_errors_nonworse':int((asa&~pc).sum())<=int((asa&~bc).sum()),
                'each_class_recall_nonworse':all(int(((y==k)&pc).sum())>=int(((y==k)&bc).sum()) for k in [0,1,2]),
                'small_slice_previous_correct_preserved':all(tails),'outside_exact':np.array_equal(p[~eligible[ids]],b[~eligible[ids]])}
            stress_review={'checks':checks,'restricted_candidate_passed':all(checks.values()),'baseline':bm,'candidate':pm,
                'fresh_blind_test':False,'deployment_accepted':False,'scope':'Previously inspected stress only; no new environment guarantee.'}
        item={'view':view,'fold':fold,'stress':stress,'evaluation_rows':len(d),'evaluation_probability_max_difference':diff,
            'context_support_reconstructed_from_fit_only':True,'baseline_state_exact':True,'outside_probabilities_exact':True,
            'benign_probability_exact':True,'binary_decision_exact':True,'feature_fallback_attacks':dict(attack_counts),
            'raw_counterfactual_rows':len(records),'raw_probability_max_difference':rawdiff,'prediction_flips':flips,
            'public_api_rows':len(representative),'public_api_probability_max_difference':api_diff,'all_metrics_match':True}
        results.append(item);print(json.dumps(item),flush=True);del model,d,prob,bp;gc.collect()
    out.mkdir(parents=True)
    if stress_review is not None:save(out/'stress_review.json',stress_review)
    save(out/'verification.json',{'all_checks_passed':True,'model_checks':results,'verified_files':verified,'raw_original_cases':len(cases),
        'counterfactual_counts':dict(counts),'new_primary_fits':3,'fixed_control_fits':0,'new_stress_fits':len(stresspaths),
        'primary_selected':selection['selected_view'],'stress_review':stress_review,'quality_accepted':False,
        'scope':'Full stored-model replay, fit-only support, exact baseline protection and bounded perturbations; no new blind or migration validation.',
        'seconds':time.perf_counter()-start,'script_sha256':sha(__file__)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);main(p.parse_args())
