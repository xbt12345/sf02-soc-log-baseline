"""Saved-model replay and finite-sample information/support diagnostics."""
import argparse
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main(root,out):
    assert not (out/'verification.json').exists()
    spec=importlib.util.spec_from_file_location('v45',out/'probe_v45_missingness.py')
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);m=v.reference(out)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    config=read(out/'configuration.json');rep=read(out/'report.json');pre=read(out/'preregistered.json')
    assert m.sha(out/'configuration.json')==pre['configuration_sha256']
    assert m.sha(out/'probe_v45_missingness.py')==pre['script_sha256']
    for p,h in config['source_bindings'].items(): assert m.sha(root/p)==h,p
    r,x,b,ab=m.data(root);e=pd.read_parquet(out/'evaluation.parquet');models={};replayed=0
    for fold in range(3):
        ix=r.fold.to_numpy()==fold
        models[fold]={}
        for view in v.VIEWS:
            p=out/('fold_%s/%s.joblib'%(fold,view));assert m.sha(p)==rep['model_sha256'][p.relative_to(out).as_posix()]
            model=joblib.load(p);models[fold][view]=model
            fields={s.split('=',1)[0] for s in model[0].get_feature_names_out()}
            assert fields<=set(m.FIELDS) and not (fields&set(v.VIEWS[view]))
            assert not any(m.MISSING in s for s in model[0].get_feature_names_out())
        for scenario,cols in v.SCENARIOS.items():
            xx=x.loc[ix].copy();xx[cols]=m.MISSING
            d=e[(e.fold==fold)&(e.scenario==scenario)].sort_values('row_position')
            assert np.array_equal(d.row_position,r.loc[ix,'row_position'])
            assert np.array_equal(d.label_index,r.loc[ix,'label_index'])
            pred={'Z':models[fold]['full'].predict_proba(v.records(xx,'full'))[:,1],
                  'R':v.predict_reduced(models[fold],xx)}
            assert np.array_equal(v.routing(xx),d.reduced_view)
            for n,q in pred.items():
                assert np.array_equal(q,d[n+'_qM'].to_numpy());replayed+=len(d)
                use=d.eligible_stress_row.to_numpy()
                # Fold reports are recomputed independently below in aggregate.
                assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    floors={};calibration={}
    for scenario,cols in v.SCENARIOS.items():
        d=e[(e.scenario==scenario)&e.eligible_stress_row].copy()
        xx=x.copy();xx[cols]=m.MISSING
        keys=np.array([json.dumps(a,separators=(',',':')) for a in xx.to_numpy().tolist()])
        indices=pd.Index(r.row_position).get_indexer(d.row_position)
        d['key']=keys[indices]
        counts=collections.defaultdict(lambda:collections.Counter())
        for fold,key,label in zip(d.fold,d.key,d.label_index): counts[(fold,key)][int(label)]+=1
        minimum=sum(min(c.get(1,0),c.get(2,0)) for c in counts.values())
        wrong=np.where(d.R_qM>=.5,1,2)!=d.label_index.to_numpy()
        mixed=np.array([len(counts[(f,k)])==2 for f,k in zip(d.fold,d.key)])
        pred_check=pd.DataFrame({'fold':d.fold,'key':d.key,'prediction':np.where(d.R_qM>=.5,1,2)})
        assert pred_check.groupby(['fold','key']).prediction.nunique().max()==1
        floors[scenario]={'rows':len(d),'fold_specific_minimum_errors':minimum,'R_errors':int(wrong.sum()),
            'R_mixed_key_errors':int((wrong&mixed).sum()),'R_pure_key_errors':int((wrong&~mixed).sum()),
            'limit':'Uses inspected evaluation labels; not population Bayes error, prospective score or guaranteed learnable improvement.'}
        calibration[scenario]={}
        for n in ['old_LR','old_CAT','Z','R']:
            yy=d.label_index.to_numpy();q=d[n+'_qM'].to_numpy();cf=np.maximum(q,1-q);correct=np.where(q>=.5,1,2)==yy
            recomputed=v.metrics(yy,q)
            for k,z in recomputed.items():
                expected=rep['scenario_metrics'][scenario][n][k]
                if isinstance(z,float):assert abs(z-expected)<1e-12
                else:assert z==expected
            points=[]
            for threshold in [.5,.6,.7,.8,.9,.95,.99,.995]:
                use=cf>=threshold
                points.append({'min_score_confidence':threshold,'rows':int(use.sum()),'coverage':float(use.mean()),
                    'error_rate':float((~correct[use]).mean()) if use.any() else None,
                    'mean_score_confidence':float(cf[use].mean()) if use.any() else None})
            calibration[scenario][n]=points
    # Diagnose the 639 residual pure-key destination-removal errors using only
    # each candidate view's own fit-side features for support counts.
    residual=[];target=e[(e.scenario=='hide_destination')&e.eligible_stress_row]
    for fold in range(3):
        for view in ['no_dst','no_ports']:
            d=target[(target.fold==fold)&(target.reduced_view==view)].copy()
            xx=x.copy();xx[v.VIEWS[view]]=m.MISSING
            keys=np.array([json.dumps(a,separators=(',',':')) for a in xx.to_numpy().tolist()])
            ix=r.fold.to_numpy()!=fold
            fit=pd.DataFrame({'key':keys[ix],'label':r.label_index.to_numpy()[ix],'body':r.body_group.to_numpy()[ix]})
            fc=pd.crosstab(fit.key,fit.label).reindex(columns=[1,2],fill_value=0)
            d['key']=keys[pd.Index(r.row_position).get_indexer(d.row_position)]
            ec=pd.crosstab(d.key,d.label_index).reindex(columns=[1,2],fill_value=0)
            d['wrong']=np.where(d.R_qM>=.5,1,2)!=d.label_index
            for key,grp in d[d.wrong].groupby('key'):
                if (ec.loc[key]>0).all():continue
                f=fc.reindex([key],fill_value=0).to_numpy()[0]
                label=int(grp.label_index.iloc[0])
                support='unseen' if not f.sum() else 'both_labels' if (f>0).all() else 'fit_pure_same_label' if f[label-1]>0 else 'fit_pure_opposite_label'
                residual.append({'fold':fold,'view':view,'errors':len(grp),'body_keys':int(grp.body_group.nunique()),
                    'row_positions':grp.row_position.astype(int).tolist(),'evaluation_counts_M_S':ec.loc[key].tolist(),
                    'fit_counts_M_S':f.tolist(),'fit_support':support,'qM':float(grp.R_qM.iloc[0]),
                    'facts':dict(zip(m.FIELDS,json.loads(key)))})
    assert sum(z['errors'] for z in residual)==639
    support=collections.Counter()
    for z in residual:support[z['fit_support']]+=z['errors']
    m.save(out/'remaining_destination_errors.json',{'scope':'Label-aware diagnostic only, not inference rules. No labels used for training or modification here.',
        'errors':639,'fold_key_groups':len(residual),'support_error_counts':dict(support),'groups':residual})
    # Reparameterize OLD full one-hot LR without a missing column: this exactly
    # reproduces its scores on values in the fit vocabulary. New unseen values
    # are a different case; we deliberately verify on fit rows only.
    algebra=[]
    for fold in range(3):
        old=joblib.load(root/('artifacts/v44_ready_methods_20260914/fold_%s/LR.joblib'%fold))
        xx=x.loc[r.fold.to_numpy()!=fold]
        names=old[0].get_feature_names_out();co=dict(zip(names,old[-1].coef_[0]))
        bias=float(old[-1].intercept_[0])+sum(float(co.get(k+'_'+m.MISSING,0)) for k in m.FIELDS)
        score=np.full(len(xx),bias)
        for k in m.FIELDS:
            missing=float(co.get(k+'_'+m.MISSING,0))
            score+=np.array([0 if a==m.MISSING else float(co[k+'_'+a])-missing for a in xx[k]])
        diff=float(np.max(np.abs(score-old.decision_function(xx))))
        assert diff<1e-10
        algebra.append({'fold':fold,'known_vocabulary_rows':len(xx),'max_margin_difference':diff})
    # Raw path: original messages -> frozen audited parser -> new observed-only
    # dictionaries and view selection, including nuisance and masking cases.
    frozen=root/'artifacts/v42_local_r1_20260914/frozen_training_runtime'
    frec=read(frozen.parent/'prepared/complete.json')
    for n,h in frec['runtime_sources'].items():assert m.sha(frozen/n)==h
    sys.path.insert(0,str(frozen));import v39_core as core
    from audit_v37_prepared import variants
    official=root/'data/official/train.parquet'
    assert m.sha(official)=='6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
    representatives=r.drop_duplicates('projection_id')
    selected=set(representatives.iloc[::16].row_position.astype(int))|{int(g['row_positions'][0]) for g in residual}
    pos=np.array(sorted(selected));raws={};offset=0
    for batch in pq.ParquetFile(official).iter_batches(batch_size=8192,columns=['message_sanitized'],use_threads=False):
        wanted=pos[np.searchsorted(pos,offset):np.searchsorted(pos,offset+len(batch))]
        for a in wanted:raws[int(a)]=batch.column(0)[int(a)-offset].as_py() or ''
        offset+=len(batch)
    original=[];changed=[];parents=[];variant_counts=collections.Counter()
    xindex=pd.Index(r.row_position)
    for index,(pos,raw) in enumerate(sorted(raws.items())):
        base=core.prepare_record({'message_sanitized':raw});enc=m.encode(base['facts'])
        assert enc==x.iloc[xindex.get_loc(pos)].tolist();original.append(enc)
        changes=[('metadata',{'message_sanitized':raw,'timestamp':'2099-12-01T00:00:00Z','product_name':None,
                 'vendor_name':'new','src_ip':'192.0.2.99','dst_ip':'203.0.113.9','username':'new','label_binary':'benign'})]
        changes += [(n,{'message_sanitized':text}) for n,text in variants(raw,'asa')]
        text=re.sub(r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)','203.0.113.99',raw)
        text=re.sub(r'dmz[-_]\d+','dmz-999',text)
        changes.append(('address_interface',{'message_sanitized':text}))
        text=re.sub(r'(?:CRED|USER|ORG|HOST|IP|EMAIL)-\d+','CRED-999999',raw)
        if text!=raw:changes.append(('redaction_identity',{'message_sanitized':text}))
        for kind,record in changes:
            p=core.prepare_record(record);q=m.encode(p['facts'])
            assert p['text']==base['text'] and q==enc,(pos,kind)
            changed.append(q);parents.append(index);variant_counts[kind]+=1
    a=pd.DataFrame(original,columns=m.FIELDS);z=pd.DataFrame(changed,columns=m.FIELDS);parent=np.array(parents)
    raw_checks=[]
    for fold,mm in models.items():
        for scenario,cols in v.SCENARIOS.items():
            aa=a.copy();zz=z.copy();aa[cols]=m.MISSING;zz[cols]=m.MISSING
            for name in ['Z','R']:
                func=lambda df: mm['full'].predict_proba(v.records(df,'full'))[:,1] if name=='Z' else v.predict_reduced(mm,df)
                diff=float(np.max(np.abs(func(aa)[parent]-func(zz))))
                assert diff==0
                raw_checks.append({'fold':fold,'scenario':scenario,'candidate':name,'variant_rows':len(z),'max_probability_difference':diff})
    orig=e[e.scenario=='original'].sort_values('row_position');bp=b[['p_benign','p_malicious','p_suspicious']].to_numpy();inside=b.route.to_numpy()=='asa'
    guards={}
    for name in ['Z','R']:
        p=bp.copy();q=orig[name+'_qM'].to_numpy();part=p[inside].copy();mass=part[:,1]+part[:,2]
        part[:,1]=mass*q;part[:,2]=mass*(1-q);fallback=(bp[inside].argmax(1)==0)|(part.argmax(1)==0)
        part[fallback]=bp[inside][fallback];p[inside]=part
        assert np.array_equal(p[~inside],bp[~inside]) and np.array_equal(p[:,0],bp[:,0])
        assert np.array_equal(p.argmax(1)==0,bp.argmax(1)==0)
        guards[name]={'rows':len(p),'outside_rows':int((~inside).sum()),'outside_probability_changes':0,'normal_boundary_changes':0,'fallback_rows':int(fallback.sum())}
    m.save(out/'information_and_calibration_audit.json',{'floors':floors,'risk_coverage_points':calibration,
        'algebra_reparameterization':algebra,'support_of_639_errors':dict(support),
        'limits':'No new calibration or training. High-confidence error count alone mixes coverage and risk; none of these score summaries is external calibration evidence.'})
    m.save(out/'verification.json',{'all_implementation_checks_passed':True,'quality_passed':False,
        'quality_gates_unchanged':rep['quality_gates'],'replayed_probability_rows':replayed,'new_fits':0,
        'raw_original_rows':len(a),'raw_variants_per_scenario_model':len(z),'raw_variant_types':dict(variant_counts),'raw_checks':raw_checks,
        'guards':guards,'source_bindings_rechecked':config['source_bindings'],'script_sha256':m.sha(__file__),
        'scope':'Previously inspected development data. Conditional heads, raw subset nuisance tests and offline protected probability reconstruction; not production integration or migration acceptance.'})
    print(json.dumps({'replayed':replayed,'raw_sources':len(a),'variants':len(z),'residual_support':dict(support),'floors':floors,'quality_passed':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(Path(a.root).resolve(),Path(a.out).resolve())
