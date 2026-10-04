"""Freeze, fit one bounded offset per fold, and review predeclared gates."""
import argparse
import collections
import datetime
import gc
import json
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import v42_core as core
from run_v39_prepare import sha,save
from run_v40_train import metrics

PROBS=['p_benign','p_malicious','p_suspicious']
CONFIG={'version':core.VERSION,'candidate':'R','fixed_control':'G','new_primary_fits':3,'new_stress_fit_limit':1,
 'minimum_distinct_body_keys':3,'l2':10.0,'maximum_conditional_odds_ratio_change':2.0,
 'scope':'Complete observed semantic body and facts; no labels in support eligibility',
 'weights':'Exact original fit-side row multiplicity, no repeat/class weighting',
 'data_roles':'Inherited v39 final body folds; no old outer evaluation rows in fitting',
 'decision':'three-class argmax, benign probability and binary benign decision exactly preserved',
 'ASA_errors_limit_exclusive':6018,'ASA_class_error_limits':[0,618,5400],
 'ASA_unseen_error_limits':{'unseen_input':1096,'unseen_parameter':652,'unseen_joint_value':740},
 'minimum_improved_folds':2,'minimum_fixed_body_keys':3,'small_class_rows':100,'small_class_body_keys':30,
 'fresh_blind_test':False,'probability_tolerance':1e-10,'selection':'Only R; all predefined gates; G diagnostic only.'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def data(prep):
    r=pq.read_table(prep/'rows.parquet').to_pandas();r=r[r.fold>=0].reset_index(drop=True)
    active,ids=np.unique(r.projection_id.to_numpy(),return_inverse=True)
    p=pq.read_table(prep/'projections.parquet').to_pandas().iloc[active].reset_index(drop=True)
    return r,ids,p.text.tolist(),[json.loads(v) for v in p.facts]


def prepare(root,run):
    assert not run.exists();old=root/'artifacts/v41_local_r1_20260913';prep=root/'artifacts/v39_local_r2_20260913/prepared'
    parent=read(old/'prepared/complete.json');baseline=read(prep/'complete.json')
    assert sha(root/'data/official/train.parquet')==baseline['official_sha256']
    for n,h in baseline['files'].items():assert sha(prep/n)==h
    for n,h in parent['runtime_sources'].items():assert sha(old/'frozen_training_runtime'/n)==h
    run.mkdir(parents=True);stage=run/'prepared';stage.mkdir();runtime=run/'frozen_training_runtime';runtime.mkdir()
    for p in (old/'frozen_training_runtime').glob('*.py'):shutil.copyfile(p,runtime/p.name)
    for n in ['v42_core.py','run_v42.py','test_v42.py','verify_v42.py']:shutil.copyfile(Path(__file__).parent/n,runtime/n)
    shutil.copyfile(root/'docs/V42_PLAN.md',stage/'approved_plan.md')
    shutil.copyfile(root/'evidence/2026-09-14/v42_preflight/coverage.json',stage/'coverage.json')
    shutil.copyfile(root/'evidence/2026-09-14/v42_preflight/sources.json',stage/'sources.json')
    # Preflight preceded the inference wrapper addition; confirm the core
    # function support semantics by recomputing every fold before fitting.
    rows,ids,texts,facts=data(prep);assert len(rows)==1378650 and rows.body_group.nunique()==508823
    assert rows.groupby('body_group').fold.nunique().max()==1
    results=[];coverage=read(stage/'coverage.json')['folds']
    for f in range(3):
        fit=rows.fold.to_numpy()!=f;s=core.ContextSupport().fit(texts,facts,ids[fit],rows.body_group.to_numpy()[fit]);ctx,_=s.encode(texts,facts)
        n=int(((ctx[ids]>=0)&~fit).sum());assert n==coverage[f]['evaluation_supported_rows']
        results.append({'fold':f,'eligible_contexts':len(s.index),'evaluation_supported_rows':n})
    meta_path=root/'artifacts/v37_prepared_r13_20260913/prepared.parquet';assert sha(meta_path)==parent['v37_raw_hash_metadata_sha256']
    m=pq.read_table(meta_path,columns=['row_position','raw_hash']).to_pandas().iloc[rows.row_position.to_numpy()].copy()
    assert np.array_equal(m.row_position,rows.row_position);m['fold']=rows.fold.to_numpy();m['empty']=rows.original_empty.to_numpy()
    assert not (m.loc[~m['empty']].groupby('raw_hash').fold.nunique()>1).any()
    bindings={}
    for f in range(3):
        for base,name in [(prep.parent,'SEMANTIC'),(old,'P')]:
            folder=base/('primary/fold_%s/%s'%(f,name));receipt=read(folder/'complete.json')
            for n,k in [('model.joblib','model_sha256'),('evaluation.parquet','predictions_sha256')]:
                assert sha(folder/n)==receipt[k];bindings[(folder/n).relative_to(root).as_posix()]=receipt[k]
    save(stage/'configuration.json',CONFIG)
    save(stage/'input_checks.json',{'all_checks_passed':True,'rows':len(rows),'body_keys':int(rows.body_group.nunique()),
        'body_cross_fold_overlap':0,'nonempty_raw_cross_fold_overlap':0,'coverage_recomputed':results,'source_bindings':bindings,
        'near_copy_note':'Same frozen body protocol; prior reviewed near-copy ambiguity persists; no new incident independence claim.'})
    save(stage/'complete.json',{'version':core.VERSION,'official_sha256':baseline['official_sha256'],'v39_prepared':str(prep),
        'v39_prepared_receipt_sha256':sha(prep/'complete.json'),'parent_v41_receipt_sha256':sha(old/'prepared/complete.json'),
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{p.name:sha(p) for p in stage.iterdir() if p.is_file()},
        'runtime_sources':{p.name:sha(p) for p in runtime.glob('*.py')}})
    print(json.dumps({'prepared':str(run),'primary_fits_planned':3,'coverage':results}),flush=True)


def fit(root,run,fold,stress=False):
    start=time.perf_counter();stage=run/'prepared';receipt=read(stage/'complete.json');conf=read(stage/'configuration.json')
    for n,h in receipt['files'].items():assert sha(stage/n)==h
    for n,h in receipt['runtime_sources'].items():assert sha(Path(__file__).parent/n)==h
    prep=Path(receipt['v39_prepared']);assert sha(prep/'complete.json')==receipt['v39_prepared_receipt_sha256']
    if stress:
        selection=read(run/'primary_review/selection.json');assert selection['proceed_to_stress'] and selection['selected_view']=='R'
        out=run/'old_protocol_stress';reference=prep.parent/'old_protocol_stress'
    else:out=run/('primary/fold_%s'%fold);reference=prep.parent/('primary/fold_%s/SEMANTIC'%fold)
    assert not out.exists();out.mkdir(parents=True)
    rows,ids,texts,facts=data(prep);y=rows.label_index.to_numpy(dtype=int)
    fitmask=(rows.inner_role.to_numpy()!=2) if stress else (rows.fold.to_numpy()!=fold);ev=~fitmask
    if stress:
        assert fitmask.sum()==1199575 and ev.sum()==179075
        assert not set(rows.loc[fitmask,'body_group'])&set(rows.loc[ev,'body_group'])
        assert not set(rows.loc[fitmask,'union_group'])&set(rows.loc[ev,'union_group'])
    ref_receipt=read(reference/'complete.json');assert sha(reference/'model.joblib')==ref_receipt['model_sha256']
    assert sha(reference/'evaluation.parquet')==ref_receipt['predictions_sha256']
    base=joblib.load(reference/'model.joblib');frozen_state=joblib.hash(base)
    xb=core.old_parameters.matrix(base,texts,facts);bp=base['model'].predict_proba(xb)
    ref=pq.read_table(reference/'evaluation.parquet').to_pandas();assert np.array_equal(ref.row_position,rows.row_position.to_numpy()[ev])
    assert np.array_equal(ref.label_index,y[ev]);assert np.max(np.abs(ref[PROBS].to_numpy()-bp[ids[ev]]))<1e-10
    support=core.ContextSupport().fit(texts,facts,ids[fitmask],rows.body_group.to_numpy()[fitmask],conf['minimum_distinct_body_keys'])
    ctx,reasons=support.encode(texts,facts);eligible=ctx>=0
    model=core.FrozenSubtypeOffset().fit(support,texts,facts,bp,ids[fitmask],y[fitmask],conf['l2'])
    predicted=model.predict(texts,facts,bp)[0]
    assert joblib.hash(base)==frozen_state
    predictions={'R':predicted};bundles={'R':{'version':core.VERSION,'view':'R','baseline':base,'offset':model}}
    if not stress:
        oldp=root/('artifacts/v41_local_r1_20260913/primary/fold_%s/P'%fold);pr=read(oldp/'complete.json')
        assert sha(oldp/'model.joblib')==pr['model_sha256'];proposal=joblib.load(oldp/'model.joblib')
        pp=proposal['model'].predict_proba(core.old_parameters.matrix(proposal,texts,facts))
        saved=pq.read_table(oldp/'evaluation.parquet').to_pandas();assert np.array_equal(saved.row_position,ref.row_position)
        assert np.max(np.abs(pp[ids[ev]]-saved[PROBS].to_numpy()))<1e-10
        predictions['G']=core.gated_reference(bp,pp,eligible)
        bundles['G']={'version':core.VERSION,'view':'G','baseline':base,'reference':proposal,'support':support}
    for view,p in predictions.items():
        folder=out if stress else out/view;folder.mkdir(exist_ok=stress)
        joblib.dump(bundles[view],folder/'model.joblib',compress=3)
        d=rows.loc[ev,['row_position','label_index','product','route','body_group','projection_id']].copy()
        for i,n in enumerate(PROBS):d[n]=p[ids[ev],i]
        d['pred_label']=np.array(['benign','malicious','suspicious'])[p[ids[ev]].argmax(1)]
        d['context_eligible']=eligible[ids[ev]];d['context_reason']=np.array(reasons)[ids[ev]]
        d['context_id']=ctx[ids[ev]];d['baseline_availability_mask']=ref.availability_mask.to_numpy()
        if not stress:
            for col in ['baseline_partition','unseen_parameter','unseen_joint_value','unsupported_joint_value']:d[col]=saved[col].to_numpy()
        outside=~d.context_eligible.to_numpy();q=p[ids[ev]];original=bp[ids[ev]]
        assert np.array_equal(q[outside],original[outside]);assert np.array_equal(q[:,0],original[:,0])
        assert np.array_equal(q.argmax(1)==0,original.argmax(1)==0)
        pq.write_table(pa.Table.from_pandas(d,preserve_index=False),folder/'evaluation.parquet',compression='zstd')
        report={'view':view,'fold':None if stress else fold,'evaluation':metrics(y[ev],q),'baseline_evaluation':metrics(y[ev],original),
            'eligible_evaluation':metrics(y[ev][~outside],q[~outside]),'eligible_baseline':metrics(y[ev][~outside],original[~outside]),
            'fixed':int(((original.argmax(1)!=y[ev])&(q.argmax(1)==y[ev])).sum()),'regressed':int(((original.argmax(1)==y[ev])&(q.argmax(1)!=y[ev])).sum()),
            'scope_counts':dict(collections.Counter(d.context_reason)),'optimization':model.optimization if view=='R' else {'new_fit':False},
            'baseline_state_unchanged':joblib.hash(base)==frozen_state,'outside_probabilities_exact':True,'benign_probability_exact':True,'binary_decision_exact':True,
            'seconds_from_fold_start':time.perf_counter()-start,'fresh_blind_test':False,'quality_accepted':False}
        save(folder/'report.json',report)
        save(folder/'binding.json',{'prepared_sha256':sha(stage/'complete.json'),'runtime_sources':receipt['runtime_sources'],
            'reference_model_sha256':ref_receipt['model_sha256'],'baseline_joblib_state_hash':frozen_state,'fit_rows':int(fitmask.sum()),
            'evaluation_rows':int(ev.sum()),'view':view,'fold':None if stress else fold,'stress':stress})
        save(folder/'complete.json',{'model_sha256':sha(folder/'model.joblib'),'predictions_sha256':sha(folder/'evaluation.parquet'),
            'report_sha256':sha(folder/'report.json'),'binding_sha256':sha(folder/'binding.json')})
        print(json.dumps({'view':view,'fold':None if stress else fold,'eligible_rows':int((~outside).sum()),'errors':report['evaluation']['errors'],
            'fixed':report['fixed'],'regressed':report['regressed'],'fit_parameters':model.optimization['context_parameters'],'seconds':round(time.perf_counter()-start)}),flush=True)


def load(folder):
    rec=read(folder/'complete.json')
    for n,k in [('model.joblib','model_sha256'),('evaluation.parquet','predictions_sha256'),('report.json','report_sha256')]:assert sha(folder/n)==rec[k]
    return pq.read_table(folder/'evaluation.parquet').to_pandas()


def review(root,run):
    conf=read(run/'prepared/configuration.json');out=run/'primary_review';assert not out.exists();out.mkdir()
    old=root/'artifacts/v39_local_r2_20260913';b=pd.concat([load(old/('primary/fold_%s/SEMANTIC'%f)) for f in range(3)]).sort_values('row_position').reset_index(drop=True)
    y=b.label_index.to_numpy();bp=b[PROBS].to_numpy();bc=bp.argmax(1)==y;asa=(b.route=='asa').to_numpy();baseline=metrics(y,bp)
    result={'baseline':baseline,'candidates':{},'new_primary_fits':3,'control_fits':0,'fresh_blind_test':False}
    for view in ['R','G']:
        tables=[]
        for f in range(3):d=load(run/('primary/fold_%s/%s'%(f,view)));d['fold']=f;tables.append(d)
        d=pd.concat(tables).sort_values('row_position').reset_index(drop=True);assert np.array_equal(d.row_position,b.row_position) and np.array_equal(d.label_index,y)
        p=d[PROBS].to_numpy();correct=p.argmax(1)==y;eligible=d.context_eligible.to_numpy();fixed=~bc&correct;regressed=bc&~correct;m=metrics(y,p)
        fold_delta=[int((asa&~correct&(d.fold.to_numpy()==f)).sum()-(asa&~bc&(d.fold.to_numpy()==f)).sum()) for f in range(3)]
        slices={'ASA':asa,'eligible':eligible,'outside':~eligible,'ASA_unseen_input':asa&(d.baseline_partition.to_numpy()==3)}
        for n in ['unseen_parameter','unseen_joint_value','unsupported_joint_value']:slices['ASA_'+n]=asa&d[n].to_numpy()
        detail={k:{'rows':int(s.sum()),'body_keys':int(b.loc[s,'body_group'].nunique()),'B_errors':int((s&~bc).sum()),'new_errors':int((s&~correct).sum()),
            'fixed':int((s&fixed).sum()),'regressed':int((s&regressed).sum()),'eligible_rows':int((s&eligible).sum())} for k,s in slices.items()}
        class_errors=[int((asa&(y==k)&~correct).sum()) for k in range(3)]
        tails=[]
        for grouping in [['route','label_index'],['route','availability_mask','label_index']]:
            for key,idx in b.groupby(grouping).indices.items():
                ix=np.asarray(idx);nb=int(b.iloc[ix].body_group.nunique())
                if len(ix)>conf['small_class_rows'] and nb>conf['small_class_body_keys']:continue
                tails.append({'slice':[int(v) if isinstance(v,np.integer) else v for v in key],'rows':len(ix),'body_keys':nb,
                    'B_correct':int(bc[ix].sum()),'new_correct':int(correct[ix].sum()),'regressed':int(regressed[ix].sum()),'new_total_loss':bool(bc[ix].any() and not correct[ix].any())})
        critical=(~asa)|b.row_position.isin([483275,1765291]).to_numpy()
        checks={'outside_exact':np.array_equal(p[~eligible],bp[~eligible]),'benign_probability_exact':np.array_equal(p[:,0],bp[:,0]),
            'binary_decision_exact':np.array_equal(p.argmax(1)==0,bp.argmax(1)==0),'ASA_pooled_improve':detail['ASA']['new_errors']<conf['ASA_errors_limit_exclusive'],
            'ASA_two_folds_improve':sum(v<0 for v in fold_delta)>=conf['minimum_improved_folds'],
            'at_least_three_fixed_body_keys':b.loc[fixed,'body_group'].nunique()>=conf['minimum_fixed_body_keys'],
            'ASA_each_class_nonworse':all(v<=lim for v,lim in zip(class_errors,conf['ASA_class_error_limits'])),
            'no_critical_regression':not (critical&regressed).any(),'no_small_slice_total_loss':not any(v['new_total_loss'] for v in tails),
            'macro_f1_nonworse':m['macro_f1']>=baseline['macro_f1'],'log_loss_nonworse':m['log_loss']<=baseline['log_loss'],
            'normal_false_alerts_nonworse':m['normal_errors']<=baseline['normal_errors']}
        for name,limit in conf['ASA_unseen_error_limits'].items():checks['ASA_'+name+'_guard']=detail['ASA_'+name]['new_errors']<=limit
        checks={k:bool(v) for k,v in checks.items()}
        change=d.loc[fixed|regressed].copy();change['change']=np.where(fixed[fixed|regressed],'fixed','regressed')
        for j,n in enumerate(PROBS):change['B_'+n]=bp[fixed|regressed,j]
        pq.write_table(pa.Table.from_pandas(change,preserve_index=False),out/('changes_%s.parquet'%view),compression='zstd')
        item={'evaluation':m,'fold_ASA_error_changes':fold_delta,'ASA_class_errors':class_errors,'slices':detail,'small_slices':tails,
            'fixed':int(fixed.sum()),'regressed':int(regressed.sum()),'fixed_body_keys':int(b.loc[fixed,'body_group'].nunique()),
            'regressed_body_keys':int(b.loc[regressed,'body_group'].nunique()),'checks':checks,'eligible_for_stress':view=='R' and all(checks.values())}
        save(out/('candidate_%s.json'%view),item);result['candidates'][view]=item
        print(json.dumps({'view':view,'errors':m['errors'],'macro_f1':m['macro_f1'],'fixed':item['fixed'],'regressed':item['regressed'],'checks':checks,'eligible':item['eligible_for_stress']}),flush=True)
    save(out/'primary_review.json',result);proceed=result['candidates']['R']['eligible_for_stress']
    save(out/'selection.json',{'selected_view':'R' if proceed else None,'proceed_to_stress':proceed,'primary_review_sha256':sha(out/'primary_review.json'),
        'configuration_sha256':sha(run/'prepared/configuration.json'),'script_sha256':sha(__file__),'G_diagnostic_only':True,'quality_accepted':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit','review']);p.add_argument('--root',required=True);p.add_argument('--run',required=True)
    p.add_argument('--fold',type=int,choices=[0,1,2],default=0);p.add_argument('--stress',action='store_true');a=p.parse_args();root=Path(a.root).resolve();run=Path(a.run).resolve()
    if a.mode=='prepare':prepare(root,run)
    elif a.mode=='fit':fit(root,run,a.fold,a.stress)
    else:review(root,run)
