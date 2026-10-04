"""A bounded missingness-mechanism experiment; no automatic model promotion.

Z: one additive classifier with only observed features, no missing categories.
R: four fixed feature-subset classifiers, each fit on ALL eligible fit rows.
Classification thresholds and all labels/multiplicities are held unchanged.
"""
import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline

GROUPS = {'src': ['src_port_fixed', 'src_port_range'],
          'dst': ['dst_port_fixed', 'dst_port_range']}
VIEWS = {'full': [], 'no_src': GROUPS['src'], 'no_dst': GROUPS['dst'],
         'no_ports': GROUPS['src'] + GROUPS['dst']}
SCENARIOS = {'original': [], 'hide_source': GROUPS['src'],
             'hide_destination': GROUPS['dst'], 'hide_both': VIEWS['no_ports']}


def reference(folder):
    p = folder / 'v44_probe_reference.py'
    spec = importlib.util.spec_from_file_location('v44_reference', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def records(x, view, missing='__UNOBSERVED__'):
    excluded = set(VIEWS[view])
    return [{k: str(v) for k, v in row.items() if k not in excluded and v != missing}
            for row in x.to_dict(orient='records')]


def routing(x):
    src = x.src_port_fixed.to_numpy() != '__UNOBSERVED__'
    dst = x.dst_port_fixed.to_numpy() != '__UNOBSERVED__'
    return np.where(src, np.where(dst, 'full', 'no_dst'), np.where(dst, 'no_src', 'no_ports'))


def predict_reduced(models, x):
    route = routing(x); q = np.empty(len(x))
    for view in VIEWS:
        ix = route == view
        if ix.any(): q[ix] = models[view].predict_proba(records(x.loc[ix], view))[:, 1]
    return q


def metrics(y, q):
    pred = np.where(q >= .5, 1, 2); wrong = pred != y
    conf = np.maximum(q, 1-q)
    both = len(np.unique(y)) == 2
    return {'rows': len(y), 'errors': int(wrong.sum()),
            'confusion_M_S': confusion_matrix(y, pred, labels=[1, 2]).tolist(),
            'conditional_log_loss': float(log_loss(y == 1, np.c_[1-q, q], labels=[False, True])),
            'brier': float(np.mean((q-(y==1))**2)),
            'auc_M': float(roc_auc_score(y==1,q)) if both else None,
            'average_precision_M': float(average_precision_score(y==1,q)) if both else None,
            'average_precision_S': float(average_precision_score(y==2,1-q)) if both else None,
            'score_confidence_at_least_95_rows': int((conf>=.95).sum()),
            'score_confidence_at_least_95_wrong': int(((conf>=.95)&wrong).sum()),
            'score_confidence_at_least_99_wrong': int(((conf>=.99)&wrong).sum())}


def prepare(root, out):
    assert not out.exists()
    old = root / 'artifacts/v44_ready_methods_20260914'
    spec=importlib.util.spec_from_file_location('v44_ref',old/'probe_v44_ready_models.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    c=json.loads((old/'configuration.json').read_text(encoding='utf-8'))
    rp=json.loads((old/'report.json').read_text(encoding='utf-8'))
    bindings=dict(c['source_bindings'])
    for p,h in bindings.items(): assert m.sha(root/p)==h,p
    for p,h in rp['model_sha256'].items():
        assert m.sha(old/p)==h
        bindings[(old/p).relative_to(root).as_posix()]=h
    for n in ['evaluation.parquet','report.json','analysis.json','configuration.json','probe_v44_ready_models.py']:
        bindings[(old/n).relative_to(root).as_posix()]=m.sha(old/n)
    r,x,_,_=m.data(root)
    out.mkdir(parents=True)
    shutil.copyfile(__file__,out/Path(__file__).name)
    shutil.copyfile(old/'probe_v44_ready_models.py',out/'v44_probe_reference.py')
    config={'version':'v45-observed-and-reduced-logistic-1','lr':c['lr'], 'fields':m.FIELDS,
            'views':VIEWS,'scenarios':SCENARIOS,'rows':len(r),'class_counts_M_S':[int((r.label_index==k).sum()) for k in [1,2]],
            'folds':'Reuse original body-isolated development folds. Already inspected, not a blind/source transfer test.',
            'training':'Every view uses every ASA fit row, once, with original label and multiplicity. No mask-based row filtering, no imputation or augmentation.',
            'routing':'Only actual readability of source/destination port selects the fixed view, never label, identity, timestamp, product, missing-marker identity or fitted support.',
            'Z':'Full observed-only additive model; removes explicit missing categories but absence remains indirectly inferable.',
            'R':'Full/no-source/no-destination/no-ports observed-only additive models; no per-pattern natural-missingness-only training.',
            'threshold':.5,'hyperparameter_search':False,'new_fits_planned':12,
            'predeclared_quality':'No automatic promotion. Natural-input M/S errors and unseen-input errors must not increase against B, plus both information-loss stresses must improve against old LR without hidden high-confidence regressions. Report all failures.',
            'source_bindings':bindings,'source_sha256':m.sha(__file__),
            'reference_sha256':m.sha(out/'v44_probe_reference.py'),'packages':c['packages'],
            'limits':['Two port groups only; not a general missing-data solver.',
                      'Retained fields can already be missing in training; paper experiments are not reproduced verbatim.',
                      'Different target missingness may change conditional labels; cannot guarantee all mechanisms simultaneously.']}
    m.save(out/'configuration.json',config)
    m.save(out/'preregistered.json',{'configuration_sha256':m.sha(out/'configuration.json'),
        'script_sha256':m.sha(out/Path(__file__).name),'new_models_fit':0})
    print(json.dumps({'prepared':str(out),'new_fits':0,'planned':12}),flush=True)


def fit(root,out):
    m=reference(out);start=time.monotonic()
    config=json.loads((out/'configuration.json').read_text(encoding='utf-8'))
    pre=json.loads((out/'preregistered.json').read_text(encoding='utf-8'))
    assert m.sha(__file__)==pre['script_sha256']
    assert m.sha(out/'configuration.json')==pre['configuration_sha256']
    assert m.sha(out/'v44_probe_reference.py')==config['reference_sha256']
    for p,h in config['source_bindings'].items(): assert m.sha(root/p)==h,p
    assert not (out/'started.json').exists()
    m.save(out/'started.json',{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'config':pre['configuration_sha256']})
    r,x,full_b,ab=m.data(root)
    old=root/'artifacts/v44_ready_methods_20260914'
    old_e=pd.read_parquet(old/'evaluation.parquet')
    assert np.array_equal(old_e.row_position,r.row_position)
    y=r.label_index.to_numpy()
    results=[];fit_log=[];coefficients=[];model_hashes={}
    for fold in range(3):
        tr=r.fold.to_numpy()!=fold;ev=~tr
        folder=out/('fold_%s'%fold);folder.mkdir()
        models={}
        for view in VIEWS:
            t=time.monotonic()
            model=make_pipeline(DictVectorizer(sparse=True),LogisticRegression(**config['lr']))
            model.fit(records(x.loc[tr],view),y[tr]==1)
            assert model[-1].n_iter_.max()<config['lr']['max_iter'],'LR did not converge'
            names=model[0].get_feature_names_out()
            assert not any(m.MISSING in s for s in names)
            assert not any(s.startswith(k+'=') for k in VIEWS[view] for s in names)
            p=folder/(view+'.joblib');joblib.dump(model,p);models[view]=joblib.load(p)
            assert np.array_equal(model.predict_proba(records(x.loc[ev],view)),models[view].predict_proba(records(x.loc[ev],view)))
            model_hashes[p.relative_to(out).as_posix()]=m.sha(p)
            fit_log.append({'fold':fold,'view':view,'fit_rows':int(tr.sum()),'features':len(names),'iterations':int(model[-1].n_iter_.max()),'seconds':time.monotonic()-t})
        lr=joblib.load(old/('fold_%s/LR.joblib'%fold))
        cat=CatBoostClassifier().load_model(old/('fold_%s/CAT.cbm'%fold))
        names=lr[0].get_feature_names_out();coef=dict(zip(names,lr[-1].coef_[0]))
        coefficients.append({'fold':fold,'missing_category_coefficients':{k:float(coef.get(k+'_'+m.MISSING,0)) for k in m.FIELDS},
                             'interpretation':'Coding-specific coefficients, not causal effects. Actual score deltas also remove the observed value terms.'})
        for scenario,columns in SCENARIOS.items():
            xx=x.loc[ev].copy();xx[columns]=m.MISSING
            d=r.loc[ev,['row_position','fold','body_group','projection_id','label_index']].copy()
            d['scenario']=scenario
            d['eligible_stress_row']=True if not columns else (
                (x.loc[ev,'src_port_fixed']!=m.MISSING).to_numpy() if scenario=='hide_source' else
                (x.loc[ev,'dst_port_fixed']!=m.MISSING).to_numpy() if scenario=='hide_destination' else
                ((x.loc[ev,'src_port_fixed']!=m.MISSING)|(x.loc[ev,'dst_port_fixed']!=m.MISSING)).to_numpy())
            d['reduced_view']=routing(xx)
            d['old_LR_qM']=lr.predict_proba(xx)[:,1]
            d['old_CAT_qM']=cat.predict_proba(xx)[:,1]
            d['Z_qM']=models['full'].predict_proba(records(xx,'full'))[:,1]
            d['R_qM']=predict_reduced(models,xx)
            if scenario=='original':
                for n,old_n in [('old_LR','LR'),('old_CAT','CAT')]:
                    assert np.array_equal(d[n+'_qM'],old_e.loc[ev,old_n+'_qM'])
            results.append(d)
        print(json.dumps({'fold':fold,'fits_done':len(fit_log),'new_original':{n:metrics(y[ev],results[-4][n+'_qM'].to_numpy()) for n in ['Z','R']}}),flush=True)
    e=pd.concat(results).sort_values(['scenario','row_position']).reset_index(drop=True)
    assert len(e)==4*len(r)
    e.to_parquet(out/'evaluation.parquet',index=False)
    report={}
    for scenario in SCENARIOS:
        d=e[(e.scenario==scenario)&e.eligible_stress_row]
        report[scenario]={n:metrics(d.label_index.to_numpy(),d[n+'_qM'].to_numpy()) for n in ['old_LR','old_CAT','Z','R']}
    orig=e[e.scenario=='original'].reset_index(drop=True)
    assert np.array_equal(orig.row_position,r.row_position)
    baseline=ab[['p_benign','p_malicious','p_suspicious']].to_numpy()
    bp=baseline.argmax(1)
    slices={}
    for scope,ix in [('unseen_original_key',~old_e.seen_fit_key.to_numpy()),
                     ('natural_src_missing',(x.src_port_fixed==m.MISSING).to_numpy()),
                     ('natural_dst_missing',(x.dst_port_fixed==m.MISSING).to_numpy()),
                     ('mixed_evaluation_input',old_e.mixed_eval_key_diagnostic_only.to_numpy()),
                     ('pure_evaluation_input',~old_e.mixed_eval_key_diagnostic_only.to_numpy())]:
        slices[scope]={'rows':int(ix.sum()),'B_errors':int((bp[ix]!=y[ix]).sum())}
        for n in ['old_LR','old_CAT','Z','R']:
            slices[scope][n]=metrics(y[ix],orig.loc[ix,n+'_qM'].to_numpy())
    gates={}
    old_primary=json.loads((old/'report.json').read_text())
    for n in ['Z','R']:
        mm=report['original'][n]['confusion_M_S']
        q=orig[n+'_qM'].to_numpy();pp=np.where(q>=.5,1,2)
        gates[n]={'M_errors_not_increased_vs_B':mm[0][1]<=618,'S_errors_not_increased_vs_B':mm[1][0]<=5400,
                  'unseen_errors_not_increased_vs_B':slices['unseen_original_key'][n]['errors']<=1096,
                  'source_stress_improved_vs_old_LR':report['hide_source'][n]['errors']<report['hide_source']['old_LR']['errors'],
                  'destination_stress_improved_vs_old_LR':report['hide_destination'][n]['errors']<report['hide_destination']['old_LR']['errors'],
                  'source_high_confidence_wrong_not_increased':report['hide_source'][n]['score_confidence_at_least_95_wrong']<=report['hide_source']['old_LR']['score_confidence_at_least_95_wrong'],
                  'destination_high_confidence_wrong_not_increased':report['hide_destination'][n]['score_confidence_at_least_95_wrong']<=report['hide_destination']['old_LR']['score_confidence_at_least_95_wrong']}
        effects={'fixes':int(((bp!=y)&(pp==y)).sum()),'regressions':int(((bp==y)&(pp!=y)).sum())}
        report['original'][n].update(effects)
    m.save(out/'report.json',{'status':'development_mechanism_probe_not_promoted','new_model_fits':12,
            'scenario_metrics':report,'natural_input_slices':slices,'quality_gates':gates,
            'all_gates_pass':{n:all(g.values()) for n,g in gates.items()},'fit_log':fit_log,
            'model_sha256':model_hashes,'source_bindings':config['source_bindings'],
            'elapsed_seconds':time.monotonic()-start,'limits':config['limits']+[
                'No threshold changes, independent calibration or external deployment.',
                'A high model score is not established real-world confidence.',
                'Synthetic removal changes evidence; equal predictions are not required.']})
    m.save(out/'old_LR_missing_coefficients.json',coefficients)
    print(json.dumps({'done':True,'seconds':time.monotonic()-start,'metrics':report,'gates':gates}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit']);p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();(prepare if a.mode=='prepare' else fit)(Path(a.root).resolve(),Path(a.out).resolve())
