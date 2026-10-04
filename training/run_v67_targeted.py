"""Fixed-config, source-held-out controls for the 578 v66 hard-S failures.

The target mask is diagnostic only; it never enters features, loss, or selection.
All fits retain official row multiplicity and labels. No validation early stopping.
"""
import argparse
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits
from train_v65_rank_heads import ROOT, load_fit, measure
from prepare_v61 import CTX_NAMES
from v61_common import FIELDS, read, save, sha

OLD = ROOT/'artifacts/v65_rank_heads_20260920'
AUDIT = ROOT/'artifacts/v66_issue_resolution_20260920'
ARMS = ['event_coarse', 'event_numeric', 'context_coarse', 'context_numeric']
PARAMS = dict(learning_rate=.05, max_iter=300, max_leaf_nodes=15,
              min_samples_leaf=20, l2_regularization=10., early_stopping=False,
              random_state=20260920)


def target_mask(d):
    return (d.label.eq(2) & d.H1_pred.ne(2) & ~d.validation_input_conflict &
            d.behavior.isin(['deny|blocked|tcp|outside|dmz', 'deny|blocked|udp|outside|dmz'])).to_numpy()


def design(frame, context, arm, train, encoder=None):
    # Pure allowlist: identities, timestamps, product, literal zones/policies cannot enter.
    categorical = [f for f in FIELDS if f not in ['src_port_fixed','dst_port_fixed']]
    if encoder is None:
        encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        encoder.fit(frame.iloc[train][categorical])
    x=encoder.transform(frame[categorical]); names=categorical.copy()
    catmask=[True]*len(names)
    if arm.endswith('numeric'):
        # Observed numbers only. No imputation of redacted port aliases.
        for col in ['src_port_fixed','dst_port_fixed']:
            value=pd.to_numeric(frame[col],errors='coerce').to_numpy(dtype=float)
            assert np.all(np.isnan(value)|((value>=0)&(value<=65535)))
            x=np.c_[x,value];names.append(col);catmask.append(False)
    if arm.startswith('context'):
        x=np.c_[x,context['stats']];names+=CTX_NAMES;catmask += [False]*len(CTX_NAMES)
    return x,encoder,names,catmask


def counts(frame, pred, base, target):
    y=frame.label.to_numpy(); result={}
    for c,name in [(0,'B'),(1,'M'),(2,'S')]:
        mask=y==c
        result[name]={'rows':int(mask.sum()),'errors':int((mask&(pred!=y)).sum()),
            'fixed':int((mask&(base!=y)&(pred==y)).sum()),
            'broken':int((mask&(base==y)&(pred!=y)).sum()),
            'source_recall':float(pd.DataFrame({'group':frame.group[mask], 'ok':pred[mask]==y[mask]}).groupby('group').ok.mean().mean()) if mask.any() else None}
    result['target_fixed']=int((target&(pred==2)).sum())
    result['target_remaining']=int((target&(pred!=2)).sum())
    result['outside_target_S_broken']=int(((y==2)&~target&(base==2)&(pred!=2)).sum())
    result['total_errors']=int((pred!=y).sum())
    result['normal_to_threat']=int(((y==0)&(pred>0)).sum())
    result['threat_to_normal']=int(((y>0)&(pred==0)).sum())
    result['target_sources_with_repairs']=int(frame.loc[target&(pred==2),'group'].nunique())
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists(), 'Never overwrite a completed or partial run'
    a.out.mkdir(parents=True)
    f,context=load_fit();d=pd.read_parquet(AUDIT/'row_diagnosis.parquet')
    assert np.array_equal(f.row_position,d.row_position)
    target=target_mask(d);assert target.sum()==578 and f[target].group.nunique()==162
    gate=pd.read_parquet(AUDIT/'normal_gate/unweighted_gate_oof.parquet')
    assert np.array_equal(f.row_position,gate.row_position)
    b=pd.read_parquet(OLD/'H1_selected_oof.parquet')
    assert np.array_equal(f.row_position,b.row_position)
    base=b[['p_B','p_M','p_S']].to_numpy().argmax(1);base[gate.normal_score.ge(.5)]=0
    assert np.array_equal(base[f.label.gt(0)],d.H1_pred[f.label.gt(0)])
    manifest=d.copy();manifest['target_578']=target;manifest['baseline_pred_with_normal_gate']=base
    manifest.to_parquet(a.out/'manifest.parquet',index=False)
    registration={'version':'v67-fixed-target-controls-1','arms':ARMS,'primary':'context_coarse',
        'parameters':PARAMS, 'fits_planned':12, 'threshold':.5, 'labels_for_training':'Original fit rows, other two original source folds only; all M/S rows, no reweighting or target mining',
        'selection':'No checkpoint, threshold, seed, or best-arm selection on validation. All four fixed arms reported.',
        'normal_gate':'Frozen v66 original-fold unweighted normal gate; no new normal training. One original fold has no normal validation, so full acceptance remains unsupported.',
        'evaluation':'All 59640 rows, fixed 578 target, per-fold and per-source repairs AND newly broken M/S. Already observed adaptive development, not blind.',
        'continuation_gate':{'min_target_repairs':58,'min_S_source_gain':.05,'max_M_row_recall_loss':.01,
            'max_M_source_recall_loss':.01,'max_row_macro_loss':.005,'each_fold_M_source_loss':.01,
            'each_fold_hard_protocol_M_source_loss':.01,'no_hard_S_source_decline':True,'normal_errors_not_increased':True},
        'sources':{'run_v67_targeted.py':sha(__file__)},
        'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [AUDIT/'row_diagnosis.parquet',AUDIT/'normal_gate/unweighted_gate_oof.parquet',OLD/'H1_selected_oof.parquet',ROOT/'artifacts/v61_source_factorial_20260914_r2/records.parquet',ROOT/'artifacts/v61_source_factorial_20260914_r2/context.npz']},
        'research':['https://arxiv.org/abs/2207.08815','https://github.com/LeoGrin/tabular-benchmark','https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html']}
    save(a.out/'preregistered.json',registration)
    outputs={arm:np.full(len(f),np.nan) for arm in ARMS}; fitstats=[];start=time.monotonic()
    with threadpool_limits(limits=4):
        for fold in range(3):
            tr=np.flatnonzero(f.fold.ne(fold)&f.label.gt(0));va=np.flatnonzero(f.fold.eq(fold))
            assert not set(f.group.iloc[tr]) & set(f.group.iloc[va])
            assert not set(f.body_group.iloc[tr]) & set(f.body_group.iloc[va])
            for arm in ARMS:
                t=time.monotonic();folder=a.out/f'fold{fold}'/arm;folder.mkdir(parents=True)
                x,encoder,names,catmask=design(f,context,arm,tr)
                model=HistGradientBoostingClassifier(**PARAMS,categorical_features=catmask)
                model.fit(x[tr],f.label.iloc[tr].eq(2).to_numpy())
                score=model.predict_proba(x[va])[:,1];outputs[arm][va]=score
                fitted=model.predict(x[tr]).astype(int)+1
                joblib.dump({'model':model,'encoder':encoder,'names':names,'train_positions':f.row_position.iloc[tr].to_numpy()},folder/'model.joblib')
                # Store thresholds as decisions, scores as uncalibrated conditional S scores.
                fitrow={'fold':fold,'arm':arm,'fit_errors':int((fitted!=f.label.iloc[tr]).sum()),
                    'fit_S_recall':float((fitted[f.label.iloc[tr].eq(2)]==2).mean()),
                    'validation_target_fixed':int((score[target[va]]>=.5).sum()),'seconds':time.monotonic()-t,'features':names}
                fitstats.append(fitrow);save(folder/'fit.json',fitrow)
                print(fitrow,flush=True)
        analysis={'baseline':counts(f,base,base,target),'arms':{},'fit_statistics':fitstats,
                  'quality_acceptance':False,'full_normal_validation_supported':False,'seconds':time.monotonic()-start}
        for arm,score in outputs.items():
            assert np.isfinite(score).all()
            pred=(score>=.5).astype(int)+1;pred[gate.normal_score.ge(.5)]=0
            o=f[['row_position','label','group','fold']].copy();o['p_S_given_threat']=score;o['pred']=pred
            o.to_parquet(a.out/(arm+'_oof.parquet'),index=False)
            entry={'all':counts(f,pred,base,target),'folds':{},'metrics':measure(f,np.eye(3)[pred])}
            for k in range(3):
                m=f.fold.eq(k).to_numpy();entry['folds'][str(k)]={'effects':counts(f[m],pred[m],base[m],target[m]),
                    'candidate':measure(f[m].reset_index(drop=True),np.eye(3)[pred[m]]),
                    'reference':measure(f[m].reset_index(drop=True),np.eye(3)[base[m]])}
            a0=entry['all'];ref=analysis['baseline'];gate_issues=[]
            if a0['target_fixed']<58:gate_issues.append('target_repairs_below_58')
            if a0['S']['source_recall']<ref['S']['source_recall']+.05-1e-12:gate_issues.append('S_source_gain_below_5pp')
            if a0['M']['errors']>ref['M']['errors']+.01*ref['M']['rows']:gate_issues.append('M_row_loss')
            if a0['M']['source_recall']<ref['M']['source_recall']-.01-1e-12:gate_issues.append('M_source_loss')
            if a0['normal_to_threat']>ref['normal_to_threat'] or a0['threat_to_normal']>ref['threat_to_normal']:gate_issues.append('normal_boundary_loss')
            for k,v in entry['folds'].items():
                cm,rm=v['candidate'],v['reference']
                if cm['M_source_recall']<rm['M_source_recall']-.01-1e-12:gate_issues.append(k+'/M_source_loss')
                if cm['ASA']['macro_f1_M_S']<rm['ASA']['macro_f1_M_S']-.005-1e-12:gate_issues.append(k+'/row_macro_loss')
                for p in ['tcp','udp']:
                    if cm['hard'][p]['1']<rm['hard'][p]['1']-.01-1e-12:gate_issues.append(k+'/'+p+'/M_source_loss')
                    if cm['hard'][p]['2']<rm['hard'][p]['2']-1e-12:gate_issues.append(k+'/'+p+'/S_source_loss')
            entry['continuation_passed']=not gate_issues;entry['rejection_reasons']=gate_issues
            analysis['arms'][arm]=entry
        save(a.out/'analysis.json',analysis)
        print('COMPLETE', {k:v['all'] for k,v in analysis['arms'].items()},flush=True)


if __name__=='__main__':main()
