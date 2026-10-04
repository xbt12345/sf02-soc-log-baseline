"""Paired source/slice analysis of registered development arms, no outer evaluation."""
import argparse
import gc
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from v61_common import FIELDS, fit_vocab, encode_facts, read, save, sha
from train_v63_factorial import (ARMS, AUDIT, OLD, ROOT, load_development, inputs_for,
                                 model_for, predict, metric_summary, budget)


def probabilities(path, frame):
    p = pd.read_parquet(path).sort_values('row_position').reset_index(drop=True)
    assert p[['row_position','label','group']].equals(frame[['row_position','label','group']].reset_index(drop=True))
    x = p[['p_B','p_M','p_S']].to_numpy()
    assert np.isfinite(x).all() and (x >= 0).all() and np.abs(x.sum(1)-1).max() < 2e-6
    return x


def paired_source(frame, base, candidate, repeats=2000):
    d = frame[['group','label']].copy()
    d['delta'] = (candidate.argmax(1) == d.label).astype(float) - (base.argmax(1) == d.label).astype(float)
    x = d[d.label.gt(0)].groupby(['group','label']).delta.mean().unstack().reindex(columns=[1,2])
    values = x.fillna(0).to_numpy(); mask = x.notna().to_numpy()
    rng = np.random.default_rng(20260920); draws = []
    for start in range(0, repeats, 50):
        index = rng.integers(len(x), size=(min(50,repeats-start),len(x)))
        sums = values[index].sum(1); count = mask[index].sum(1)
        assert (count > 0).all(); draws.append(sums/count)
    draws = np.concatenate(draws)
    return {'source_M_delta':float(x[1].mean()), 'source_S_delta':float(x[2].mean()),
            'source_balanced_delta':float(x.mean().mean()),
            'M_95_interval':np.quantile(draws[:,0],[.025,.975]).tolist(),
            'S_95_interval':np.quantile(draws[:,1],[.025,.975]).tolist(),
            'balanced_95_interval':np.quantile(draws.mean(1),[.025,.975]).tolist(),
            'whole_source_joint_class_resampling':True, 'draws':repeats,
            'scope':'Conditional adaptive-development diagnostic, not corrected for checkpoint/method selection; not blind evidence'}


def optimistic_threshold(part, scores):
    """Selection-label oracle: diagnostic feasibility only, never a deployable threshold."""
    n=len(part);m=part.label.eq(1).to_numpy();s=part.label.eq(2).to_numpy()
    assert m.any() and s.any() and (m|s).all()
    weights={}
    for c,mask in [(1,m),(2,s)]:
        sub=part[mask];count=sub.groupby('group').size();w=np.zeros(n)
        w[mask]=1/len(count)/sub.group.map(count).to_numpy();weights[c]=w
    order=np.argsort(-scores,kind='stable');sorted_scores=scores[order]
    ends=np.r_[np.flatnonzero(sorted_scores[:-1]!=sorted_scores[1:]),n-1]
    mc=np.cumsum(m[order])/m.sum();ms=np.cumsum(weights[1][order]);ss=np.cumsum(weights[2][order])
    valid=ends[(mc[ends]<=.01+1e-12)&(ms[ends]<=.01+1e-12)]
    if len(valid)==0 or ss[valid].max()==0:
        return np.zeros(n,dtype=bool),{'predict_none':True,'threshold':None,'S_source_recall':0.}
    best=max(valid,key=lambda k:(ss[k],-mc[k],-k));threshold=float(sorted_scores[best])
    pred=scores>=threshold
    return pred,{'predict_none':False,'threshold':threshold,'S_source_recall':float(ss[best]),
                 'M_row_false_S_rate':float(mc[best]),'M_source_false_S_rate':float(ms[best])}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    assert (a.run/'training_complete.json').exists()
    params=read(a.run/'preregistered.json')
    for name,digest in params['sources'].items(): assert sha(ROOT/'training'/name)==digest,name
    frame,pool,context=load_development()
    coarse=frame[FIELDS[:5]].astype(str).agg('|'.join,axis=1)
    majority=frame[frame.role.eq('fit')].assign(behavior=coarse[frame.role.eq('fit')]).groupby(['behavior','label']).size().unstack(fill_value=0).idxmax(axis=1)
    fit=np.flatnonzero(frame.role.eq('fit'));dev=np.flatnonzero(frame.role.eq('selection'))
    dv=frame.iloc[dev].reset_index(drop=True)
    diagnostics=pd.read_parquet(AUDIT/'development_diagnostics.parquet')
    dv=dv.merge(diagnostics[['row_position','fit_support','empty_context','single','key','behavior']],on='row_position',validate='one_to_one')
    facts=frame[FIELDS].astype(str).to_numpy();encoded=encode_facts(facts,fit_vocab(facts[fit]))
    src=encoded[dev,FIELDS.index('src_port_fixed')]==1;dst=encoded[dev,FIELDS.index('dst_port_fixed')]==1
    dv['port_slice']=np.where(dst,'destination_OOV',np.where(src,'source_only_OOV','ports_seen'))
    dv['size_slice']=np.where(dv.group.map(dv.groupby('group').size()).eq(2),'two_rows','other_sizes')
    for name,col in [('single_conflict','single'),('D_cached_conflict','key')]:
        mixed=dv.groupby(col).label.nunique().gt(1)
        dv[name]=dv[col].map(mixed)
    guard=read(a.run/'additional_baseline_guard.json')
    rule_labels=coarse.iloc[dev].map(majority).fillna(frame.iloc[fit].label.mode().iloc[0]).to_numpy(dtype=int)
    rule_prob=np.eye(3,dtype=np.float32)[rule_labels]
    predictions={'old_D':probabilities(OLD/'meanmax/selection.parquet',dv),
                 'A0_row_peak':probabilities(a.run/'A0'/guard['row_peak_A0']['predictions_file'],dv)}
    selections={};fit_results={};last_fit_results={};checkpoint_rows=[];replay={}
    for arm in ARMS:
        selected=read(a.run/arm/'selection.json');selections[arm]=selected
        check=selected['checkpoint']
        assert sha(a.run/arm/check['model_file'])==selected['model_sha256']
        assert sha(a.run/arm/check['predictions_file'])==selected['predictions_sha256']
        predictions[arm]=probabilities(a.run/arm/check['predictions_file'],dv)
        for item in read(a.run/arm/'curve.json'):
            checkpoint_rows.append({'arm':arm,'step':item['step'],'epoch_fraction':item['epoch_fraction'],
                                    'row_macro':item['metrics']['ASA']['macro_f1_M_S'],
                'M_source':item['metrics']['M_source_recall'],
                'S_source':item['metrics']['S_source_recall']})
    coarse_reference={'scope':'Post-registration fit-only majority diagnostic, not a training candidate or calibrated probability model',
        'metrics':metric_summary(dv,rule_prob),
        'selected_agreement':{arm:{'all_rows':int((p.argmax(1)==rule_labels).sum()),
                                 'ASA_rows':int(((p.argmax(1)==rule_labels)&dv.label.gt(0)).sum())} for arm,p in predictions.items()}}
    pd.DataFrame(checkpoint_rows).to_csv(a.run/'checkpoint_comparison.csv',index=False)
    rows=[]
    masks={'all':np.ones(len(dv),dtype=bool)}
    for column in ['behavior','fit_support','empty_context','port_slice','size_slice','single_conflict','D_cached_conflict']:
        for value in dv[column].unique(): masks[column+'='+str(value)]=dv[column].eq(value).to_numpy()
    for (behavior,port),sub in dv.groupby(['behavior','port_slice']):
        masks['behavior_port='+behavior+'/'+port]=dv.index.isin(sub.index)
    for reference in ['A0','old_D','A0_row_peak']:
        base=predictions[reference].argmax(1)
        for arm in ARMS:
            cand=predictions[arm].argmax(1)
            for slice_name,mask in masks.items():
                for label in [1,2]:
                    sub=dv[mask & dv.label.eq(label)].copy()
                    if sub.empty: continue
                    ix=sub.index.to_numpy();y=sub.label.to_numpy()
                    sub['a']=base[ix]==y;sub['b']=cand[ix]==y
                    rows.append({'reference':reference,'arm':arm,'slice':slice_name,'label':label,'rows':len(sub),
                                 'sources':sub.group.nunique(),'base_recall':float(sub.a.mean()),'recall':float(sub.b.mean()),
                                 'base_source_recall':float(sub.groupby('group').a.mean().mean()),
                                 'source_recall':float(sub.groupby('group').b.mean().mean()),
                                 'fixed':int((~sub.a & sub.b).sum()),'broken':int((sub.a & ~sub.b).sum())})
    pd.DataFrame(rows).to_csv(a.run/'slice_comparison.csv',index=False)
    ranking=[]
    for name,mask in masks.items():
        if not (name=='all' or name.startswith('behavior=') or name.startswith('behavior_port=')):
            continue
        sub=dv[mask & dv.label.gt(0)]
        if sub.label.nunique()!=2: continue
        weights=np.empty(len(sub),dtype=float)
        for c in [1,2]:
            local=sub.label.eq(c).to_numpy();part=sub[sub.label.eq(c)]
            counts=part.groupby('group').size()
            weights[local]=.5/len(counts)/part.group.map(counts).to_numpy()
        for arm,p in predictions.items():
            score=p[sub.index,2];target=sub.label.eq(2).to_numpy()
            ranking.append({'arm':arm,'slice':name,'rows':len(sub),
                'S_prevalence':float(target.mean()),
                'S_average_precision':float(average_precision_score(target,score)),
                'source_class_balanced_S_average_precision':float(average_precision_score(target,score,sample_weight=weights))})
    pd.DataFrame(ranking).to_csv(a.run/'ranking_diagnostics.csv',index=False)
    oracle={}
    for arm in ARMS:
        # Other behaviors use the fit-only coarse majority rule. Only two prespecified
        # hard buckets receive optimistically chosen selection-label thresholds.
        pred=rule_labels.copy();details={}
        for protocol in ['tcp','udp']:
            mask=dv.transport_protocol.eq(protocol)&dv.src_role.eq('outside')&dv.dst_role.eq('dmz')
            sub=dv[mask];p=predictions[arm][mask]
            score=p[:,2]/np.maximum(p[:,1]+p[:,2],1e-15)
            decision,detail=optimistic_threshold(sub,score)
            pred[mask]=np.where(decision,2,1);details[protocol]=detail
        oracle[arm]={'metrics':metric_summary(dv,np.eye(3,dtype=np.float32)[pred]),'buckets':details}
    save(a.run/'optimistic_threshold_diagnostic.json',{'scope':'Post-registration selection-label oracle with fit-only coarse majority for other behaviors. An optimistic feasibility diagnostic, NOT validation, NOT a trained/calibrated model, NOT eligible for promotion. Thresholds must not be reused for deployment.',
          'hard_bucket_M_row_and_source_false_S_budget':.01,'models':oracle})
    comparisons={}
    for base,candidate in [('A0','A1'),('A0','P0'),('A1','P1'),('P0','P1'),('old_D','A0'),('old_D','A1'),('old_D','P0'),('old_D','P1')]+[('A0_row_peak',x) for x in ARMS]:
        comparisons[candidate+'-minus-'+base]=paired_source(dv,predictions[base],predictions[candidate])
    # Restore each selected checkpoint in a fresh model and verify saved predictions,
    # then measure the same checkpoint on fit to distinguish fit vs transfer gaps.
    torch.set_num_threads(4);device=torch.device('cuda:0')
    for arm in ARMS:
        inputs=inputs_for(frame,context,device,arm)
        model=model_for(inputs.vocab,arm)
        encoder_name,encoder_tensor=next(model.encoder.named_parameters())
        initial_encoder_hash=hashlib.sha256(encoder_tensor.detach().numpy().tobytes()).hexdigest()
        model.load_state_dict(torch.load(a.run/arm/selections[arm]['checkpoint']['model_file'],map_location='cpu',weights_only=True))
        trained_encoder_hash=hashlib.sha256(dict(model.encoder.named_parameters())[encoder_name].detach().numpy().tobytes()).hexdigest()
        assert initial_encoder_hash!=trained_encoder_hash
        model.to(device)
        check=predict(model,inputs,dev)
        np.testing.assert_allclose(check,predictions[arm],rtol=1e-5,atol=1e-6)
        assert np.array_equal(check.argmax(1),predictions[arm].argmax(1))
        replay[arm]={'rows':len(dev),'argmax_identical':True,
                     'maximum_probability_difference':float(np.abs(check-predictions[arm]).max()),
                     'checked_encoder_parameter':encoder_name,'encoder_parameter_changed':True,
                     'initial_encoder_parameter_sha256':initial_encoder_hash,
                     'trained_encoder_parameter_sha256':trained_encoder_hash}
        p=predict(model,inputs,fit)
        part=frame.iloc[fit][['row_position','label','group']].copy();part[['p_B','p_M','p_S']]=p
        part.to_parquet(a.run/arm/'selected_fit.parquet',index=False)
        fit_results[arm]=metric_summary(frame.iloc[fit],p)
        print('SELECTED_FIT '+arm+' '+str(fit_results[arm]['source_balanced']),flush=True)
        last=read(a.run/arm/'curve.json')[-1]
        if last['model_file']==selections[arm]['checkpoint']['model_file']:
            last_fit_results[arm]=fit_results[arm]
        else:
            model.load_state_dict(torch.load(a.run/arm/last['model_file'],map_location='cpu',weights_only=True))
            last_prob=predict(model,inputs,fit)
            last_part=frame.iloc[fit][['row_position','label','group']].copy()
            last_part[['p_B','p_M','p_S']]=last_prob
            last_part.to_parquet(a.run/arm/'last_epoch_fit.parquet',index=False)
            last_fit_results[arm]=metric_summary(frame.iloc[fit],last_prob)
        print('LAST_FIT '+arm+' '+str(last_fit_results[arm]['source_balanced']),flush=True)
        del model,inputs;gc.collect();torch.cuda.empty_cache()
    registered_passing=[arm for arm in ARMS[1:] if selections[arm]['screen_passed']]
    guard_results={arm:budget(selections[arm]['checkpoint']['metrics'],guard['row_peak_A0']['metrics']) for arm in ARMS}
    passing=[arm for arm in registered_passing if guard_results[arm]]
    # The interaction must beat simpler eligible arms before retaining P1.
    if 'P1' in passing:
        simpler=[x for x in passing if x!='P1']
        if any(selections['P1']['checkpoint']['metrics']['source_balanced'] <= selections[x]['checkpoint']['metrics']['source_balanced'] for x in simpler):
            passing.remove('P1')
    save(a.run/'analysis.json',{'status':'four_arm_development_analysis_completed','quality_acceptance':False,
        'passing_screen_candidates':passing,'registered_screen_candidates':registered_passing,
        'additional_conservative_row_peak_guard':guard_results,
        'additional_guard_declared_before_A1_checkpoint':not guard['A1_checkpoint_file_exists_at_declaration'],
        'no_new_outer_evaluation':True,'selected':selections,
        'fit_metrics':fit_results,'last_epoch_fit_metrics':last_fit_results,'paired_source':comparisons,
        'checkpoint_replay':replay,'analysis_source_sha256':sha(__file__),
        'coarse_majority_diagnostic':coarse_reference,
        'limitations':['Single seed, adaptive selection; no new network/domain validation.',
                      'Ranking diagnostics are post-registration analyses, not threshold selection or quality gates.',
                      'Optimistic threshold diagnostic uses selection labels; its scores are not out-of-sample model results.',
                      'Port OOV is not randomized; factorial changes identify module effects, not label causality.',
                      'Normal controls too few for operational false-positive estimation.',
                      'Passing this screen only permits registered replication, not model promotion.']})
    print('ANALYSIS_COMPLETE passing_screen_candidates='+str(passing),flush=True)


if __name__=='__main__': main()
