"""Seal choices before outer evaluation; paired source-cluster uncertainty and controls."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from v61_common import save, read, sha, effects
from v61_runtime import load_data, score, predictions, source_receipt


def paired_cluster(base,candidate,resamples=2000,seed=20260915):
    assert base.row_position.tolist()==candidate.row_position.tolist()
    assert base.label.tolist()==candidate.label.tolist()
    keep=base.label.gt(0).to_numpy();b=base[keep];c=candidate[keep]
    groups,gi=np.unique(b.group.to_numpy(),return_inverse=True);y=b.label.to_numpy()
    cm=[]
    for frame in [b,c]:
        pred=frame[['p_B','p_M','p_S']].to_numpy().argmax(1)
        m=np.zeros((len(groups),9),dtype=np.float64);np.add.at(m,(gi,y*3+pred),1);cm.append(m)
    def summarise(m):
        m=m.reshape((-1,3,3));diagonal=np.diagonal(m,axis1=1,axis2=2)
        denom=m.sum(1)+m.sum(2);support=m.sum(2)
        f1=np.divide(2*diagonal,denom,out=np.zeros_like(diagonal),where=denom>0)
        rec=np.divide(diagonal,support,out=np.zeros_like(diagonal),where=support>0)
        return np.column_stack([f1[:,1:].mean(1),rec[:,1],rec[:,2]])
    rng=np.random.default_rng(seed);deltas=[]
    for start in range(0,resamples,100):
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=min(100,resamples-start))
        deltas.append(summarise(weights@cm[1])-summarise(weights@cm[0]))
    d=np.concatenate(deltas)
    return {'resamples':resamples,'paired_unit':'source entity','ASA_source_groups':len(groups),
        'delta_95_percentile_CI':dict(zip(['macro_f1_M_S','recall_M','recall_S'],np.quantile(d,[.025,.975],axis=0).T.tolist())),
        'cluster_intervals_do_not_establish_external_transfer':True}


def collect_selection(out):
    result={};base=pd.read_parquet(out/'A_20260915/selection.parquet')
    amendment=read(out/'ceiling_gate_amendment.json')
    a=read(out/'A_20260915/selection.json')['selection']['ASA']
    for arm in 'ABCD':
        p=out/(arm+'_20260915');report=read(p/'selection.json')
        model=p/('model.joblib' if arm in 'AC' else 'model.pt')
        assert sha(model)==report['model_sha256'];assert sha(p/'selection.parquet')==report['predictions_sha256']
        eff=effects(a,report['selection']['ASA']);ci=paired_cluster(base,pd.read_parquet(p/'selection.parquet'))
        dr=eff['delta_recall_B_M_S']
        meets=(eff['delta_macro_f1_M_S']>=.02 and dr[2]>=amendment['replacement_delta_S_requirement'] and dr[1]>=-.01)
        result[arm]={'fit':report['fit'],'selection':report['selection'],'effects_from_A':eff,'paired_interval':ci,
            'amended_inner_effect_size_met':bool(meets),'continue_seed_replication':bool(meets and ci['delta_95_percentile_CI']['macro_f1_M_S'][0]>0),
            'model_sha256':sha(model),'selection_report_sha256':sha(p/'selection.json')}
    save(out/'inner_comparison.json',{'arms':result,'outer_used':False,'gate_amendment_sha256':sha(out/'ceiling_gate_amendment.json')})
    return result


def seal(out,data):
    choices=collect_selection(out);runs=sorted(p for p in out.iterdir() if p.is_dir() and (p/'selection.json').exists())
    eligible=[k for k,v in choices.items() if v['continue_seed_replication']]
    for arm in ['A']+eligible if eligible else []:
        for seed in [20260916,20260917]:
            assert (out/f'{arm}_{seed}/selection.json').exists(),f'Complete fixed replication before outer: {arm} {seed}'
    source=Path(__file__).parent
    receipt={'data_configuration_sha256':sha(data/'configuration.json'),'selection_before_outer':True,
        'eligible_for_replication':eligible,'runs':{},'analysis_source_sha256':sha(__file__),
        'stage':'research entity holdout in previously studied official corpus; not fresh external validation'}
    for p in runs:
        report=read(p/'selection.json');model='model.joblib' if p.name[0] in 'AC' else 'model.pt'
        for name,digest in read(p/'preregistered.json')['source_sha256'].items():assert sha(source/name)==digest,name
        receipt['runs'][p.name]={name:sha(p/name) for name in ['preregistered.json','selection.json','selection.parquet',model]}
    target=out/'outer_seal.json';assert not target.exists(),'Outer selection already sealed'
    save(target,receipt);print({'sealed_runs':list(receipt['runs']),'replicated_candidates':eligible},flush=True)


def context_permutation(df,indices):
    rng=np.random.default_rng(20260915);permutation=indices.copy()
    for proto in df.transport_protocol.iloc[indices].unique():
        mask=df.transport_protocol.iloc[indices].eq(proto).to_numpy()
        permutation[mask]=rng.permutation(indices[mask])
    return permutation


def evaluate(out,data,model_path):
    receipt=read(out/'outer_seal.json');assert sha(data/'configuration.json')==receipt['data_configuration_sha256']
    assert sha(__file__)==receipt['analysis_source_sha256']
    df,ctx,_=load_data(data);idx=np.flatnonzero(df.role.eq('evaluation'));shuffled=context_permutation(df,idx)
    outer=out/'outer';assert not outer.exists(),'No repeated outer examination';outer.mkdir()
    reports={};neural_inputs=None
    for run,binding in receipt['runs'].items():
        p=out/run;arm=run[0]
        for name,digest in binding.items():assert sha(p/name)==digest,name
        if arm in 'AC':
            from train_v61_linear import features
            b=joblib.load(p/'model.joblib');x,xc,_=features(df,ctx,b['features'])
            probabilities=b['model'].predict_proba((xc if arm=='C' else x)[idx])
            diagnostic=None
            if arm=='C':
                cs=b['features']['scale'].transform(ctx['stats'][shuffled])
                changed=sparse.hstack([x[idx],sparse.csr_matrix(cs)],format='csr')
                diagnostic=b['model'].predict_proba(changed)
        else:
            import torch
            from train_v61_neural import Inputs, Classifier, predict
            torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pp=read(p/'preprocessing.json')
            if neural_inputs is None:neural_inputs=Inputs(df,ctx,model_path,device,pp['vocab'],pp['scaling'])
            assert neural_inputs.vocab==pp['vocab'] and neural_inputs.scaling==pp['scaling']
            m=Classifier(model_path,pp['vocab'],arm,pretrained=False).to(device)
            m.load_state_dict(torch.load(p/'model.pt',map_location=device,weights_only=True))
            probabilities=predict(m,neural_inputs,idx)
            diagnostic=predict(m,neural_inputs,idx,context_permutation=shuffled) if arm=='D' else None
            del m;torch.cuda.empty_cache() if device.type=='cuda' else None
        predictions(outer/(run+'.parquet'),df,idx,probabilities)
        reports[run]={'evaluation':score(df.label.iloc[idx],probabilities)}
        if diagnostic is not None:
            reports[run]['label_blind_context_shuffle_within_protocol']={'metrics':score(df.label.iloc[idx],diagnostic),
                'prediction_flip_fraction':float((diagnostic.argmax(1)!=probabilities.argmax(1)).mean()),
                'meaning':'Changes possible causal context; sensitivity diagnostic, not a label-preserving robustness test'}
        print(run+' outer prediction saved.',flush=True)
    base=pd.read_parquet(outer/'A_20260915.parquet')
    for run in reports:
        reports[run]['paired_vs_A_first_seed']=paired_cluster(base,pd.read_parquet(outer/(run+'.parquet')))
        reports[run]['effects_vs_A_first_seed']=effects(reports['A_20260915']['evaluation']['ASA'],reports[run]['evaluation']['ASA'])
        frame=pd.read_parquet(outer/(run+'.parquet'));pred=frame[['p_B','p_M','p_S']].to_numpy().argmax(1)
        err=frame.assign(error=pred!=frame.label).groupby('group').agg(errors=('error','sum'),rows=('error','size'))
        reports[run]['largest_error_source_groups']=err.sort_values('errors',ascending=False).head(10).reset_index().to_dict('records')
        reports[run]['sources_with_error']=int(err.errors.gt(0).sum())
    save(outer/'summary.json',{'runs':reports,'seal_sha256':sha(out/'outer_seal.json'),
        'source_entity_protocol_only':True,'normal_controls':2,'normal_control_entities':1,
        'no_official_validation_labels_no_external_acceptance':True})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['selection','seal','evaluation'],required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--model',type=Path)
    a=ap.parse_args()
    if a.stage=='selection':
        r=collect_selection(a.out);print({k:{'macro_f1':v['selection']['ASA']['macro_f1_M_S'],'continue':v['continue_seed_replication']} for k,v in r.items()})
    elif a.stage=='seal':seal(a.out,a.data)
    else:evaluate(a.out,a.data,a.model)


if __name__=='__main__':main()
