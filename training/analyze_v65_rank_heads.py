"""Replay actual head weights and all probabilities before the registered continuation gate."""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train_v65_rank_heads import (ROOT, ASSET, Head, Inputs, load_fit, measure,
                                  predict, read, save, sha, pair_schedule, BATCH)
from v64_selection import select, flatten
from analyze_v63_factorial import probabilities, paired_source


def compare(actual, expected, path=''):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for k in actual: compare(actual[k], expected[k], path+'/'+str(k))
    elif isinstance(actual, list):
        assert len(actual) == len(expected), path
        for i,(a,b) in enumerate(zip(actual,expected)): compare(a,b,path+'/'+str(i))
    elif isinstance(actual, (float,int)) and not isinstance(actual,bool):
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10,err_msg=path)
    else: assert actual == expected, path


def continuation_gate(folds):
    if any(f['selection']['selected'] is None for f in folds.values()):
        return {'passed':False,'reason':'At least one fold has no budget-feasible checkpoint',
                'pooled_S_gain':None,'all_folds_feasible':False}
    weights=np.array([f['S_validation_sources'] for f in folds.values()],dtype=float)
    changes=[];hard=[]
    for f in folds.values():
        m=flatten(f['selection']['selected']['metrics'])
        refs=[flatten(r) for r in f['references'].values()]
        changes.append(m['S_source']-max(r['S_source'] for r in refs))
        hard.extend(m[p+'_S_source']-max(r[p+'_S_source'] for r in refs) for p in ['tcp','udp'])
    delta=float(np.average(changes,weights=weights))
    no_decline=min(hard)>=-1e-12;improved=max(hard)>1e-12
    passed=delta>=.05-1e-12 and no_decline and improved
    return {'passed':bool(passed),'all_folds_feasible':True,'pooled_S_gain':delta,
            'per_fold_S_gain':changes,'all_hard_S_deltas':hard,
            'no_hard_S_decline':bool(no_decline),'some_hard_S_improvement':bool(improved),
            'reason':'Eligible for fixed-configuration development confirmation only' if passed else
                     'Insufficient pooled S gain and/or inconsistent hard-behavior transfer'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    assert (a.run/'training_complete.json').exists()
    params=read(a.run/'preregistered.json')
    for name,digest in params['sources'].items(): assert sha(ROOT/'training'/name)==digest
    assert sha(ROOT/'docs/V64_DIRECTION_AND_SELECTION_REPAIR.md')==params['plan_sha256']
    for name,digest in read(ASSET/'download_receipt.json')['sha256'].items(): assert sha(ASSET/name)==digest
    frame,context=load_fit();manifest=pd.read_parquet(a.run/'fold_manifest.parquet')
    pd.testing.assert_frame_equal(frame[manifest.columns],manifest)
    cache=read(a.run/'text_cache_receipt.json')
    assert sha(a.run/'upstream_text_features.npy')==cache['features_sha256']
    assert sha(a.run/'unique_text.parquet')==cache['texts_sha256']
    features=np.load(a.run/'upstream_text_features.npy');codes,texts=pd.factorize(frame.text,sort=True)
    assert np.array_equal(texts,pd.read_parquet(a.run/'unique_text.parquet').text)
    torch.set_num_threads(4);device=torch.device('cuda:0')
    folds={};replay={};checkpoint_table=[];source_rows=[];effect_rows=[]
    oof={k:np.full((len(frame),3),np.nan,dtype=np.float32) for k in ['H0_peak','H0_guarded','H0_last','H1_last','coarse','H1_selected']}
    total_metrics=0;total_pairs=0
    for fold in range(3):
        folder=a.run/f'fold{fold}';train=np.flatnonzero(frame.fold.ne(fold));valid=np.flatnonzero(frame.fold.eq(fold))
        ft=frame.iloc[train].reset_index(drop=True);dv=frame.iloc[valid].reset_index(drop=True)
        inputs=Inputs(frame,context,features,codes,train,device)
        prep=read(folder/'preprocessing.json')
        assert inputs.vocab==prep['vocab'];compare(inputs.scaling,prep['scaling'])
        pool=frame.iloc[train][['role','behavior','group','label']].copy()
        pool['index']=train;pool['view']=inputs.views[train]
        schedule=read(folder/'pair_schedule.json');pairs=pd.read_parquet(folder/'pair_exposures.parquet')
        steps=int(np.ceil(len(train)/BATCH));mixed=pool.groupby('view').label.nunique().gt(1)
        for epoch in range(1,params['epochs']+1):
            regenerated,details=pair_schedule(pool,epoch,steps,fold)
            compare({str(k):v for k,v in regenerated.items()},schedule['schedule'][epoch-1])
            compare(details,schedule['support'][epoch-1])
            selected=pairs[pairs.epoch.eq(epoch)]
            actual=[]
            for slot,batch in regenerated.items():
                actual.extend((slot,int(m),int(s)) for m,s in batch)
            assert actual==list(selected[['slot','M_index','S_index']].itertuples(index=False,name=None))
        for side,c in [('M',1),('S',2)]:
            ix=pairs[side+'_index'].to_numpy()
            assert np.isin(ix,train).all() and frame.label.iloc[ix].eq(c).all()
            assert np.array_equal(frame.row_position.iloc[ix],pairs[side+'_row_position'])
            assert np.array_equal(frame.group.iloc[ix],pairs[side+'_group'])
            assert not pd.Series(inputs.views[ix]).map(mixed).any()
        assert pairs.M_group.ne(pairs.S_group).all()
        assert np.array_equal(frame.behavior.iloc[pairs.M_index],frame.behavior.iloc[pairs.S_index])
        exposures=pd.concat([pairs[['epoch','behavior','M_group']].rename(columns={'M_group':'group'}),
                             pairs[['epoch','behavior','S_group']].rename(columns={'S_group':'group'})])
        assert exposures.groupby(['epoch','behavior','group']).size().max()<=8
        total_pairs+=len(pairs)
        curves={};saved={}
        for arm in ['H0','H1']:
            curve=read(folder/arm/'curve.json');curves[arm]=curve
            for item in curve:
                p=probabilities(folder/arm/item['predictions_file'],dv)
                compare(measure(dv,p),item['metrics']);total_metrics+=1
                saved[(arm,item['step'])]=p
                checkpoint_table.append({'fold':fold,'arm':arm,'epoch':item['epoch_fraction'],
                                         **flatten(item['metrics'])})
        ref=read(folder/'references_before_H1.json');assert not ref['H1_started']
        peak=max(curves['H0'],key=lambda x:(x['metrics']['ASA']['macro_f1_M_S'],x['metrics']['S_source_recall'],-x['step']))
        compare(peak,ref['H0_row_peak'])
        majority=ft.groupby(['behavior','label']).size().unstack(fill_value=0).idxmax(axis=1)
        coarse=dv.behavior.map(majority).fillna(int(ft.label.mode().iloc[0])).to_numpy(dtype=int)
        baseline=pd.read_parquet(folder/'coarse_reference.parquet')
        assert np.array_equal(coarse,baseline.pred) and np.array_equal(dv.row_position,baseline.row_position)
        compare(measure(dv,np.eye(3)[coarse]),ref['references']['fit_coarse'])
        compare(peak['metrics'],ref['references']['H0_row_peak'])
        selection=select(curves['H1'],ref['references']);compare(selection,read(folder/'selection.json'))
        training={arm:read(folder/arm/'training.json') for arm in ['H0','H1']}
        assert training['H0']['initial_sha256']==training['H1']['initial_sha256']
        for arm in training:
            assert training[arm]['initial_sha256']!=training[arm]['final_sha256']
            assert not training[arm]['encoder_trained']
        assert training['H1']['rank_gradient_norm_first_batch']>0
        picks={'H0_peak':('H0',peak),'H0_last':('H0',curves['H0'][-1]),'H1_last':('H1',curves['H1'][-1])}
        # Extra fairness diagnostic, never changes the registered reference/gate:
        # apply exactly the same guarded S-first selection to H0 as to H1.
        h0_guarded=select(curves['H0'],ref['references'])
        if h0_guarded['selected'] is not None:picks['H0_guarded']=('H0',h0_guarded['selected'])
        if selection['selected'] is not None:picks['H1_selected']=('H1',selection['selected'])
        fit_metrics={};val_metrics={};already={}
        for label,(arm,item) in picks.items():
            key=(arm,item['step'])
            if key not in already:
                model=Head(inputs.vocab,features.shape[1]).to(device)
                model.load_state_dict(torch.load(folder/arm/item['model_file'],map_location='cpu',weights_only=True))
                check=predict(model,inputs,valid);expected=saved[key]
                np.testing.assert_allclose(check,expected,rtol=1e-5,atol=1e-6)
                assert np.array_equal(check.argmax(1),expected.argmax(1))
                fitp=predict(model,inputs,train)
                d=ft[['row_position','label','group']].copy();d[['p_B','p_M','p_S']]=fitp
                d.to_parquet(folder/arm/(item['model_file']+'.fit.parquet'),index=False)
                replay[f'{fold}/{arm}/{item["step"]}']={'rows':len(valid),'argmax_identical':True,
                    'max_probability_difference':float(np.abs(check-expected).max()),
                    'model_sha256':sha(folder/arm/item['model_file'])}
                already[key]=(measure(ft,fitp),check)
                del model;gc.collect();torch.cuda.empty_cache()
            fit_metrics[label]=already[key][0];val_metrics[label]=item['metrics'];oof[label][valid]=already[key][1]
        oof['coarse'][valid]=np.eye(3)[coarse]
        for cand in ['H1_selected','H1_last']:
            if cand not in picks:continue
            base=oof['H0_peak'][valid].argmax(1);pred=oof[cand][valid].argmax(1)
            d=dv[['group','label','behavior']].copy();d['base']=base==d.label;d['candidate']=pred==d.label
            for (g,c),part in d.groupby(['group','label']):
                source_rows.append({'fold':fold,'candidate':cand,'group':int(g),'label':int(c),
                     'rows':len(part),'base_recall':float(part.base.mean()),'recall':float(part.candidate.mean()),
                     'fixed':int((~part.base&part.candidate).sum()),'broken':int((part.base&~part.candidate).sum())})
            for (b,c),part in d.groupby(['behavior','label']):
                effect_rows.append({'fold':fold,'candidate':cand,'behavior':b,'label':int(c),
                     'rows':len(part),'sources':part.group.nunique(),
                     'fixed':int((~part.base&part.candidate).sum()),'broken':int((part.base&~part.candidate).sum()),
                     'base_source_recall':float(part.groupby('group').base.mean().mean()),
                     'source_recall':float(part.groupby('group').candidate.mean().mean())})
        folds[str(fold)]={'selection':selection,'references':ref['references'],
            'S_validation_sources':int(dv[dv.label.eq(2)].group.nunique()),
            'validation_metrics':val_metrics,'fit_metrics':fit_metrics,
            'posthoc_H0_same_selector_diagnostic':h0_guarded,
            'normal_validation_rows':int(dv.label.eq(0).sum()),
            'pair_exposures':len(pairs),'training':training,
            'mixed_view_training_rows_excluded_only_from_pairs':int(pool.view.map(mixed).sum())}
        print('VERIFIED_FOLD',fold,selection['status'],flush=True)
        del inputs;gc.collect();torch.cuda.empty_cache()
    gate=continuation_gate(folds)
    pd.DataFrame(checkpoint_table).to_csv(a.run/'checkpoint_comparison.csv',index=False)
    pd.DataFrame(source_rows).to_parquet(a.run/'source_effects.parquet',index=False)
    pd.DataFrame(effect_rows).to_csv(a.run/'behavior_effects.csv',index=False)
    aggregate={};comparisons={}
    for name,p in oof.items():
        if np.isfinite(p).all():
            aggregate[name]=measure(frame,p)
            d=frame[['row_position','label','group','fold']].copy();d[['p_B','p_M','p_S']]=p
            d.to_parquet(a.run/(name+'_oof.parquet'),index=False)
    for candidate in ['H1_last','H1_selected']:
        if candidate in aggregate:
            comparisons[candidate+'-minus-H0_peak']=paired_source(frame,oof['H0_peak'],oof[candidate])
    comparisons['H1_last-minus-H0_last']=paired_source(frame,oof['H0_last'],oof['H1_last'])
    if 'H0_guarded' in aggregate and 'H1_selected' in aggregate:
        comparisons['H1_selected-minus-H0_guarded']=paired_source(frame,oof['H0_guarded'],oof['H1_selected'])
    summary={'status':'inner_three_fold_rank_training_passed_requires_confirmation' if gate['passed'] else
                        'inner_three_fold_rank_training_failed_gate_no_model_promoted',
       'quality_acceptance':False,'six_head_fits_executed':True,'gate':gate,'folds':folds,
       'aggregate':aggregate,'paired_source':comparisons,'all_checkpoint_metrics_replayed':total_metrics,
       'weight_replays':replay,'pair_exposures_verified':total_pairs,
       'previous_selection_and_outer_evaluated':False,'analysis_source_sha256':sha(__file__),
       'scope':'Adaptive inner development folds inside old fit only. H1_last is diagnostic, never a fallback selected model. H0_guarded is a post-hoc same-selector fairness check, not a changed gate. OOF is not nested or independent test performance.',
       'limitations':['One fixed seed; checkpoint selection uses these validation labels.',
          'Source symbols are isolation proxies, not verified physical assets or domains.',
          'Only 12 normal rows and a validation fold with none: no operational FPR estimate.',
          'No official validation truth or full SOC quality acceptance.',
          'Unchanged inputs do not establish that all useful event information is present.']}
    save(a.run/'analysis.json',summary)
    print('ANALYSIS_COMPLETE',summary['status'],gate,flush=True)


if __name__=='__main__':main()
