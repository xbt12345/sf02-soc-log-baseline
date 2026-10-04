"""Recount actual sealed probe ledgers and perturbation geometry without fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from experiment_review import read,sha,check_bindings
from v135_runtime import load_data,fit_context
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v154_directional_neighborhood_v2_20261001'
OLD=ROOT/'artifacts/v146_guarded_pair_training_20261001'

def flat(values):return torch.cat([v.reshape(-1) for v in values.values()])

def main():
    a=read(OUT/'audit.json');check_bindings(a['source_sha256'])
    check_bindings(read(OUT/'run_seal.json')['source_sha256'])
    _,d=load_data();registry=read(ROOT/'artifacts/v142_second_layer_training_20261001/verified_TRAIN_mastery_registry.json')
    protection={r['training_role']:pd.read_parquet(ROOT/r['guard']) for r in registry['scopes']}
    ledger_checks=0;geometry_checks=0
    for arm in ['A','B']:
        for f in range(3):
            folder=OUT/f'fold{f}_{arm}';r=read(folder/'probe.json')
            assert r['full_classifier_gradient_evaluations']==3 and r['classifier_state_unchanged']
            assert r['before_model_tensor_sha256']==r['after_model_tensor_sha256']
            assert sha(folder/'directions.pt')==r['directions_sha256'] and sha(folder/'gradients.pt')==r['gradients_sha256']
            ds=torch.load(folder/'directions.pt',map_location='cpu',weights_only=True)
            gs=torch.load(folder/'gradients.pt',map_location='cpu',weights_only=True)
            for k,v in gs['base_gradient'].items():assert torch.equal(v,gs['repeated_base_gradient'][k])
            norm=float(flat(gs['base_gradient']).norm());rho=r['radius_L2']
            assert np.isclose(norm,r['base_gradient_L2'],rtol=1e-12,atol=1e-15)
            for k,v in ds['plus_TRAIN_gradient'].items():
                assert torch.equal(-v,ds['minus_TRAIN_gradient'][k])
                assert torch.equal(v,gs['base_gradient'][k]*(rho/r['base_gradient_L2']))
            for values in ds.values():assert np.isclose(float(flat(values).norm()),rho,rtol=1e-12,atol=1e-15)
            assert np.isclose(rho,.001*r['parameter_L2'],rtol=1e-15);geometry_checks+=1
            frame,c,pure,_,ids=fit_context(d,f);old=np.load(OLD/f'fold{f}_{arm}/sealed_all_prob.npy')
            assert r['original_class_mass_per_gradient']==c.sum(0).tolist()
            for point in r['points']:
                name=point['point'];q=np.load(folder/f'{name}_all_prob.npy');rows=pd.read_parquet(folder/f'{name}_TRAIN_rows.parquet')
                assert point['probability_sha256']==sha(folder/f'{name}_all_prob.npy')
                assert point['TRAIN_rows_sha256']==sha(folder/f'{name}_TRAIN_rows.parquet')
                assert np.array_equal(rows.row_position,frame.row_position) and np.array_equal(rows.truth,frame.truth)
                assert np.array_equal(rows.pred,q[frame.local].argmax(1))
                for cl in range(3):assert np.array_equal(rows[f'p{cl}'],q[frame.local,cl])
                if name=='base':assert np.array_equal(q,old)
                wrong=rows.pred.ne(rows.truth)
                assert point['TRAIN_class_errors']==[int((wrong&rows.truth.eq(cl)).sum()) for cl in range(3)]
                assert point['pure_TRAIN_errors']==int((wrong&pure[frame.local].astype(bool)).sum())
                ledger_checks+=1
        for point in ['base','plus_TRAIN_gradient','minus_TRAIN_gradient','fixed_random']:
            train=pd.read_parquet(OUT/f'{arm}_{point}_all_TRAIN_role_rows.parquet')
            outer=pd.read_parquet(OUT/f'{arm}_{point}_outer_original_rows.parquet')
            assert len(train)==225614 and not train.duplicated(['row_position','training_role']).any()
            assert np.array_equal(outer.row_position,d.row_position) and np.array_equal(outer.truth,d.truth)
            guards=read(OUT/'joint_TRAIN_retention_checks.json')[arm+'_'+point]
            for f in range(3):
                guard=protection[f];pred=train[train.training_role.eq(f)].set_index('row_position').pred
                failures=int((guard.row_position.map(pred).to_numpy()!=guard.truth.to_numpy()).sum())
                actual=next(z for z in guards['mastered_TRAIN'] if z['scope_id']==registry['scopes'][f]['id'])
                assert bool(actual['repair_protection_passed'])==(failures==0)
            for s in [z for z in a['outer_summaries'] if z['arm']==arm and z['point']==point and z['cohort']=='all_ASA']:
                assert s['outer_class_errors']==[int((outer.pred.ne(outer.truth)&outer.truth.eq(cl)).sum()) for cl in range(3)]
    assert ledger_checks==24 and geometry_checks==6
    assert a['full_classifier_gradient_evaluations']==18 and a['functional_perturbed_points']==18
    assert a['classifier_fits']==a['persistent_parameter_updates']==0
    print(json.dumps(dict(status='actual_probe_recount_passed',endpoint_geometry_checks=geometry_checks,
        local_probability_TRAIN_ledger_checks=ledger_checks,all_original_ASA_rows=112807,
        legal_TRAIN_role_rows_per_arm=225614,classifier_fits=0,updates=0),ensure_ascii=False))

if __name__=='__main__':main()
