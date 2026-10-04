"""Reload models and independently reproduce predictions, roles and gates."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, read, save, sha, load_sparse, metrics
from v79_execute import rows
from v81_training_contract import accept_candidate, compare
from v82_capacity import DEST, LAST, OLD, counts_for


def cm(y,p,mask):return np.bincount(y[mask]*3+p[mask],minlength=9).reshape(3,3)


def main():
    if (DEST/'verification.json').exists():raise FileExistsError('Already verified')
    contract=read(DEST/'registration.json')
    assert contract['source_sha256']==sha(ROOT/'training/v82_capacity.py')
    assert contract['sample_sha256']==sha(DEST/'selected_rows.npy')
    assert contract['manifest_sha256']==sha(DEST/'manifest.parquet')
    assert read(DEST/'supervision_preflight.json')['original_train_sha256']==sha(ROOT/'data/official/train.parquet')
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X')
    selected=np.zeros(len(r),bool);selected[np.load(DEST/'selected_rows.npy')]=True
    cal=r.fold.eq(2).to_numpy();held=r.fold.eq(0).to_numpy();fit=~(cal|held)
    masks=dict(np.load(DEST/'validation_masks.npz'));old=joblib.load(OLD/'O_sgd.joblib')
    baseline=((x@old.coef_.T+old.intercept_).argmax(1).astype(np.int8))[fid]
    reference=cm(y,baseline,cal);subbase={g:cm(y,baseline,m) for g,m in masks.items()}
    count=0;rows_compared=0;matrix_count=0;results={};trials={}
    for protocol in ['primary','ASA_zero_M','CEF_zero_M']:
        active=selected.copy();route=None
        if protocol!='primary':
            route='asa' if protocol.startswith('ASA') else 'cef_fields'
            active &= ~(r.route.eq(route).to_numpy()&(y==1))
        used=np.flatnonzero(counts_for(fid,y,active,x.shape[0]).sum(1))
        basis=joblib.load(DEST/(protocol+'_basis.joblib'));landmarks=np.load(DEST/(protocol+'_landmark_ids.npy'))
        assert np.array_equal(landmarks,used[basis.component_indices_])
        # Every stored landmark is an unmodified original R0 row.
        assert (basis.components_-x[landmarks]).nnz==0
        k=np.load(DEST/(protocol+'_kernel.npy'),mmap_mode='r')
        ix=np.unique(np.linspace(0,x.shape[0]-1,37,dtype=int))
        delta=float(np.max(abs(basis.transform(x[ix]).astype(np.float32)-k[ix])))
        assert delta<1e-5,delta
        trials[protocol]={'landmark_feature_delta':0,'kernel_recompute_max_delta':delta,'landmarks_exclusively_actual_fit':True}
        for arm in ['linear','nonlinear']:
            name=protocol+'_'+arm;record=read(DEST/(name+'_fit.json'));selection=read(DEST/(name+'_selection.json'))
            assert record['exposed_rows']==int(active.sum())
            assert record['exposed_per_class']==np.bincount(y[active],minlength=3).tolist()
            for state,item in zip(record['states'],selection['trace']):
                label=state['name'];model=joblib.load(DEST/(label+'.joblib'));cached=np.load(DEST/(label+'_prediction.npy'))
                recomputed=np.empty(x.shape[0],np.int8)
                for start in range(0,x.shape[0],3071):
                    scores=np.asarray(x[start:start+3071]@model['coef'])+model['intercept']
                    if arm=='nonlinear':scores+=np.asarray(k[start:start+3071])@model['kernel_coef']
                    recomputed[start:start+3071]=np.argmax(scores,axis=1)
                assert np.array_equal(recomputed,cached),label
                p=recomputed[fid];count+=1;rows_compared+=len(r)
                for role,mask in [('C',cal),('fit',fit)]:
                    assert metrics(cm(y,p,mask))==item[role];matrix_count+=1
                gate=accept_candidate(reference,cm(y,p,cal),{g:(subbase[g],cm(y,p,m)) for g,m in masks.items()})
                assert gate==item['gate'];matrix_count+=1+len(masks)
            final=record['states'][-1]['name'];p=np.load(DEST/(final+'_prediction.npy'))[fid]
            results[name]={'final':final,'converged':record['converged'],'C':metrics(cm(y,p,cal)),'H':metrics(cm(y,p,held)),
                           'fit':metrics(cm(y,p,fit)),'selected':selection['selected']}
            if route is not None:
                for role,mask in [('C',cal),('H',held)]:
                    carrier=mask&r.route.eq(route).to_numpy()
                    results[name][role+'_carrier']=metrics(cm(y,p,carrier))
                    results[name][role+'_other']=metrics(cm(y,p,mask&~r.route.eq(route).to_numpy()))
                    results[name][role+'_carrier_components']=int(r.loc[carrier,'component'].nunique())
    # Compute the strictly registered continuation gates. A failed selected H
    # cannot be replaced with an unselected converged model.
    candidate=results['primary_nonlinear']['selected'];linear=results['primary_linear']['selected']
    primary_gate={'eligible':False,'reason':'no_nonlinear_checkpoint_passed_C'}
    if candidate is not None:
        p=np.load(DEST/(candidate+'_prediction.npy'))[fid]
        primary_gate={'historical_H':compare(cm(y,baseline,held),cm(y,p,held),False)}
        if linear is not None:
            lp=np.load(DEST/(linear+'_prediction.npy'))[fid]
            primary_gate['matched_linear_C']=compare(cm(y,lp,cal),cm(y,p,cal),True)
        primary_gate['eligible']=all(g['eligible'] for g in primary_gate.values())
    transfer={}
    for protocol,route in [('ASA_zero_M','asa'),('CEF_zero_M','cef_fields')]:
        a=results[protocol+'_linear'];b=results[protocol+'_nonlinear']
        transfer[protocol]={'same_protocol_H_carrier':compare(np.array(a['H_carrier']['cm']),np.array(b['H_carrier']['cm']),False),
            'same_protocol_H_other':compare(np.array(a['H_other']['cm']),np.array(b['H_other']['cm']),False),
            'supervised_vs_zero_M_linear':compare(np.array(results['primary_linear']['H']['cm']),np.array(a['H']['cm']),False),
            'supervised_vs_zero_M_nonlinear':compare(np.array(results['primary_nonlinear']['H']['cm']),np.array(b['H']['cm']),False),
            'missing_true_classes_H':np.flatnonzero(np.array(a['H_carrier']['support'])==0).tolist(),
            'scope':'Zero-M ASA/CEF proxy, not VPC quality or real unseen-format assurance. Fixed converged states diagnose mechanism only.'}
    may_continue=primary_gate['eligible'] and all(v['same_protocol_H_carrier']['eligible'] and v['same_protocol_H_other']['eligible'] and v['supervised_vs_zero_M_nonlinear']['eligible'] for v in transfer.values())
    save(DEST/'continuation.json',{'primary':primary_gate,'transfer':transfer,'shadow_refit_authorized_by_evidence':may_continue,
          'full_data_fit_ready':False,'promotion':False,'all_issues_solved':False,'target_answers_read':False})
    save(DEST/'result_summary.json',results)
    report={'status':'passed','classifier_fits':6,'calibration_fits':0,'model_states_reloaded':count,
            'original_row_predictions_compared':rows_compared,'confusion_matrices_checked':matrix_count,
            'fit_only_basis_verification':trials,'raw_train_unchanged':True,'source_sha256':sha(__file__),
            'scope':'Implementation/receipt verification, not quality acceptance. All roles are previously inspected development.'}
    save(DEST/'verification.json',report)
    print(json.dumps({'verification':report,'continuation':read(DEST/'continuation.json')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
