"""Select only from V after BOTH equal-budget fits; never open historic labels."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT, read, save, sha
from v85_protection import cm_from_counts, changes
from v81_training_contract import compare
from v97_prepare import DEST


def class_table(cm):
    cm=np.asarray(cm,dtype=np.int64);support=cm.sum(1);correct=cm.diagonal();pred=cm.sum(0)
    precision=np.divide(correct,pred,out=np.zeros(3,float),where=pred>0)
    recall=np.divide(correct,support,out=np.zeros(3,float),where=support>0)
    f1=np.divide(2*correct,support+pred,out=np.zeros(3,float),where=support+pred>0)
    return {'support':support.tolist(),'correct':correct.tolist(),'predicted':pred.tolist(),
        'precision':precision.tolist(),'recall':recall.tolist(),'f1':f1.tolist(),
        'errors':int(cm.sum()-correct.sum()),'confusion':cm.tolist()}


def main():
    assert not (DEST/'selection.json').exists()
    reg=read(DEST/'registration.json');teacher=read(DEST/'teacher_fit.json')
    assert teacher['converged'] and teacher['actual_classifier_fits']==1
    assert reg['counts_sha256']==sha(DEST/'AB_train_counts.npz')
    reports={arm:read(DEST/f'{arm}_fit.json') for arm in reg['arms']}
    assert all(v['steps_executed']==200 and v['actual_classifier_fits']==1 for v in reports.values())
    assert all(v['source_sha256']==sha(ROOT/'training/v97_branches.py') for v in reports.values())
    assert len({v['initial_state_sha256'] for v in reports.values()})==1
    assert all(v['teacher_model_sha256']==teacher['model_sha256'] for v in reports.values())
    assert all(not v['V_labels_loaded'] and not v['inner_C_H_labels_loaded'] for v in reports.values())
    # The parquet filter materializes V only; no inner/C/H labels or metrics in selection.
    frame=pd.read_parquet(ROOT/'artifacts/v95_cross_component_training_20260928/full_format_roles.parquet',
        columns=['row_position','component','route','label_index','role'],filters=[('role','=','V')])
    assert len(frame)==168615 and frame.role.eq('V').all()
    frame=frame[frame.route.eq('asa')].copy()
    assert len(frame)==7482 and frame.label_index.value_counts().to_dict()=={1:6158,2:1324}
    fid=np.load(ROOT/'artifacts/v79_execution_20260927/row_feature_id.npy',mmap_mode='r')
    pos=frame.row_position.to_numpy();ids=np.asarray(fid[pos]);y=frame.label_index.to_numpy().astype(np.int8)
    base=np.load(DEST/'teacher_prediction.npy',mmap_mode='r')
    models={arm:np.load(DEST/f'{arm}_prediction.npy',mmap_mode='r') for arm in reg['arms']}
    def cm(p):
        out=np.zeros((3,3),np.int64);np.add.at(out,(y,p[ids]),1);return out
    b=cm(base);stats={'teacher':class_table(b)};eligibility={}
    for arm,pred in models.items():
        cand=cm(pred);paired=changes(np.bincount(ids*3+y,minlength=len(base)*3).reshape(-1,3),base,pred)
        guard=compare(b,cand,True)
        train_flip=reports[arm]['final_train_metrics']['teacher_correct_training_negative_flips']
        ok=bool(guard['eligible'] and paired['positive_flips']>0 and paired['negative_flips']==0 and train_flip==0)
        stats[arm]={**class_table(cand),'repairs_vs_teacher':paired['positive_flips'],
            'negative_flips_vs_teacher':paired['negative_flips'],'training_teacher_correct_negative_flips':train_flip}
        eligibility[arm]={'eligible':ok,'guard':guard}
    # Paired component uncertainty for magnitude-method claim, independent of selection outcome.
    gain=(models['MAG'][ids]==y).astype(np.int8)-(models['ERM'][ids]==y).astype(np.int8)
    components,inv=np.unique(frame.component.to_numpy(),return_inverse=True)
    delta=np.bincount(inv,weights=gain,minlength=len(components)).astype(np.int64)
    rng=np.random.default_rng(9701);boot=np.empty(2000,np.int64)
    for k in range(len(boot)):boot[k]=delta[rng.integers(0,len(delta),len(delta))].sum()
    q=np.quantile(boot,[.025,.5,.975]).tolist()
    comparison={'V_ASA_MAG_minus_ERM_correct_rows':int(gain.sum()),'V_ASA_components':len(components),
        'changed_components':int((delta!=0).sum()),'component_bootstrap_2000_q025_q50_q975':q,
        'bootstrap_probability_MAG_positive':float((boot>0).mean()),
        'stable_MAG_gain_gate':bool(gain.sum()>0 and q[0]>0 and stats['MAG']['correct'][1]>=stats['ERM']['correct'][1] and
            stats['MAG']['correct'][2]>=stats['ERM']['correct'][2]),
        'scope':'Previously inspected V development, component resampling is uncertainty diagnostic not independent transfer evidence.'}
    pd.DataFrame({'component':components,'MAG_minus_ERM_correct_rows':delta}).to_csv(DEST/'V_ASA_component_delta.csv',index=False)
    save(DEST/'V_ASA_method_comparison.json',comparison)
    qualified=[arm for arm in reg['arms'] if eligibility[arm]['eligible']]
    best=min(qualified,key=lambda arm:(stats[arm]['errors'],arm)) if qualified else None
    report={'status':'V_only_selection_frozen_before_historic_diagnosis','selection_endpoint_step':200,
        'selection_criterion':reg['selection'],'teacher_and_arms':stats,'eligibility':eligibility,
        'selected_arm':best,'matched_method_comparison':comparison,
        'MAG_method_gain_claim_supported':bool(best=='MAG' and comparison['stable_MAG_gain_gate']),
        'initial_state_equal':True,'V_ASA_rows':len(frame),'V_ASA_class_rows':[0,6158,1324],
        'inner_C_H_labels_loaded':False,'fit_rows_used_for_V_gradient':0,
        'actual_classifier_fits':3,'actual_calibration_fits':0,'source_sha256':sha(__file__),
        'artifact_bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in
            [DEST/'teacher_fit.json',DEST/'ERM_fit.json',DEST/'MAG_fit.json',DEST/'teacher_prediction.npy',
             DEST/'ERM_prediction.npy',DEST/'MAG_prediction.npy']}}
    save(DEST/'selection.json',report)
    print(json.dumps({'stage':'V_selection_frozen','teacher_V_ASA':stats['teacher'],
        'ERM_V_ASA':stats['ERM'],'MAG_V_ASA':stats['MAG'],'selected_arm':best,
        'method_comparison':comparison},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
