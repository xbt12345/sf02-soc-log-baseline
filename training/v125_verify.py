"""Independent original-row and artifact audit of the sealed V125 trial."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v125_order_trial_20260929'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for b in iter(lambda:stream.read(1048576),b''):h.update(b)
    return h.hexdigest()


def obj(path):return json.loads(path.read_text(encoding='utf-8'))


def exact(value,message):
    if not value:raise ValueError(message)


def counts(y,p):
    result={}
    for cls in range(3):
        actual=y==cls;pred=p==cls;tp=int((actual&pred).sum());n=int(actual.sum());called=int(pred.sum())
        result[str(cls)]={'support':n,'correct':tp,'missed':n-tp,'false_called':called-tp,
                          'recall':tp/n if n else None,'precision':tp/called if called else None,
                          'f1':2*tp/(n+called) if n and n+called else None}
    return result


def main():
    dest=OUT/'verification.json'
    if dest.exists():raise FileExistsError(dest)
    seal=obj(OUT/'run_seal.json');reg=obj(OUT/'registration.json');primary=obj(OUT/'primary_evaluation.json')
    exact(seal['status']=='sealed_before_fit' and reg['status']=='registered_before_any_optimizer_step','Invalid prefit seal')
    exact(reg['seal_sha256']==digest(OUT/'run_seal.json'),'Registration seal changed')
    for rel,h in seal['source_sha256'].items():exact(digest(ROOT/rel)==h,'Bound source changed: '+rel)
    files=[];total_steps=0
    for fold in range(3):
        for arm in 'ABC':
            folder=OUT/f'fold{fold}_{arm}'
            fit=obj(folder/'fit.json');progress=obj(folder/'progress.json');checkpoints=obj(folder/'checkpoints.json')
            exact((fit['fold'],fit['arm'],fit['seed'],fit['completed_epochs'],fit['prediction_epoch'])==(fold,arm,10201,25,25),'Wrong fit identity')
            exact(fit['seal_sha256']==reg['seal_sha256'] and fit['model_sha256']==digest(folder/'epoch25_model.pt') and fit['prob_sha256']==digest(folder/'epoch25_prob.npy'),'Endpoint artifact identity changed')
            exact([z['epoch'] for z in progress]==list(range(1,26)) and [z['epoch'] for z in checkpoints]==[1,2,5,10,15,20,25],'Incomplete training trajectory')
            exact([z['optimizer_steps_cumulative'] for z in progress]==[(i+1)*reg['schedule'][fold]['logical_batches'] for i in range(25)],'Optimizer step trajectory changed')
            exact(all(z['mass_reconstruction_max_error']==0 for z in progress),'Original row mass changed')
            exact([z['M_mass'] for z in progress]==[reg['schedule'][fold]['class_mass'][1]]*25 and [z['S_mass'] for z in progress]==[reg['schedule'][fold]['class_mass'][2]]*25,'Class mass trajectory changed')
            for z in checkpoints:
                exact(z['model_sha256']==digest(folder/f"epoch{z['epoch']}_model.pt") and z['prob_sha256']==digest(folder/f"epoch{z['epoch']}_prob.npy"),'Diagnostic checkpoint changed')
            exact(fit['optimizer_steps']==reg['schedule'][fold]['steps_per_arm'],'Fit steps changed')
            total_steps+=fit['optimizer_steps'];files.append({'fold':fold,'arm':arm,'steps':fit['optimizer_steps'],'receipt_sha256':digest(folder/'fit.json')})
    exact(total_steps==13350 and len(files)==9,'Incomplete nine fits')
    official=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['event_id','label_binary'])
    ledger=pd.read_parquet(OUT/'full_prediction_ledger.parquet')
    exact(len(official)==len(ledger)==2056871 and np.array_equal(ledger.row_position.to_numpy(),np.arange(len(ledger))),'Full row identity changed')
    truth=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(dtype=np.int8)
    exact(np.isin(truth,[0,1,2]).all(),'Unknown official label')
    predicted={arm:ledger[f'final_pred_{arm}'].to_numpy(dtype=np.int8) for arm in 'ABC'}
    for arm,p in predicted.items():
        exact(np.isin(p,[0,1,2]).all(),'Invalid prediction')
        by=counts(truth,p)
        for cls in '012':
            for key,val in by[cls].items():
                expected=primary['metrics']['full_task'][arm][cls][key]
                exact(val==expected or (val is not None and expected is not None and abs(val-expected)<1e-12),'Full class recount mismatch')
    asa=ledger.gate_open.to_numpy(dtype=bool)
    expert=pd.read_parquet(OUT/'expert_ASA_predictions.parquet')
    exact(int(asa.sum())==len(expert)==112807 and np.array_equal(ledger.row_position.to_numpy()[asa],expert.row_position.to_numpy()),'ASA identity changed')
    exact((~asa).sum()==1944064 and int((ledger.teacher_prediction.to_numpy()[~asa]!=truth[~asa]).sum())==107,'Frozen non-ASA population changed')
    for arm in 'ABC':
        p=expert[f'expert_pred_{arm}'].to_numpy(dtype=np.int8)
        exact(np.array_equal(predicted[arm][asa],p),'ASA predictions do not match full task')
        by=counts(truth[asa],p)
        for cls in '12':
            for key,val in by[cls].items():
                expected=primary['metrics']['ASA_expert'][arm][cls][key]
                exact(val==expected or (val is not None and expected is not None and abs(val-expected)<1e-12),'ASA class recount mismatch')
    exact(primary['primary_quality_passed'] is False and primary['confirmation_allowed'] is False and primary['classifier_fits_new']==9,'Incorrect quality/confirmation status')
    exact(all(v is False or k in ('complete_full_population','all_required_classes_present','M_S_present_in_ASA') for k,v in primary['quality']['gates'].items()),'Unexpected passing quality gate')
    ablation=obj(OUT/'postmortem_ablation.json');coverage=obj(OUT/'postmortem_coverage.json')
    exact(ablation['primary_evaluation_sha256']==digest(OUT/'primary_evaluation.json') and coverage['A_correct_S_to_B_M']['original_rows']==2046,'Postfit diagnosis changed')
    report={'status':'independent_audit_passed_no_promotion','official_full_rows':len(official),
            'ASA_rows':int(asa.sum()),'source_bound_files':len(seal['source_sha256']),
            'fit_receipts':files,'fits':9,'optimizer_steps':total_steps,
            'official_class_support':{str(i):int((truth==i).sum()) for i in range(3)},
            'recounted_full':{arm:counts(truth,predicted[arm]) for arm in 'ABC'},
            'recounted_ASA':{arm:counts(truth[asa],expert[f'expert_pred_{arm}'].to_numpy(dtype=np.int8)) for arm in 'ABC'},
            'non_ASA_frozen_errors':107,'model_replay_max_abs_diff':max(v['max_abs_probability_diff'] for v in primary['model_replays']),
            'primary_quality_passed':False,'confirmation_fits':0,'model_promoted':False,
            'source_sha256':digest(ROOT/'training/v125_verify.py')}
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'fits':9,'steps':total_steps,'rows':len(official),'source_bound_files':report['source_bound_files']},ensure_ascii=False))


if __name__=='__main__':main()
