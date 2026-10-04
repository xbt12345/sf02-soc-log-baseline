"""Independent receipts/count checks for V110 failed developmental experiment."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT, save, sha
from v110_layer_probes import DEST, LEDGER, PLAN, read_json


def main():
    assert not (DEST/'verification.json').exists()
    r=read_json(DEST/'registration.json');e=read_json(DEST/'evaluation.json')
    p=read_json(DEST/'postfit_diagnosis_r2.json');f=read_json(DEST/'original_readout_feasibility.json')
    d=pd.read_parquet(LEDGER,columns=['row_position','root','fold','truth','local','N1_TabM25','N2_TabM25'])
    t=pd.read_parquet(DEST/'OOF_probe_comparison.parquet')
    checks={'registration_before_fit':r['status']=='registered_before_any_fit' and r['classifier_fits_at_registration']==0,
      'source_unchanged':sha(ROOT/'training/v110_layer_probes.py')==r['source_sha256'] and sha(ROOT/'training/v110_layer_probes_evaluate.py')==r['evaluation_source_sha256'],
      'all_input_hashes':all(sha(ROOT/q)==h for q,h in r['input_sha256'].items()),
      'contract_unchanged':sha(PLAN)==r['contract_sha256'],
      'model_source_unchanged':sha(ROOT/'training/v104_phase_b.py')==r['model_source_sha256'],
      'zero_extra_fits':e['actual_classifier_fits']==6 and p['classifier_fits']==f['classifier_fits']==0,
      'row_order':len(d)==len(t)==112807 and np.array_equal(d.row_position.to_numpy(),t.row_position.to_numpy()),
      'OOF_hash':sha(DEST/'OOF_probe_comparison.parquet')==e['OOF_ledger_sha256']==p['OOF_sha256'],
      'model_not_promoted':e['model_promoted'] is False and all(not x['formal_model_replacement'] for x in e['gates'].values()),
      'source_roots_fold_closed':int(d.groupby('root').fold.nunique().max())==1,
      'r2_port_correction_documented':'supersedes' in p and 'incorrectly empty' in p['supersedes'],
      'feasible_old_readout':all(x['original_mean_logit_linear_reconstruction_max_abs_diff']<1e-5 and
        x['objective_new_converged_P2_head']<x['objective_original_mean_logit_head'] for x in f['rows'])}
    y=d.truth.to_numpy();fold=d.fold.to_numpy();local=d.local.to_numpy()
    for arm in ('P1','P2'):
        pred=t[arm+'_prediction'].to_numpy();score=t[arm+'_S_probability'].to_numpy()
        checks[arm+'_fixed_threshold']=np.array_equal(pred,np.where(score>=.5,2,1))
        checks[arm+'_per_class_errors']=all(int(((y==c)&(pred!=c)).sum())==e['variants'][arm]['ASA']['class'][n]['missed']
          for c,n in [(1,'malicious'),(2,'suspicious')])
        checks[arm+'_gate_honest']=not e['gates'][arm]['research_continuation_preliminary'] and not e['gates'][arm]['N1_full_task_replacement_count_entry']
        checks[arm+'_full_task_count']=e['full_task'][arm]['rows']==2056871 and e['full_task'][arm]['class']['benign']['missed']==0
        checks[arm+'_no_port_r2']=p['support_slices'][arm]['no_fixed_destination_port_in_behavior']['S_rows']==1597
        for k in (0,1,2):
            path=DEST/f'fold{k}_{arm}';fit=read_json(path/'fit.json')
            held=fold==k;train=~held
            checks[f'{arm}_fold{k}_fit_integrity']=bool(fit['converged_to_registered_gradient_gate']
                and fit['gradient_inf']<=1e-5 and fit['iterations']<=2000
                and fit['registration_sha256']==sha(DEST/'registration.json')
                and fit['source_sha256']==r['source_sha256']
                and sha(path/'probe.npz')==fit['probe_sha256']
                and sha(path/'ASA_input_S_probability.npy')==fit['prob_sha256'])
            checks[f'{arm}_fold{k}_training_mass']=fit['train_original_rows']==int(train.sum())
            s=np.load(path/'ASA_input_S_probability.npy')
            checks[f'{arm}_fold{k}_OOF_replay']=np.array_equal(pred[held],np.where(s[local[held]]>=.5,2,1))
            checks[f'{arm}_fold{k}_M_S_mass']=fit['train_M_rows']==int((train&(y==1)).sum()) and fit['train_S_rows']==int((train&(y==2)).sum())
    checks['N1_original_full_baseline']=e['full_task']['N1_reference']['errors']==2519
    checks['all_official_population']=r['all_official_rows']==2056871 and e['class_population_ASA']==[0,78748,34059]
    report={'all_checks_passed':all(checks.values()),'checks':checks,'checks_count':len(checks),
      'new_classifier_fits':6,'new_calibration_fits':0,'model_promoted':False,
      'scope':'Real trained probe receipts, saved OOF and full-count recount; development-only quality, no external test.',
      'artifact_sha256':{q.relative_to(ROOT).as_posix():sha(q) for q in [DEST/'registration.json',DEST/'evaluation.json',
        DEST/'OOF_probe_comparison.parquet',DEST/'postfit_diagnosis_r2.json',DEST/'original_readout_feasibility.json']},
      'source_sha256':sha(__file__)}
    save(DEST/'verification.json',report)
    assert report['all_checks_passed'],{k:v for k,v in checks.items() if not v}
    print(json.dumps({'all_checks_passed':True,'checks_count':len(checks),'new_fits':6,'model_promoted':False}))


if __name__=='__main__':main()
