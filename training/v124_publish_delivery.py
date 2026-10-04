"""Seal executed V124 artifacts after independent evaluation; no fitting."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v124_header_trial_20260929'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    target=OUT/'delivery.json'
    if target.exists():raise FileExistsError(target)
    audit=read(OUT/'independent_postfit_audit.json')
    quality=read(OUT/'primary_evaluation.json')
    assert audit['classifier_fits_new']==quality['classifier_fits_new']==6
    assert audit['optimizer_steps']==quality['optimizer_steps']==8900
    assert not audit['primary_quality_passed'] and not quality['confirmation_allowed']
    assert audit['model_promoted'] is False and audit['quality_acceptance'] is False
    assert all(not any(OUT.glob(f'seed{seed}_fold*_*/fit.json')) for seed in [10202,10203])
    for rel,h in read(OUT/'run_seal.json')['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    files=[p for p in OUT.rglob('*') if p.is_file() and p!=target]
    files += [ROOT/p for p in [
        'docs/V124_HEADER_TRAINING_RESULTS_AND_STOP.md',
        'training/review_policy/v124_header_trial.json',
        'training/v124_experiment_review.py','training/v124_header.py',
        'training/v124_prepare_header_input.py','training/v124_qualification.py',
        'training/v124_train.py','training/v124_evaluate.py',
        'training/v124_postfit_audit.py','training/v124_fit_support_probe.py',
        'training/v124_publish_delivery.py','training/test_v124_input_review.py']]
    files=sorted(set(files))
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    assert len(bindings)>80
    delivery={'status':'v124_six_fits_executed_failed_quality_gates',
        'summary':'V124 six paired fits completed; header input contract repaired, classification gates failed and no model was promoted.',
        'latest_actual_training':'V124','classifier_fits_new':6,'optimizer_steps':8900,
        'calibration_fits':0,'confirmation_fits':0,'quality_acceptance':False,
        'model_promoted':False,'input_contract_repaired':True,
        'ASA_M_errors_A_B':[audit['ASA_per_class'][x]['1']['missed'] for x in ('A','B')],
        'ASA_S_errors_A_B':[audit['ASA_per_class'][x]['2']['missed'] for x in ('A','B')],
        'failed_gates':[k for k,v in audit['gates'].items() if not v],
        'actual_classification_source':'artifacts/v124_header_trial_20260929/independent_postfit_audit.json',
        'report':'docs/V124_HEADER_TRAINING_RESULTS_AND_STOP.md',
        'artifact_sha256':bindings,
        'validation_scope':'Repeatedly inspected local source-closed development folds, scored over all 2,056,871 official training rows; no independent external test.',
        'scope':'Local development folds and complete official training rows; no independent external test or platform submission.'}
    target.write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in delivery.items() if k!='artifact_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
