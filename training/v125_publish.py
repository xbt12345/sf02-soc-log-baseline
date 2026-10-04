"""Seal the completed but quality-failed V125 experiment as inspectable evidence."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v125_order_trial_20260929'
REPORT=ROOT/'docs/V125_ORDERED_BODY_TRAINING_RESULTS_AND_STOP.md'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    target=OUT/'delivery.json'
    if target.exists():raise FileExistsError(target)
    evaluation=json.loads((OUT/'primary_evaluation.json').read_text(encoding='utf-8'))
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    if evaluation['primary_quality_passed'] or evaluation['confirmation_allowed'] or verification['status']!='independent_audit_passed_no_promotion':
        raise ValueError('V125 delivery status disagrees with observed failure')
    if not REPORT.is_file():raise FileNotFoundError(REPORT)
    files=sorted(p for p in OUT.rglob('*') if p.is_file())
    artifact={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    sources=sorted(ROOT/'training'/name for name in (
        'v125_input.py','v125_model.py','v125_train.py','v125_evaluate.py',
        'v125_experiment_review.py','v125_postmortem.py','v125_coverage_audit.py',
        'v125_verify.py','v125_publish.py','v125_replay_counterexamples.py'))
    manifest={'id':'v125-delivery','status':'full_nine_fit_trial_verified_quality_failed',
              'latest_actual_training':'V125','date':'2026-09-29',
              'classifier_fits':9,'optimizer_steps':13350,'calibration_fits':0,
              'confirmation_fits':0,'quality_acceptance':False,'model_promoted':False,
              'official_full_rows_scored':2056871,'ASA_rows_scored':112807,
              'validation_scope':'Repeatedly inspected local source-closed development folds, with complete recount over 2,056,871 official training rows; no independent external or official-platform test.',
              'primary_evaluation':'artifacts/v125_order_trial_20260929/primary_evaluation.json',
              'independent_verification':'artifacts/v125_order_trial_20260929/verification.json',
              'report':'docs/V125_ORDERED_BODY_TRAINING_RESULTS_AND_STOP.md',
              'artifact_sha256':artifact,
              'source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},
              'report_sha256':digest(REPORT),
              'registered_plan_sha256':digest(ROOT/'training/review_policy/v125_next_training_plan.json'),
              'risk_addendum':'training/review_policy/v125_outcome_cases.json',
              'risk_addendum_sha256':digest(ROOT/'training/review_policy/v125_outcome_cases.json'),
              'limitations':['Repeatedly inspected development folds, no new blind external test.',
                             'Failed B/C models are retained for diagnosis and must not be promoted.']}
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'files':len(artifact),'artifact_bytes':sum(p.stat().st_size for p in files),
                      'delivery_sha256':digest(target)},ensure_ascii=False))


if __name__=='__main__':main()
