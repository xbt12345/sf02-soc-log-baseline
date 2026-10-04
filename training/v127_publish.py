"""Seal completed V127 training as failed quality evidence, without model promotion."""
import json

from v126_frozen_audit import ROOT, sha
from v127_train import OUT, PLAN

REPORT=ROOT/'docs/V127_FROZEN_BRANCH_TRAINING_RESULTS_AND_STOP.md'
RISKS=ROOT/'training/review_policy/v127_outcome_cases.json'


def main():
    target=OUT/'delivery.json'
    if target.exists():raise FileExistsError(target)
    evaluation=json.loads((OUT/'primary_evaluation.json').read_text(encoding='utf-8'))
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    risks=json.loads(RISKS.read_text(encoding='utf-8'))
    if (evaluation['status']!='v127_nine_fit_source_closed_evaluated'
            or verification['status']!='V127_independent_full_row_audit_passed'
            or evaluation['primary_quality_passed'] or verification['quality_gates_passed']
            or evaluation['confirmation_allowed'] or risks['confirmation_executed']
            or evaluation['ASA_total_errors']!={'A0':2392,'AH':2432,'K':2256,'W':2298,'P':2306}
            or not REPORT.is_file()):
        raise ValueError('Delivery conflicts with actual quality failure or missing report')
    files=sorted(q for q in OUT.rglob('*') if q.is_file())
    sources=sorted(ROOT/'training'/name for name in
        ('v127_model.py','v127_experiment_review.py','v127_train.py','v127_evaluate.py',
         'v127_dynamics.py','v127_verify.py','v127_publish.py',
         'v127_mechanism_probe.py','v127_header_projection_probe.py',
         'v127_verify_plan_evidence.py'))
    manifest={'id':'v127-delivery','status':'v127_nine_fit_trial_verified_quality_failed',
              'latest_actual_training':'V127','date':'2026-09-29',
              'classifier_fits':9,'network_fits':6,'constant_fits':3,
              'network_optimizer_steps':17800,
              'constant_optimizer_iterations':evaluation['constant_optimizer_iterations'],
              'calibration_fits':0,'confirmation_fits':0,'quality_acceptance':False,
              'model_promoted':False,'official_full_rows_scored':2056871,'ASA_rows_scored':112807,
              'primary_evaluation':'artifacts/v127_frozen_branch_trial_20260929/primary_evaluation.json',
              'independent_verification':'artifacts/v127_frozen_branch_trial_20260929/verification.json',
              'mechanism_diagnostics':'artifacts/v127_frozen_branch_trial_20260929/branch_dynamics.json',
              'report':REPORT.relative_to(ROOT).as_posix(),
              'report_sha256':sha(REPORT),'risk_addendum':RISKS.relative_to(ROOT).as_posix(),
              'risk_addendum_sha256':sha(RISKS),'registered_plan_sha256':sha(PLAN),
              'artifact_sha256':{q.relative_to(ROOT).as_posix():sha(q) for q in files},
              'source_sha256':{q.relative_to(ROOT).as_posix():sha(q) for q in sources},
              'validation_scope':'Repeatedly inspected local source-closed development folds; no independent external or official-platform evaluation.',
              'limitations':['P is not accepted and none of K/W/P is promoted.',
                  'Total-error improvement is concentrated in source group29 and M recall falls.',
                  'Recorded parameter and representation learning did not deliver stable input-specific source-held decisions.']}
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'fits':manifest['classifier_fits'],
                      'artifacts':len(files),'sha256':sha(target)},ensure_ascii=False))


if __name__=='__main__':main()
