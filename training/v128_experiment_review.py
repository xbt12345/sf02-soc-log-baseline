"""V128 execution contract. Preparation and quality are separate decisions."""
import json
from pathlib import Path

from experiment_review import ROOT, ReviewError, check_bindings, sha

PLAN = ROOT / 'training/review_policy/v128_next_training_plan.json'
REVIEW = ROOT / 'artifacts/v128_mechanism_review_20260929/verification.json'
ROLES = ROOT / 'artifacts/v128_mechanism_review_20260929/nested_score_roles.parquet'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def check_plan():
    p = read(PLAN)
    r = read(REVIEW)
    if p.get('version') != 'V128' or p.get('matched_primary', {}).get('primary_fits_including_teachers') != 21:
        raise ReviewError('Wrong V128 21-fit plan')
    if p['teacher_bank']['new_fits'] != 9 or p['matched_primary']['branch_checkpoints'] != [0,1,2,5,10,15,20,25,35,50]:
        raise ReviewError('Teacher or endpoint changed')
    if p['quality']['ASA_M_errors_max'] != 318 or p['quality']['ASA_S_errors_max'] != 2074:
        raise ReviewError('Historical M/S gate changed')
    if r.get('status') != 'V128_no_fit_evidence_and_role_preparation_verified':
        raise ReviewError('V128 role audit missing')
    check_bindings(r['artifact_sha256'])
    if sha(ROLES) != p['data']['new_inner_manifest_sha256']:
        raise ReviewError('Role manifest changed')
    return p


def seal_run(trainer, sources, output):
    if Path(output).exists():
        raise FileExistsError(output)
    check_plan()
    files = {Path(trainer).resolve(), Path(__file__).resolve(),
             (ROOT / 'training/experiment_review.py').resolve(), PLAN.resolve(), REVIEW.resolve(), ROLES.resolve()}
    files.update(Path(x).resolve() for x in sources)
    bound = {}
    for path in sorted(files):
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise ReviewError('Unbound/missing V128 dependency: ' + str(path))
        bound[path.relative_to(ROOT).as_posix()] = sha(path)
    receipt = {'status':'sealed_before_fit', 'version':'V128',
               'trainer':Path(trainer).resolve().relative_to(ROOT).as_posix(),
               'plan_sha256':sha(PLAN), 'source_sha256':bound,
               'quality_acceptance':False, 'model_promoted':False}
    save(output, receipt)
    return receipt


def require_run_seal(output, trainer):
    seal = read(output)
    if seal.get('status') != 'sealed_before_fit' or seal.get('version') != 'V128':
        raise ReviewError('V128 run not registered')
    if (ROOT / seal['trainer']).resolve() != Path(trainer).resolve() or seal['plan_sha256'] != sha(PLAN):
        raise ReviewError('V128 trainer or plan changed')
    check_bindings(seal['source_sha256'])
    return check_plan()


def require_checkpoint(plan, receipt, kind):
    epochs = 25 if kind == 'teacher' else 50
    if (receipt.get('status') != 'fit_executed' or receipt.get('completed_epochs') != epochs
            or receipt.get('prediction_epoch') != epochs):
        raise ReviewError('Incomplete or selected early V128 endpoint')
