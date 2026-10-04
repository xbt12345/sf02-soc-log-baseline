"""V127-specific runtime checks; reuses the project's bound-source checker."""
import json
from pathlib import Path

from experiment_review import check_bindings, sha, ReviewError
from v127_verify_plan_evidence import ROOT, PLAN, review_plan, read


def check_plan():
    plan = read(PLAN)
    review_plan(plan)
    evidence = ROOT / 'artifacts/v127_plan_review_20260929/verification.json'
    receipt = read(evidence)
    if receipt['status'] != 'V127_plan_evidence_verified_not_training_runtime':
        raise ReviewError('V127 prior evidence missing')
    check_bindings(receipt['artifact_sha256'])
    return plan


def seal_run(trainer, sources, destination):
    if destination.exists():
        raise FileExistsError(destination)
    plan = check_plan()
    files = {Path(trainer).resolve(), Path(__file__).resolve(),
             (ROOT / 'training/experiment_review.py').resolve(), PLAN.resolve()}
    files |= {Path(x).resolve() for x in sources}
    bound = {}
    for path in sorted(files):
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise ReviewError('Bad source in V127 execution seal: ' + str(path))
        bound[path.relative_to(ROOT).as_posix()] = sha(path)
    seal = {'status': 'sealed_before_fit', 'plan_sha256': sha(PLAN),
            'trainer': Path(trainer).resolve().relative_to(ROOT).as_posix(),
            'source_sha256': bound, 'quality_acceptance': False,
            'model_promoted': False, 'scope': 'V127 specific 9-fit primary trial only'}
    destination.write_text(json.dumps(seal, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return seal


def require_run_seal(destination, trainer):
    seal = read(destination)
    if seal.get('status') != 'sealed_before_fit' or (ROOT / seal['trainer']).resolve() != Path(trainer).resolve():
        raise ReviewError('Wrong V127 run seal')
    if seal['plan_sha256'] != sha(PLAN):
        raise ReviewError('V127 plan changed')
    check_bindings(seal['source_sha256'])
    plan = read(PLAN)
    review_plan(plan)
    return plan


def require_checkpoint(plan, receipt):
    if (receipt.get('status') != 'fit_executed' or receipt.get('completed_epochs') != plan['training']['epochs']
            or receipt.get('prediction_epoch') != plan['training']['epochs']):
        raise ReviewError('Incomplete or unregistered V127 endpoint')
