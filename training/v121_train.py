"""V120 registered paired A/B experiment. Fit only after a complete run seal."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from experiment_review import ROOT, read, sha, review_plan, seal_run, require_run_seal, require_checkpoint
from v104_phase_b import SparseTabM, csr_tensor, predict_all, K, BATCH, DEVICE
from v75_views import BYTE_FEATURES
from v117_gradient_batch_audit import stratified
from v107_matched_training import FOLDS, ROWS, FID, DEST as TEACHERS
from v116_preflight import VIEW

PLAN = ROOT / 'training/review_policy/v120_next_batch_plan.json'
DEST = ROOT / 'artifacts/v121_paired_batch_training_20260929'
MANIFEST = ROOT / 'artifacts/v116_nested_selection_20260929/inner_split_manifest.parquet'
OFFICIAL = ROOT / 'data/official/train.parquet'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
ORDER = [(0, 'A'), (0, 'B'), (1, 'B'), (1, 'A'), (2, 'A'), (2, 'B')]
CHECKPOINTS = (1, 2, 5, 10, 15, 20, 25)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def root_relative_sources():
    paths = {Path(__file__), PLAN, VIEW, MANIFEST, OFFICIAL, TRACE, ROWS, FOLDS, FID}
    for fold in range(3):
        folder = TEACHERS / f'fold{fold}_N1_teacher'
        paths.update((folder / 'scores_all_input_ids.npy', folder / 'fit.json'))
    # All local Python modules actually imported by this runner, including transitive imports.
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if (p.suffix == '.py' and p.is_file() and p.is_relative_to(ROOT)
                    and not p.relative_to(ROOT).parts[0].startswith('.venv')):
                paths.add(p)
    paths.add(ROOT / 'training/v121_evaluate.py')
    paths.add(ROOT / 'training/v121_confirm.py')
    return sorted(paths)


def require_v120(plan):
    if not review_plan(plan)['plan_review_passed']:
        raise ValueError('Base V119 plan review failed.')
    if plan['experiment_id'] != 'V120_N1_mass_stratified_batches':
        raise ValueError('Wrong experiment.')
    contract = plan['batch_contract']
    if (contract['A'] != 'uniform_unique_input_batches'
            or contract['B'] != 'original_class_mass_stratified_batches'
            or contract['updates_per_logical_batch'] != 1
            or contract['default_microbatching'] is not False
            or plan['training_epochs'] != 25 or plan['selector']['epoch'] != 25
            or plan['primary_fits'] != 6 or plan['primary_seed'] != 10201):
        raise ValueError('Unregistered V120 change.')
    if plan['evaluation_contract']['raw_ASA_rows'] != 112807 or plan['evaluation_contract']['full_task_rows'] != 2056871:
        raise ValueError('Population registration changed.')
    if plan['architecture'] != {'input':'unchanged_N1','model':'SparseTabM','members':16,
                                 'dtype':'float32','loss':'original_row_frequency_member_mean_three_class_CE'}:
        raise ValueError('Input/model/loss changed.')
    if plan['optimizer'] != {'name':'AdamW','lr':.002,'weight_decay':.0003,'same_initial_state_per_pair':True}:
        raise ValueError('Optimizer changed.')


def counts_for_fold(manifest, n_input, fold):
    fit = manifest[(manifest.outer_fold == fold) & (manifest.fold != fold)]
    test = manifest[(manifest.outer_fold == fold) & (manifest.fold == fold)]
    if set(fit.root) & set(test.root):
        raise ValueError('Root leakage into heldout fold.')
    counts = np.bincount(fit.local.to_numpy(dtype=np.int64) * 3 + fit.truth.to_numpy(dtype=np.int64),
                         minlength=n_input * 3).reshape(n_input, 3)
    if len(fit) != counts.sum() or counts[:, 0].any() or set(fit.truth.unique()) != {1, 2}:
        raise ValueError('ASA class or original-row mass mismatch.')
    return fit, counts


def initial_state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode('utf-8')); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def checkpoint_diagnostics(model, x, counts, used, probability, device):
    y_mass = counts[used].astype(np.float64)
    member_ce = np.zeros(3); ensemble_ce = np.zeros(3); brier = np.zeros(3)
    none_correct = np.zeros(3, dtype=np.int64)
    nonconflict_errors = np.zeros(3, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(used), 256):
            ids = used[start:start+256]
            block = x[ids]
            fact = torch.as_tensor(block[:, BYTE_FEATURES:].toarray(), device=device, dtype=torch.float32)
            z = model(csr_tensor(block, device), fact)
            log_member = torch.log_softmax(z, -1).double().mean(1).cpu().numpy()
            member_prediction = z.argmax(-1).cpu().numpy()
            p = probability[ids].astype(np.float64)
            weight = counts[ids].astype(np.float64)
            unique_label = (weight[:, 1] > 0) ^ (weight[:, 2] > 0)
            for cls in (1, 2):
                member_ce[cls] -= float(np.sum(weight[:, cls] * log_member[:, cls]))
                ensemble_ce[cls] -= float(np.sum(weight[:, cls] * np.log(np.clip(p[:, cls], 1e-12, 1))))
                target = np.zeros(3); target[cls] = 1
                brier[cls] += float(np.sum(weight[:, cls] * np.square(p-target).sum(1)))
                none_correct[cls] += int(np.sum(weight[:, cls] * (member_prediction != cls).all(1)))
                nonconflict_errors[cls] += int(np.sum(weight[:, cls] * unique_label * (p.argmax(1) != cls)))
    pred = probability[used].argmax(1)
    return {str(c): {'support': int(counts[:, c].sum()),
                     'correct': int(np.sum(counts[used,c] * (pred == c))),
                     'member_mean_CE': float(member_ce[c] / max(1, counts[:, c].sum())),
                     'ensemble_CE': float(ensemble_ce[c] / max(1, counts[:, c].sum())),
                     'Brier': float(brier[c] / max(1, counts[:, c].sum())),
                     'no_correct_member_rows': int(none_correct[c]),
                     'nonconflict_error_rows': int(nonconflict_errors[c])} for c in (1, 2)}


def register():
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError('Existing V121 output must not be overwritten.')
    plan = read(PLAN); require_v120(plan)
    if not torch.cuda.is_available() or DEVICE != 'cuda':
        raise RuntimeError('Registered CUDA execution unavailable.')
    x = sparse.load_npz(VIEW)
    manifest = pd.read_parquet(MANIFEST)
    schedule = []
    for fold in range(3):
        fit, counts = counts_for_fold(manifest, x.shape[0], fold)
        used = np.flatnonzero(counts.sum(1)); batches = math.ceil(len(used) / BATCH)
        expected = plan['expected_primary_schedule'][fold]
        actual = {'fold': fold, 'train_rows': len(fit), 'class_mass': counts.sum(0).tolist(),
                  'unique_inputs': len(used), 'logical_batches_per_epoch': batches,
                  'registered_optimizer_steps_per_arm': 25*batches}
        if any(expected[k] != v for k, v in actual.items()):
            raise ValueError('Schedule changed before fitting: ' + str(actual))
        schedule.append(actual)
    import tabm
    tabm_path = Path(tabm.__file__).resolve()
    package = {'torch': torch.__version__, 'numpy': np.__version__,
               'pandas': pd.__version__, 'scipy': importlib.metadata.version('scipy'),
               'tabm': importlib.metadata.version('tabm'),
               'tabm_source_path': str(tabm_path), 'tabm_source_sha256': sha(tabm_path),
               'cuda_runtime': torch.version.cuda, 'cuda_device': torch.cuda.get_device_name(0)}
    DEST.mkdir(exist_ok=True)
    seal_run(PLAN, Path(__file__), root_relative_sources(), DEST/'run_seal.json')
    save(DEST/'registration.json', {'status':'registered_before_fits','created_unix':time.time(),
        'plan_sha256':sha(PLAN),'seal_sha256':sha(DEST/'run_seal.json'),
        'fit_order': ORDER,'schedule':schedule,'package_identity':package,
        'quality_acceptance':False,'model_promoted':False,'classifier_fits_at_registration':0,
        'evaluation_contract':'Raw expert ASA plus frozen teacher route on all original official rows.',
        'limitations':['Source inventory includes loaded project Python modules and explicit data dependencies; no claim of hermetic environment.',
                       'Previously inspected folds remain developmental.']})
    print(json.dumps({'stage':'registered','fits':6,'device':package['cuda_device'],'schedule':schedule}),flush=True)


def check_registration():
    plan = require_run_seal(DEST/'run_seal.json', Path(__file__))
    require_v120(plan)
    reg = read(DEST/'registration.json')
    if reg['status'] != 'registered_before_fits' or reg['plan_sha256'] != sha(PLAN) or reg['seal_sha256'] != sha(DEST/'run_seal.json'):
        raise ValueError('Changed run registration.')
    import tabm
    if sha(Path(tabm.__file__)) != reg['package_identity']['tabm_source_sha256']:
        raise ValueError('Model package changed.')
    return plan, reg


def fit(fold, arm, seed=10201):
    plan, reg = check_registration()
    if [fold, arm] not in reg['fit_order']:
        raise ValueError('Unregistered arm or fold.')
    if seed != plan['primary_seed']:
        if seed not in plan['confirmation_seeds']:
            raise ValueError('Unregistered confirmation seed.')
        primary = read(DEST/'primary_evaluation.json')
        if not primary['confirmation_allowed'] or primary['classifier_fits_new'] != 6:
            raise ValueError('All primary quality gates must pass before confirmation.')
    folder = DEST / (f'fold{fold}_{arm}' if seed == plan['primary_seed'] else f'seed{seed}_fold{fold}_{arm}')
    if folder.exists():
        if (folder/'fit.json').exists():
            receipt = read(folder/'fit.json')
            require_checkpoint(plan, receipt)
            if receipt['model_sha256'] != sha(folder/'epoch25_model.pt'):
                raise ValueError('Existing completed model identity changed.')
            return
        raise FileExistsError('Interrupted fit; preserve evidence and diagnose before restarting: ' + str(folder))
    manifest = pd.read_parquet(MANIFEST)
    x = sparse.load_npz(VIEW)
    fit_rows, counts = counts_for_fold(manifest, x.shape[0], fold)
    used = np.flatnonzero(counts.sum(1))
    batches = math.ceil(len(used) / BATCH)
    schedule = reg['schedule'][fold]
    if (len(fit_rows) != schedule['train_rows'] or len(used) != schedule['unique_inputs']
            or counts.sum(0).tolist() != schedule['class_mass'] or batches != schedule['logical_batches_per_epoch']):
        raise ValueError('Training schedule drifted.')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model = SparseTabM().to(DEVICE)
    init_hash = initial_state_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=plan['optimizer']['lr'],
                            weight_decay=plan['optimizer']['weight_decay'])
    rng = np.random.default_rng(seed + fold)
    folder.mkdir()
    save(folder/'started.json', {'status':'started','seed':seed,'fold':fold,'arm':arm,
         'initial_state_sha256':init_hash,'train_rows':len(fit_rows),'train_mass':counts.sum(0).tolist(),
         'seal_sha256':sha(DEST/'run_seal.json'),'heldout_gradient_rows':0,'started_unix':time.time(),
         'primary_evaluation_sha256':sha(DEST/'primary_evaluation.json') if seed != plan['primary_seed'] else None})
    normalizer = len(fit_rows)/batches
    history = []; checkpoints = []; start = time.monotonic(); steps = 0
    for epoch in range(1, plan['training_epochs']+1):
        require_run_seal(DEST/'run_seal.json', Path(__file__))
        model.train();mass_seen=np.zeros_like(counts,dtype=np.float64)
        if arm == 'A':
            order = used[rng.permutation(len(used))]
            logical_batches = [(order[i:i+BATCH],counts[order[i:i+BATCH]].astype(np.float64))
                               for i in range(0,len(used),BATCH)]
        else:
            logical_batches = stratified(counts,batches,rng)
        if len(logical_batches) != batches:
            raise ValueError('Batch count changed.')
        numerator_total=0.; max_unique=0
        for ids, weights in logical_batches:
            if not len(ids) or len(ids) != len(np.unique(ids)) or not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError('Invalid batch content.')
            mass_seen[ids] += weights
            max_unique=max(max_unique,len(ids))
            block=x[ids]
            fact=torch.as_tensor(block[:, BYTE_FEATURES:].toarray(),device=DEVICE,dtype=torch.float32)
            mass=torch.as_tensor(weights,device=DEVICE,dtype=torch.float32)
            z=model(csr_tensor(block,DEVICE),fact)
            numerator=-(torch.log_softmax(z,-1)*mass[:,None,:]).sum()/K
            loss=numerator/normalizer
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError('Nonfinite training loss.')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();steps+=1
            numerator_total += float(numerator.detach().item())
        mass_error=float(np.max(np.abs(mass_seen-counts)))
        if mass_error > 1e-8 or steps != epoch*batches:
            raise ValueError('Original input/class mass or optimizer steps changed.')
        item={'epoch':epoch,'optimizer_steps_cumulative':steps,'logical_batches':batches,
              'maximum_logical_unique_inputs':max_unique,'mass_reconstruction_max_error':mass_error,
              'online_original_row_CE':numerator_total/len(fit_rows),'seconds_elapsed':time.monotonic()-start,
              'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
        history.append(item);save(folder/'progress.json',history)
        if epoch in CHECKPOINTS:
            model.eval()
            probability=predict_all(model,'TabM',x,DEVICE)
            np.save(folder/f'epoch{epoch}_prob.npy',probability)
            ckpt={'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                  'seed':seed,'fold':fold,'arm':arm,'epoch':epoch,'seal_sha256':sha(DEST/'run_seal.json')}
            torch.save(ckpt,folder/f'epoch{epoch}_model.pt')
            diagnostic=checkpoint_diagnostics(model,x,counts,used,probability,DEVICE)
            checkpoint={'epoch':epoch,'model_sha256':sha(folder/f'epoch{epoch}_model.pt'),
                        'prob_sha256':sha(folder/f'epoch{epoch}_prob.npy'),
                        'fit_by_class':diagnostic}
            checkpoints.append(checkpoint);save(folder/'checkpoints.json',checkpoints)
            print(json.dumps({'stage':'checkpoint','fold':fold,'arm':arm,'epoch':epoch,
                              'steps':steps,'fit_S_correct':diagnostic['2']['correct'],
                              'fit_M_correct':diagnostic['1']['correct'],
                              'seconds':round(time.monotonic()-start,1)}),flush=True)
    require_checkpoint(plan,{'status':'fit_executed','completed_epochs':25,'prediction_epoch':25})
    save(folder/'fit.json',{'status':'fit_executed','fold':fold,'arm':arm,'seed':seed,
         'completed_epochs':25,'prediction_epoch':25,'optimizer_steps':steps,
         'train_rows':len(fit_rows),'train_mass':counts.sum(0).tolist(),
         'initial_state_sha256':init_hash,'model_sha256':sha(folder/'epoch25_model.pt'),
         'prob_sha256':sha(folder/'epoch25_prob.npy'),'checkpoint_report_sha256':sha(folder/'checkpoints.json'),
         'progress_sha256':sha(folder/'progress.json'),'seal_sha256':sha(DEST/'run_seal.json'),
         'seconds':time.monotonic()-start,'quality_acceptance':False})
    print(json.dumps({'stage':'fit_complete','fold':fold,'arm':arm,'steps':steps,
                      'seconds':round(time.monotonic()-start,1)}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('register','run','fit','confirm'))
    parser.add_argument('--fold',type=int);parser.add_argument('--arm',choices=('A','B'))
    parser.add_argument('--seed',type=int,default=10201)
    args=parser.parse_args()
    if args.stage=='register':register()
    elif args.stage=='fit':fit(args.fold,args.arm,args.seed)
    elif args.stage=='run':
        for fold,arm in ORDER:
            fit(fold,arm)
    else:
        plan, _ = check_registration()
        for seed in plan['confirmation_seeds']:
            for fold,arm in ORDER:
                fit(fold,arm,seed)


if __name__=='__main__':main()
