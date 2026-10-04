"""v6.3.1 registered four-arm full fine-tuning, development-only evaluation."""
import argparse
import contextlib
import hashlib
import math
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup
from v61_common import FIELDS, MISSING, REVISION, read, save, sha, metrics
from train_v61_neural import Classifier, Inputs, seed_all, optimizer_for, amp, predict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'artifacts/v61_source_factorial_20260914_r2'
AUDIT = ROOT/'artifacts/v63_support_20260920_final'
ASSET = ROOT/'artifacts/v61_asset_cache/securebert2'
OLD = ROOT/'artifacts/v62_capacity_20260915'
SEED = 20260920
ARMS = ['A0', 'A1', 'P0', 'P1']
PORTS = [FIELDS.index('src_port_fixed'), FIELDS.index('dst_port_fixed')]


def port_codes(values):
    result = []
    for value in values:
        if value == MISSING:
            result.append(0)
        elif isinstance(value, str) and re.fullmatch(r'[0-9]+', value) and int(value) <= 65535:
            result.append(int(value) + 1)
        else:
            raise ValueError('Invalid fixed port; no silent UNKNOWN fallback')
    return np.asarray(result, dtype=np.int64)


def nibbles(codes):
    if torch.any((codes < 0) | (codes > 65536)):
        raise ValueError('Port code outside missing or uint16 domain')
    v = (codes - 1).clamp_min(0)
    return torch.stack([(v >> shift) & 15 for shift in [12, 8, 4, 0]], dim=-1)


class PortEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.digits = nn.ModuleList([nn.Embedding(16, 4) for _ in range(4)])
        self.missing = nn.Parameter(torch.randn(16))

    def forward(self, codes):
        digits = nibbles(codes)
        value = torch.cat([emb(digits[..., j]) for j, emb in enumerate(self.digits)], dim=-1)
        return torch.where((codes == 0).unsqueeze(-1), self.missing, value)


def model_for(vocab, arm):
    seed_all(SEED)
    model = Classifier(ASSET, vocab, 'B')
    # All common tensors are initialized before replacing only the port modules.
    if arm.startswith('P'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 631)
            for j in PORTS:
                model.embeddings[j] = PortEmbedding()
    return model


def common_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not any(name.startswith(f'embeddings.{j}.') for j in PORTS):
            h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_development():
    report = read(AUDIT/'support_audit.json')
    for name, digest in report['input_sha256'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in report['output_sha256'].items():
        assert sha(AUDIT/name) == digest, name
    receipt = read(ASSET/'download_receipt.json')
    assert receipt['revision'] == REVISION
    for name, digest in receipt['sha256'].items():
        assert sha(ASSET/name) == digest, name
    frame = pd.read_parquet(DATA/'records.parquet', filters=[('role', 'in', ['fit', 'selection'])])
    frame = frame.sort_values('row_position').reset_index(drop=True)
    assert frame.groupby('group').role.nunique().max() == 1
    assert frame.groupby('body_group').role.nunique().max() == 1
    assert set(frame.role) == {'fit', 'selection'} and frame.row_position.is_unique
    pool = pd.read_parquet(AUDIT/'auxiliary_pool.parquet')
    index = pd.Series(frame.index, index=frame.row_position)
    pool['index'] = pool.row_position.map(index)
    assert pool['index'].notna().all()
    linked = frame.iloc[pool['index']]
    assert linked.role.eq('fit').all()
    for col in ['row_position', 'label', 'group']:
        assert np.array_equal(pool[col], linked[col])
    # B inputs do not consume context; do not load any outer context or labels.
    n = len(frame)
    context = {'stats': np.zeros((n, 16), dtype=np.float32),
               'neighbors': np.full((n, 1), -1, dtype=np.int32),
               'relation': np.zeros((n, 1, 2), dtype=np.float32)}
    return frame, pool, context


def auxiliary_batches(pool, epoch, steps):
    """Eight S exposures per source/bucket, matched M, no selection information."""
    rng = np.random.default_rng(SEED + 10000 + epoch)
    batches = []
    for behavior, bucket in pool.groupby('behavior', sort=True):
        choices = {c: {g: rng.permutation(x['index'].to_numpy()).tolist()
                       for g, x in bucket[bucket.label.eq(c)].groupby('group')} for c in [1, 2]}
        sources_s = rng.permutation(list(choices[2])).tolist()
        sources_m = rng.permutation(list(choices[1])).tolist()
        assert len(sources_s) >= 3 and len(sources_m) >= 3
        sequence_s = sources_s * 8
        used = {1: {}, 2: {}}
        position_m = 0
        count = min(8, len(sources_s), len(sources_m))
        for start in range(0, len(sequence_s), count):
            ss = sequence_s[start:start+count]
            indices = []
            for source in ss:
                k = used[2].get(source, 0); options = choices[2][source]
                indices.append(options[k % len(options)]); used[2][source] = k + 1
            mm = []
            for _ in ss:
                # Distinct M within batch, preferably not the S anchor source.
                for attempt in range(len(sources_m) * 2):
                    source = sources_m[position_m % len(sources_m)]; position_m += 1
                    if source not in mm and source not in ss and used[1].get(source, 0) < 8:
                        break
                else:
                    raise ValueError('Insufficient independent M support for registered auxiliary schedule')
                k = used[1].get(source, 0); options = choices[1][source]
                indices.append(options[k % len(options)]); used[1][source] = k + 1; mm.append(source)
            assert len(ss) == len(set(ss)) and max(used[1].values()) <= 8
            batches.append({'behavior': behavior, 'indices': indices})
    rng.shuffle(batches)
    assert len(batches) <= steps
    slots = np.linspace(0, steps-1, len(batches), dtype=int)
    assert len(set(slots)) == len(batches)
    return {int(slot): batch for slot, batch in zip(slots, batches)}


def metric_summary(frame, prob):
    d = frame[['label', 'group', 'transport_protocol', 'src_role', 'dst_role']].copy()
    d['correct'] = prob.argmax(1) == d.label.to_numpy()
    result = {'ASA': metrics(d.label[d.label.gt(0)], prob[d.label.gt(0)]),
              'M_source_recall': float(d[d.label.eq(1)].groupby('group').correct.mean().mean()),
              'S_source_recall': float(d[d.label.eq(2)].groupby('group').correct.mean().mean()),
              'hard': {}, 'normal_control_errors': int((d.label.eq(0) & ~d.correct).sum())}
    result['source_balanced'] = (result['M_source_recall'] + result['S_source_recall']) / 2
    for protocol in ['tcp', 'udp']:
        sub = d[d.transport_protocol.eq(protocol) & d.src_role.eq('outside') & d.dst_role.eq('dmz')]
        result['hard'][protocol] = {str(c): float(sub[sub.label.eq(c)].groupby('group').correct.mean().mean()) for c in [1, 2]}
    return result


def budget(candidate, reference):
    return (candidate['M_source_recall'] >= reference['M_source_recall']-.01-1e-12 and
            candidate['ASA']['recall_B_M_S'][1] >= reference['ASA']['recall_B_M_S'][1]-.01-1e-12 and
            candidate['ASA']['macro_f1_M_S'] >= reference['ASA']['macro_f1_M_S']-.005-1e-12 and
            all(candidate['hard'][p]['1'] >= reference['hard'][p]['1']-.01-1e-12 for p in ['tcp', 'udp']))


def rank(item):
    m = item['metrics']
    return (m['source_balanced'], m['ASA']['macro_f1_M_S'], -item['step'])


def select_checkpoint(curve, base=None, old=None):
    if base is None:
        selected = max(curve, key=rank)
        return {'checkpoint': selected, 'budget_eligible': True, 'screen_passed': False, 'baseline_only': True}
    eligible = [x for x in curve if budget(x['metrics'], base) and budget(x['metrics'], old)]
    selected = max(eligible or curve, key=rank)
    m = selected['metrics']
    hard_delta = [m['hard'][p]['2'] - base['hard'][p]['2'] for p in ['tcp', 'udp']]
    passed = bool(eligible and m['S_source_recall'] >= base['S_source_recall']+.05-1e-12 and
                  min(hard_delta) >= -1e-12 and max(hard_delta) > 1e-12)
    return {'checkpoint': selected, 'budget_eligible': bool(eligible), 'screen_passed': passed,
            'baseline_only': False, 'fallback_is_diagnostic_only': not bool(eligible)}


def inputs_for(frame, context, device, arm):
    inputs = Inputs(frame, context, ASSET, device)
    assert inputs.length_audit['all_unique_texts_lossless_token_roundtrip']
    assert inputs.length_audit['unknown_tokens'] == 0
    if arm.startswith('P'):
        for j in PORTS:
            inputs.facts[:, j] = torch.tensor(port_codes(frame[FIELDS[j]].tolist()), device=device)
    return inputs


def norm(model):
    return float(torch.stack([p.grad.detach().float().norm()**2 for p in model.parameters() if p.grad is not None]).sum().sqrt())


def gradient_probe(model, inputs, main, aux, device):
    result = {}
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        for name, indices in [('main', main), ('auxiliary', aux)]:
            seed_all(SEED + 123); model.train(); model.zero_grad(set_to_none=True)
            with amp(device):
                loss = nn.functional.cross_entropy(model(**inputs.batch(indices, False)), inputs.y[indices])
            loss.backward(); result[name] = {'CE': float(loss.detach()), 'gradient_norm': norm(model)}
    model.zero_grad(set_to_none=True)
    return result


def smoke(frame, pool, context, out, device):
    result = {'scope': 'Implementation only; not quality validation', 'arms': {}}
    digests = []
    fit = np.flatnonzero(frame.role.eq('fit'))
    for arm in ['A0', 'P0']:
        inputs = inputs_for(frame, context, device, arm)
        model = model_for(inputs.vocab, arm); digests.append(common_digest(model)); model.to(device)
        opt = optimizer_for(model); indices = fit[:32]
        before = next(model.encoder.parameters()).detach().clone()
        timings = []
        for step in range(3):
            model.train(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
            t = time.monotonic()
            with amp(device):
                loss = nn.functional.cross_entropy(model(**inputs.batch(indices, False)), inputs.y[indices])
            loss.backward(); grad = norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step(); torch.cuda.synchronize()
            timings.append(time.monotonic()-t)
        assert not torch.equal(before, next(model.encoder.parameters()).detach()) and np.isfinite(grad) and grad > 0
        expected = predict(model, inputs, indices)
        state = out/f'smoke_{arm}.pt'; torch.save(model.state_dict(), state)
        model.load_state_dict(torch.load(state, weights_only=True, map_location=device))
        np.testing.assert_allclose(predict(model, inputs, indices), expected, rtol=1e-5, atol=1e-6)
        state.unlink()
        result['arms'][arm] = {'encoder_updated': True, 'reload_prediction_matches': True,
                              'gradient_norm': grad, 'microbatch32_seconds': timings,
                              'tokenization': inputs.length_audit}
        del model, opt, inputs, before; torch.cuda.empty_cache()
    assert len(set(digests)) == 1
    result['shared_initial_weights_match'] = True
    result['shared_initial_sha256'] = digests[0]
    result['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
    save(out/'smoke.json', result); print('SMOKE '+str(result), flush=True)


def run_arm(frame, pool, context, schedules, out, device, arm):
    dest = out/arm
    if (dest/'selection.json').exists():
        return
    if dest.exists():
        raise RuntimeError('Partial arm exists; inspect recovery state before restarting')
    dest.mkdir()
    inputs = inputs_for(frame, context, device, arm)
    model = model_for(inputs.vocab, arm)
    assert common_digest(model) == read(out/'smoke.json')['shared_initial_sha256']
    save(dest/'preprocessing.json', {'vocab': inputs.vocab, 'port_mode': 'four_hex_digits' if arm.startswith('P') else 'categorical',
                                    'tokenization': inputs.length_audit, 'parameters': sum(p.numel() for p in model.parameters())})
    model.to(device); opt = optimizer_for(model)
    fit = np.flatnonzero(frame.role.eq('fit')); dev = np.flatnonzero(frame.role.eq('selection'))
    steps = math.ceil(len(fit)/64); total = steps*3
    scheduler = get_linear_schedule_with_warmup(opt, int(total*.1), total)
    old = read(out/'old_reference.json')
    base = None if arm == 'A0' else read(out/'A0/selection.json')['checkpoint']['metrics']
    curve = []; t0 = time.monotonic()
    seed_all(SEED + 50000)  # Same main-task dropout stream, independent of construction.
    checkpoints = {math.ceil(steps * fraction / 4) for fraction in [1, 2, 3, 4]}
    for epoch in range(1, 4):
        order = np.random.default_rng(SEED+epoch).permutation(fit)
        schedule = schedules[epoch-1]
        probe = gradient_probe(model, inputs, fit[:64], schedule[min(schedule)]['indices'], device)
        save(dest/f'gradient_probe_epoch{epoch}.json', probe)
        main_total = aux_total = 0.; seen = auxiliary_seen = 0
        for local, start in enumerate(range(0, len(order), 64)):
            model.train(); opt.zero_grad(set_to_none=True); indices = order[start:start+64]
            for begin in range(0, len(indices), 32):
                ii = indices[begin:begin+32]
                with amp(device):
                    loss = nn.functional.cross_entropy(model(**inputs.batch(ii, False)), inputs.y[ii], reduction='sum')/len(indices)
                loss.backward(); main_total += float(loss.detach())*len(indices); seen += len(ii)
            if arm.endswith('1') and local in schedule:
                ii = schedule[local]['indices']
                with torch.random.fork_rng(devices=[device.index or 0]):
                    torch.manual_seed(SEED+epoch*100000+local); torch.cuda.manual_seed_all(SEED+epoch*100000+local)
                    with amp(device):
                        auxiliary = nn.functional.cross_entropy(model(**inputs.batch(ii, False)), inputs.y[ii])
                    (0.5*auxiliary).backward()
                aux_total += float(auxiliary.detach())*len(ii); auxiliary_seen += len(ii)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(grad), 'Nonfinite gradient'
            opt.step(); scheduler.step()
            global_step = (epoch-1)*steps+local+1
            if (local+1) % 100 == 0:
                print(f'{arm} epoch={epoch} step={local+1}/{steps} main_CE={main_total/seen:.5f} elapsed={time.monotonic()-t0:.1f}', flush=True)
            if local+1 in checkpoints:
                prob = predict(model, inputs, dev)
                summary = metric_summary(frame.iloc[dev], prob)
                name = f'step_{global_step:04d}'
                torch.save(model.state_dict(), dest/(name+'.pt'))
                pred = frame.iloc[dev][['row_position','label','group']].copy()
                pred[['p_B','p_M','p_S']] = prob; pred.to_parquet(dest/(name+'.parquet'), index=False)
                item = {'step': global_step, 'epoch_fraction': (epoch-1)+(local+1)/steps,
                        'metrics': summary, 'main_CE': main_total/seen,
                        'aux_CE': aux_total/auxiliary_seen if auxiliary_seen else None,
                        'aux_rows_seen_this_epoch': auxiliary_seen, 'model_file': name+'.pt',
                        'predictions_file': name+'.parquet', 'elapsed_seconds': time.monotonic()-t0}
                curve.append(item); save(dest/'curve.json', curve)
                print('CHECKPOINT '+__import__('json').dumps({'arm': arm, 'step': global_step, 'macro': summary['ASA']['macro_f1_M_S'],
                    'M_source': summary['M_source_recall'], 'S_source': summary['S_source_recall']}), flush=True)
        # Recoverable epoch boundary, discarded only after the experiment is delivered.
        torch.save({'epoch': epoch, 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                    'model_file': curve[-1]['model_file'], 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all()}, dest/'latest_optimizer.pt')
    selected = select_checkpoint(curve, base, old)
    checkpoint = selected['checkpoint']
    selected.update(arm=arm, status='development_screen_only', quality_acceptance=False,
                    model_sha256=sha(dest/checkpoint['model_file']),
                    predictions_sha256=sha(dest/checkpoint['predictions_file']),
                    elapsed_seconds=time.monotonic()-t0, peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save(dest/'selection.json', selected)
    print('SELECTED '+__import__('json').dumps(selected), flush=True)
    del inputs, model, opt, scheduler; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke-only', action='store_true'); a = ap.parse_args()
    torch.set_num_threads(4); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    device = torch.device('cuda:0')
    frame, pool, context = load_development()
    fit = frame.role.eq('fit'); steps = math.ceil(int(fit.sum())/64)
    schedules = [auxiliary_batches(pool, epoch, steps) for epoch in range(1, 4)]
    params = {'version': 'v63.1-four-arm-1', 'seed': SEED, 'epochs': 3, 'batch': 64, 'microbatch': 32,
              'arms': ARMS, 'encoder_lr': 2e-5, 'other_lr': 1e-4, 'weight_decay': .01,
              'warmup': .1, 'clip': 1., 'auxiliary_coefficient': .5,
              'main_loss': 'all original fit rows uniform CE; no relabeling or removal',
              'fit_rows': int(fit.sum()), 'selection_rows': int((~fit).sum()),
              'data_sha256': sha(DATA/'records.parquet'), 'pool_sha256': sha(AUDIT/'auxiliary_pool.parquet'),
              'plan_sha256': sha(ROOT/'docs/V63_1_TARGETED_TRAINING_PLAN.md'),
              'sources': {name: sha(ROOT/'training'/name) for name in ['train_v63_factorial.py','train_v61_neural.py','v61_common.py','v61_runtime.py']},
              'torch': torch.__version__, 'model_revision': REVISION,
              'old_outer_loaded': False, 'quality_acceptance': False}
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'preregistered.json').exists():
        assert read(a.out/'preregistered.json') == params, 'Run binding changed'
    else:
        save(a.out/'preregistered.json', params)
        save(a.out/'auxiliary_schedule.json', schedules)
    assert read(a.out/'auxiliary_schedule.json') == [{str(k):v for k,v in s.items()} for s in schedules]
    exposures = []
    for epoch, schedule in enumerate(schedules, 1):
        for slot, batch in schedule.items():
            for index in batch['indices']:
                row = frame.iloc[index]
                exposures.append({'epoch': epoch, 'step': slot, 'row_position': int(row.row_position),
                                  'behavior': batch['behavior'], 'group': int(row.group), 'label': int(row.label)})
    exposure = pd.DataFrame(exposures)
    assert exposure.groupby(['epoch','behavior','label','group']).size().max() <= 8
    assert exposure.groupby(['epoch','behavior','label']).size().unstack().apply(lambda x:x[1]==x[2], axis=1).all()
    exposure.to_parquet(a.out/'auxiliary_exposures.parquet', index=False)
    dev = frame[~fit]
    old = pd.read_parquet(OLD/'meanmax/selection.parquet').sort_values('row_position')
    assert np.array_equal(dev.row_position, old.row_position) and np.array_equal(dev.label, old.label)
    save(a.out/'old_reference.json', metric_summary(dev, old[['p_B','p_M','p_S']].to_numpy()))
    if not (a.out/'smoke.json').exists():
        smoke(frame, pool, context, a.out, device)
    if a.smoke_only:
        return
    for arm in ARMS:
        run_arm(frame, pool, context, schedules, a.out, device, arm)
    save(a.out/'training_complete.json', {'status': 'four_arm_development_training_completed',
                                        'quality_acceptance': False,
                                        'arms': {arm: read(a.out/arm/'selection.json') for arm in ARMS}})


if __name__ == '__main__':
    main()
