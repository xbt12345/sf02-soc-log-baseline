"""Post-V128 frozen-output interventions and support audit. Never fits a model.

All uses of outer labels below are retrospective diagnosis on inspected
development data. No result authorizes selection, a new threshold or promotion.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v128_nested_score_trial_20260929_r3'
OUT = ROOT / 'artifacts/v129_mechanism_review_20260930'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
LADDER = ROOT / 'artifacts/v123_targeted_plan_20260929/support_ladder.parquet'
BASE = ROOT / 'artifacts/v127_frozen_branch_trial_20260929'
BODY = ROOT / 'artifacts/v125_order_trial_20260929/ordered_body_bytes.npy'
LENGTH = BODY.with_name('body_lengths.npy')
OFFICIAL = ROOT / 'data/official/train.parquet'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def errors(y, p):
    pred = p.argmax(1) if p.ndim == 2 else p
    return {str(c): {'rows': int((y == c).sum()),
                    'errors': int(((y == c) & (pred != c)).sum())}
            for c in (1, 2)}


def q(values):
    return np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist() if len(values) else []


def conflict(d, name):
    g = d.groupby([name, 'truth']).size().unstack(fill_value=0).reindex(columns=[1, 2], fill_value=0)
    mixed = (g[1] > 0) & (g[2] > 0)
    return {'unique_keys': len(g), 'mixed_keys': int(mixed.sum()),
            'empirical_min_errors': int(g.min(1).sum()),
            'rows_in_mixed_keys': int(g.loc[mixed].sum().sum())}


def main():
    if OUT.exists():
        raise FileExistsError('Preserve previous audit: ' + str(OUT))
    d = pd.read_parquet(TRACE)
    pred = pd.read_parquet(RUN / 'expert_ASA_predictions.parquet')
    ladder = pd.read_parquet(LADDER)
    for field in ('row_position', 'local', 'root', 'fold', 'truth'):
        assert np.array_equal(d[field], pred[field]), field
        assert np.array_equal(d[field], ladder[field]), field
    assert len(d) == 112807 and not d.row_position.duplicated().any()
    assert d.groupby('root').fold.nunique().max() == 1
    official = pd.read_parquet(OFFICIAL, columns=['event_id', 'label_binary', 'src_port', 'message_sanitized'])
    assert len(official) == 2056871
    labels = official.label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    ids = d.row_position.to_numpy()
    assert np.array_equal(labels[ids], d.truth)
    assert np.array_equal(official.message_sanitized.iloc[ids], d.raw_message)
    d['record_src_port'] = official.src_port.iloc[ids].astype(str).to_numpy()
    for name in ('pred_A0', 'pred_AH', 'pred_K_CF', 'pred_P_IS', 'pred_P_CF'):
        d[name] = pred[name]
    facts = d.facts_json.map(json.loads)
    d['destination_key'] = ladder.destination_key
    d['diagnostic_bucket'] = ladder.diagnostic_bucket
    d['parameter_observed'] = ladder.parameter_observed
    body = np.load(BODY, mmap_mode='r')
    lengths = np.load(LENGTH)
    hashes = [hashlib.sha256(bytes(body[i, :int(n)])).hexdigest() for i, n in enumerate(lengths)]
    d['body_key'] = d.local.map(dict(enumerate(hashes)))
    # Typed metadata projection retains missing/opaque state without treating an
    # opaque marker as a recovered numerical port or an attack fact.
    def port_key(v):
        if v.isdecimal() and int(v) <= 65535:
            return 'observed:' + str(int(v))
        return 'unobserved'
    d['body_record_key'] = d.body_key + '|' + d.record_src_port.map(port_key)
    baseline = {name: errors(d.truth.to_numpy(), d[name].to_numpy())
                for name in ('pred_A0', 'pred_AH', 'pred_K_CF', 'pred_P_IS', 'pred_P_CF')}
    assert [baseline['pred_A0'][str(c)]['errors'] for c in (1, 2)] == [318, 2074]
    assert [baseline['pred_P_CF'][str(c)]['errors'] for c in (1, 2)] == [1090, 2174]
    paths = {Path(__file__), TRACE, LADDER, BODY, LENGTH, OFFICIAL,
             RUN/'expert_ASA_predictions.parquet', RUN/'verification.json', RUN/'run_seal.json',
             RUN/'delivery.json', RUN/'score_role_postmortem.json'}
    interventions = []
    d['P_IS_delta_S_minus_M'] = np.nan
    d['P_CF_delta_S_minus_M'] = np.nan
    d['AH_log_probability_S_over_M'] = np.nan
    intervention_ledger = d[['row_position', 'local', 'root', 'fold', 'truth']].copy()
    for fold in range(3):
        mask = d.fold == fold
        held = d.loc[mask]
        ix = held.local.to_numpy(np.int64)
        y = held.truth.to_numpy()
        outer_path = BASE / f'fold{fold}_base_member_logits.npy'
        paths.add(outer_path)
        outer = np.load(outer_path)
        teachers = []
        for j in range(3):
            p = RUN / f'teacher{fold}_{j}/canonical_member_logits.npy'
            paths.update((p, p.with_name('fit.json')))
            teachers.append(np.load(p))
        bank = np.concatenate(teachers, axis=1)
        po = softmax(outer[ix], axis=-1).mean(1)
        d.loc[mask, 'AH_log_probability_S_over_M'] = np.log(np.maximum(po[:, 2], 1e-12) / np.maximum(po[:, 1], 1e-12))
        assert np.max(np.abs(po[:, 2] - pred.loc[mask, 'S_probability_AH'])) < 2e-6
        for base_name, z in [('outer_AH', outer), ('inner_bank_equal', bank)] + [
                (f'inner_teacher_{j}', z) for j, z in enumerate(teachers)]:
            for arm in ('none', 'P_IS', 'P_CF'):
                if arm == 'none':
                    extra = np.zeros((len(body), 3), np.float32)
                else:
                    ep = RUN / f'fold{fold}_{arm}/epoch50_residual.npy'
                    paths.add(ep)
                    extra = np.load(ep)
                    if base_name == 'outer_AH':
                        # Ensure this is the historical endpoint, not a refit.
                        endp = RUN / f'fold{fold}_{arm}/epoch50_prob.npy'
                        paths.add(endp)
                        replay = softmax(outer + extra[:, None, :], axis=-1).mean(1)
                        assert np.max(np.abs(replay - np.load(endp))) < 2e-6
                        assert np.array_equal(replay[ix].argmax(1), held['pred_' + arm])
                        d.loc[mask, arm + '_delta_S_minus_M'] = (extra[ix, 2] - extra[ix, 1])
                p = softmax(z[ix] + extra[ix, None, :], axis=-1).mean(1)
                name = base_name + '_' + arm
                intervention_ledger.loc[mask, name] = p.argmax(1)
                interventions.append({'fold': fold, 'base': base_name, 'residual': arm,
                                      'per_class': errors(y, p)})
        for arm in ('P_IS', 'P_CF'):
            checkpoints = json.loads((RUN/f'fold{fold}_{arm}/checkpoints.json').read_text(encoding='utf-8'))
            paths.add(RUN/f'fold{fold}_{arm}/checkpoints.json')
            mean = np.array(checkpoints[-1]['train_mean_residual'])
            mprob = softmax(outer[ix] + mean[None, None, :], axis=-1).mean(1)
            intervention_ledger.loc[mask, 'outer_AH_' + arm + '_fit_mean'] = mprob.argmax(1)
            interventions.append({'fold': fold, 'base': 'outer_AH', 'residual': arm + '_fit_mean',
                                  'per_class': errors(y, mprob)})
    # Diagnosis of where the gain or damage lies, with original-row denominators.
    cohorts = {
        'all_ASA': np.ones(len(d), bool),
        'CF_new_M_errors': (d.truth == 1) & (d.pred_A0 == 1) & (d.pred_P_CF != 1),
        'CF_repaired_S': (d.truth == 2) & (d.pred_A0 != 2) & (d.pred_P_CF == 2),
        'IS_repaired_S': (d.truth == 2) & (d.pred_A0 != 2) & (d.pred_P_IS == 2),
        'IS_new_S_errors': (d.truth == 2) & (d.pred_A0 == 2) & (d.pred_P_IS != 2),
        'persistent_S': (d.truth == 2) & (d.diagnostic_bucket != 'outside_persistent_S'),
        'root2868_M': (d.root == 2868) & (d.truth == 1),
        'root29_M': (d.root == 29) & (d.truth == 1),
        'root29_S': (d.root == 29) & (d.truth == 2)}
    cohort_results = []
    for name, mask in cohorts.items():
        r = d.loc[mask]
        s = {'cohort': name, 'original_rows': len(r), 'unique_locals': r.local.nunique(),
             'roots': r.root.nunique(), 'body_inputs': r.body_key.nunique(),
             'M_rows': int((r.truth == 1).sum()), 'S_rows': int((r.truth == 2).sum()),
             'AH_log_probability_S_over_M_quantiles': q(r.AH_log_probability_S_over_M),
             'P_IS_delta_quantiles': q(r.P_IS_delta_S_minus_M),
             'P_CF_delta_quantiles': q(r.P_CF_delta_S_minus_M),
             'errors': {a: int((r[a] != r.truth).sum())
                        for a in ('pred_A0', 'pred_AH', 'pred_P_IS', 'pred_P_CF')},
             'destination_keys': r.destination_key.nunique(),
             'parameter_observed_rows': int(r.parameter_observed.sum())}
        support = ladder.loc[mask]
        for level in ('exact', 'destination', 'behavior'):
            same = np.where(r.truth == 1, support[level+'_M_rows'], support[level+'_S_rows'])
            opp = np.where(r.truth == 1, support[level+'_S_rows'], support[level+'_M_rows'])
            sr = np.where(r.truth == 1, support[level+'_M_roots'], support[level+'_S_roots'])
            oroot = np.where(r.truth == 1, support[level+'_S_roots'], support[level+'_M_roots'])
            s[level+'_support'] = {'same_only': int(((same>0)&(opp==0)).sum()),
                                  'opposite_only': int(((same==0)&(opp>0)).sum()),
                                  'both': int(((same>0)&(opp>0)).sum()),
                                  'neither': int(((same==0)&(opp==0)).sum()),
                                  'two_roots_each_class': int(((sr>=2)&(oroot>=2)).sum())}
        cohort_results.append(s)
    # All behavior keys are auditable proxies, not stable semantic label rules.
    group_records = []
    for (fold, key), held in d.groupby(['fold', 'destination_key']):
        fit = d[(d.fold != fold) & (d.destination_key == key)]
        item = {'fold': int(fold), 'destination_key': key,
                'held_M': int((held.truth == 1).sum()), 'held_S': int((held.truth == 2).sum())}
        for c, label in ((1, 'M'), (2, 'S')):
            m = fit.truth == c
            item['fit_'+label] = int(m.sum())
            item['fit_'+label+'_roots'] = int(fit.loc[m, 'root'].nunique())
            for arm in ('A0', 'P_IS', 'P_CF'):
                item['held_'+label+'_errors_'+arm] = int(((held.truth == c) & (held['pred_'+arm] != c)).sum())
        group_records.append(item)
    body_concentration = []
    for (fold, bkey), rows in d.groupby(['fold', 'body_key']):
        if rows.truth.nunique() == 2:
            body_concentration.append({'fold': int(fold), 'body_sha256': bkey, 'rows': len(rows),
                                       'M': int((rows.truth == 1).sum()), 'S': int((rows.truth == 2).sum()),
                                       'record_ports': rows.record_src_port.nunique(),
                                       'roots': rows.root.nunique()})
    # Emit row keys, not raw messages, for review and independent count replay.
    columns = ['row_position','local','root','fold','truth','body_key','body_record_key','facts_json',
               'destination_key','diagnostic_bucket','parameter_observed','pred_A0','pred_AH',
               'pred_P_IS','pred_P_CF','AH_log_probability_S_over_M','P_IS_delta_S_minus_M','P_CF_delta_S_minus_M']
    OUT.mkdir()
    d[columns].to_parquet(OUT/'row_mechanism_ledger.parquet', index=False)
    intervention_ledger.to_parquet(OUT/'frozen_interventions.parquet', index=False)
    pd.DataFrame(group_records).to_csv(OUT/'behavior_support_and_errors.csv', index=False, encoding='utf-8-sig')
    result = {'status':'post_v128_frozen_diagnostic_not_new_training', 'new_classifier_fits':0,
              'optimizer_steps':0, 'official_rows':len(official), 'ASA_rows':len(d),
              'historical_endpoint_replayed':True, 'original_truth_and_messages_verified':True,
              'baseline':baseline, 'interventions':interventions, 'cohorts':cohort_results,
              'projections':{name:conflict(d,name) for name in ('body_key','body_record_key','facts_json','destination_key')},
              'mixed_body_groups_by_fold':body_concentration,
              'source_sha256':sha(__file__),
              'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)},
              'limitations':['All outer folds have been inspected before; no new blind test.',
                             'Frozen teacher-bank substitution changes the base ensemble; it only tests sufficiency of this substitution, not a pure causal isolation of teacher role.',
                             'Behavior keys omit details; mixed proxy labels do not prove erroneous labels or complete-input irreducibility.',
                             'Replacing a residual with its fit mean is a diagnostic intervention, not a selectable candidate.',
                             'No opaque marker is decoded, no label modified, no official validation input used for fitting.']}
    save(OUT/'audit.json',result)
    save(OUT/'output_receipt.json',{'source_sha256':sha(__file__),
          'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'status':result['status'],'baseline':baseline,'cohorts':cohort_results,
                      'projections':result['projections']},ensure_ascii=False))


if __name__ == '__main__':
    main()
