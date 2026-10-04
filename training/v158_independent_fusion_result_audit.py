"""Independent V158 saved-output, gold and cost audit; never runs a classifier.

This consumes immutable saved arrays and tensors only. It does not instantiate
models, execute official feature/classifier functions, fit, or compute gradients.
"""
from pathlib import Path
import hashlib
import json
import traceback
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v158_fusion_trial_20261001'
BANK = ROOT / 'artifacts/v158_legal_fusion_bank_v2_20261001'
OUT = ROOT / 'artifacts/v158_independent_fusion_result_audit_20261001'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def bind(d):
    for name, identity in d.items():
        p = Path(name)
        if not p.is_absolute():
            p = ROOT / p
        assert p.is_file() and sha(p) == identity, name


def events(p):
    return [json.loads(z) for z in p.read_text(encoding='utf-8').splitlines() if z.strip()]


def tensor_id(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def classes(y, p):
    result = {}
    for c in range(3):
        n, called, tp = int((y == c).sum()), int((p == c).sum()), int(((y == c) & (p == c)).sum())
        result[str(c)] = dict(support=n, correct=tp, missed=n-tp, false_called=called-tp,
            recall=tp/n if n else None, precision=tp/called if called else None,
            f1=2*tp/(n+called) if n else None)
    return result


def changes(y, before, after):
    return {str(c): dict(support=int((y == c).sum()),
        before_correct=int(((y == c) & (before == y)).sum()),
        after_correct=int(((y == c) & (after == y)).sum()),
        before_errors=int(((y == c) & (before != y)).sum()),
        after_errors=int(((y == c) & (after != y)).sum()),
        repairs=int(((y == c) & (before != y) & (after == y)).sum()),
        new_errors=int(((y == c) & (before == y) & (after != y)).sum())) for c in range(3)}


def risk_rows(frame, q):
    y = frame.truth.to_numpy(np.int64)
    ce = -np.log(np.maximum(q[np.arange(len(y)), y], np.finfo(np.float64).tiny))
    brier = ((q - np.eye(3)[y])**2).sum(1)
    return dict(mean_CE=float(ce.mean()), classes={str(c): dict(support=int((y == c).sum()),
        errors=int(((y == c) & (q.argmax(1) != y)).sum()),
        mean_CE=float(ce[y == c].mean()), mean_Brier=float(brier[y == c].mean()))
        for c in range(3) if (y == c).any()})


def check_ledger(path, frame, q):
    d = pd.read_parquet(path)
    cols = ['row_position', 'local', 'root', 'fold', 'truth', 'canonical_key']
    assert d[cols].equals(frame[cols].reset_index(drop=True)), str(path)
    qq = d[['p0', 'p1', 'p2']].to_numpy()
    assert np.isfinite(qq).all() and (qq >= 0).all()
    assert np.allclose(qq.sum(1), 1, atol=1e-12, rtol=0)
    assert np.array_equal(d.pred, qq.argmax(1))
    if q is not None:
        assert np.allclose(qq, q[d.local], atol=2e-12, rtol=0), str(path)
    return d, qq


def main():
    assert not OUT.exists(), 'Preserve every audit snapshot'
    assert (RUN/'final_delivery.json').is_file()
    seal = read(RUN/'run_seal.json')
    bind(seal['source_sha256'])
    bind(read(RUN/'evaluation_input_bindings.json')['source_sha256'])
    assert seal['plan_sha256'] == sha(ROOT/'training/review_policy/v158_complete_nested_trial_v2_plan.json')
    assert seal['contract_sha256'] == sha(ROOT/'training/review_policy/v158_fusion_execution_contract.json')
    delivery, quality = read(RUN/'final_delivery.json'), read(RUN/'quality.json')
    assert delivery['seal_sha256'] == sha(RUN/'run_seal.json')
    assert delivery['quality_sha256'] == sha(RUN/'quality.json')
    official = pd.read_parquet(ROOT/'data/official/train.parquet', columns=['event_id', 'label_binary'])
    rows = pd.read_parquet(ROOT/'artifacts/v75_four_arm_20260921_r2/rows.parquet',
                           columns=['row_position', 'event_id', 'label_index', 'route'])
    folds = pd.read_parquet(ROOT/'artifacts/v106_frozen_audit_20260928/proposed_body_closed_folds.parquet',
                            columns=['row_position', 'root', 'proposed_fold'])
    y = official.label_binary.map({'benign':0, 'malicious':1, 'suspicious':2}).to_numpy(np.int8)
    assert len(rows) == len(folds) == len(y) == 2056871
    assert np.array_equal(rows.row_position, np.arange(len(y)))
    assert np.array_equal(folds.row_position, rows.row_position)
    assert np.array_equal(official.event_id, rows.event_id) and np.array_equal(y, rows.label_index)
    full = pd.read_parquet(RUN/'full_prediction_ledger.parquet')
    for name, value in [('row_position', rows.row_position), ('truth', y), ('root', folds.root),
                        ('fold', folds.proposed_fold), ('route', rows.route)]:
        assert np.array_equal(full[name], value), name
    asa = pd.read_parquet(RUN/'ASA_prediction_ledger.parquet')
    pos = np.flatnonzero(rows.route.eq('asa'))
    assert len(asa) == 112807 and np.array_equal(asa.row_position, pos)
    trace = pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet')
    for c in ['row_position', 'local', 'root', 'fold', 'truth']:
        assert np.array_equal(asa[c], trace[c]), c
    assert np.array_equal(asa.truth, y[pos])
    old = pd.read_parquet(ROOT/'artifacts/v146_guarded_pair_training_20261001/full_prediction_ledger.parquet')
    assert np.array_equal(full.pred_A0, old.pred_A0) and np.array_equal(full.pred_V146_A, old.pred_A)
    ref_train = pd.read_parquet(ROOT/'artifacts/v130_learning_review_20260930_r2/training_role_error_ledger.parquet')
    keys = ref_train[['local', 'canonical_key']].drop_duplicates()
    assert keys.local.is_unique
    keymap = keys.set_index('local').canonical_key
    assert np.array_equal(asa.canonical_key, asa.local.map(keymap))
    budgets = read(ROOT/'training/review_policy/v158_fusion_execution_contract.json')['role_call_budgets']
    stages, learning_history = [], []
    zero = np.empty((len(asa), 3))
    probability = {a:np.empty((len(asa), 3)) for a in 'AB'}
    bank_blocked = np.zeros(len(asa), bool)
    bank_has_correct = np.zeros(len(asa), bool)
    bank_margin = np.empty(len(asa))
    train_parts = {a:[] for a in 'AB'}
    for f in range(3):
        frame = asa[asa.fold.ne(f)][['row_position','local','root','fold','truth','canonical_key']].copy().reset_index(drop=True)
        fr = pd.read_parquet(BANK/f'fold{f}/legal_FIT_reference.parquet')
        assert frame.equals(fr)
        counts = frame.groupby(['canonical_key','truth']).size().unstack(fill_value=0)
        floor = int((counts.sum(1)-counts.max(1)).sum())
        assert floor == [22,6,28][f]
        pure_keys = set(counts.index[(counts > 0).sum(1).eq(1)])
        mass = np.bincount(frame.truth, minlength=3).tolist()
        mask = asa.fold.eq(f).to_numpy()
        deployed = np.load(BANK/f'fold{f}/deployment_probabilities.npy')
        assert deployed.shape == (22546,17,3)
        current = deployed[:,:16].mean(1)
        initial_correct = current[frame.local].argmax(1) == frame.truth.to_numpy()
        p0a = np.load(RUN/f'preflight{f}_A/deployment_probability.npy')
        p0b = np.load(RUN/f'preflight{f}_B/deployment_probability.npy')
        assert np.array_equal(p0a,p0b)
        zero[mask] = p0a[asa.loc[mask,'local']]
        b = deployed[asa.loc[mask,'local']]
        yy = asa.loc[mask,'truth'].to_numpy()
        bank_has_correct[mask] = (b.argmax(2) == yy[:,None]).any(1)
        other = np.where(yy == 1,2,1)
        truep = np.take_along_axis(b, np.broadcast_to(yy[:,None,None],(len(yy),17,1)),axis=2)[...,0]
        otherp = np.take_along_axis(b, np.broadcast_to(other[:,None,None],(len(yy),17,1)),axis=2)[...,0]
        bank_margin[mask] = (truep-otherp).max(1)
        bank_blocked[mask] = bank_margin[mask] < 0
        for arm in 'AB':
            folder = RUN/f'fold{f}_{arm}'
            r = read(folder/'fit.json')
            assert (r['fold'],r['arm'],r['termination']) == (f,arm,'accepted_update_budget')
            assert not r['selected_by_score'] and r['seal_sha256'] == sha(RUN/'run_seal.json')
            assert all(r[k] == 200 for k in ['full_gradients','proposal_evaluations','accepted_updates'])
            for name,k in [('endpoint.pt','model_sha256'),('endpoint_deployment_probability.npy','deployment_probability_sha256'),
                            ('endpoint_OOF_probability.npy','OOF_probability_sha256')]:
                assert sha(folder/name) == r[k]
            initial = torch.load(RUN/f'initial_{arm}.pt',map_location='cpu',weights_only=True)
            end = torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)
            assert tensor_id(initial['state']) == r['initial_parameter_sha256']
            assert tensor_id(end['state']) == r['endpoint_parameter_sha256']
            assert (end['fold'],end['arm'],end['seal_sha256']) == (f,arm,r['seal_sha256'])
            grad = events(folder/'gradients.jsonl')
            attempts = [g for g in grad if g['event'] == 'attempt']
            completed = [g for g in grad if g['event'] == 'completed']
            proposals = events(folder/'proposals.jsonl')
            history = read(folder/'progress.json')
            assert len(attempts) == len(completed) == len(proposals) == len(history) == 200
            assert all(g['original_class_mass'] == mass for g in completed)
            assert all(v['classification_guard'] and v['accepted'] and v['trial_CE'] <= v['Armijo_bound'] for v in proposals)
            assert [p['parameter_sha256'] for p in proposals] == [h['parameter_sha256'] for h in history]
            calls = events(folder/'classifier_calls.jsonl')
            ac = sum(x['event']=='attempt' for x in calls)
            assert ac == sum(x['event']=='completed' for x in calls) == r['classifier_forward_calls']
            assert ac <= budgets[f]['fit_classifier_forward_cap_per_arm']
            ec = events(folder/'evaluation_classifier_calls.jsonl')
            ecn = sum(x['event']=='attempt' for x in ec)
            assert ecn == sum(x['event']=='completed' for x in ec) <= budgets[f]['evaluation_classifier_forward_cap_per_arm']
            assert not any(x.get('grad_enabled',False) for x in ec)
            q = np.load(folder/'endpoint_deployment_probability.npy')
            qo = np.load(folder/'endpoint_OOF_probability.npy')
            ids = np.sort(frame.local.unique())
            assert np.isfinite(q).all() and np.isfinite(qo[ids]).all()
            assert np.allclose(q.sum(1),1,rtol=0,atol=1e-12)
            fit,qq = check_ledger(folder/'endpoint_original_FIT_rows.parquet',frame,q)
            oof,oq = check_ledger(folder/'endpoint_original_OOF_rows.parquet',frame,qo)
            assert np.all(fit.pred.to_numpy()[initial_correct] == frame.truth.to_numpy()[initial_correct])
            wrong = fit.pred.ne(fit.truth)
            assert int((wrong & fit.truth.eq(1)).sum()) == 0
            assert int((wrong & fit.truth.eq(2)).sum()) == floor
            assert not (wrong & fit.canonical_key.isin(pure_keys)).any()
            assert int(initial_correct.sum()) == r['protected_current_correct_original_FIT_rows']
            oofrisk = risk_rows(oof,oq)
            assert abs(oofrisk['mean_CE']-r['final_OOF_CE']) <= 1e-12
            initqo = np.load(RUN/f'preflight{f}_{arm}/OOF_probability.npy')
            initrisk = risk_rows(frame,initqo[frame.local])
            assert abs(initrisk['mean_CE']-r['initial_OOF_CE']) <= 1e-12
            windows = []
            for v in r['last5']:
                state = torch.load(folder/f"accepted{v['update']}.pt",map_location='cpu',weights_only=True)
                assert tensor_id(state['state']) == v['parameter_sha256']
                wl,wq = check_ledger(folder/f"accepted{v['update']}_deployment_FIT_rows.parquet",frame,None)
                assert np.all(wl.pred.to_numpy()[initial_correct] == frame.truth.to_numpy()[initial_correct])
                wb = wl.pred.ne(wl.truth)
                assert int((wb & wl.truth.eq(1)).sum()) == 0 and int((wb & wl.truth.eq(2)).sum()) == floor
                windows.append(v['parameter_sha256'])
            assert len(windows) == len(set(windows)) == 5
            train_parts[arm].append(fit)
            probability[arm][mask] = q[asa.loc[mask,'local']]
            stage = dict(fold=f,arm=arm,gradients=200,proposals=200,updates=200,
                classifier_calls=ac,evaluation_classifier_calls=ecn,initial_correct_role_rows=int(initial_correct.sum()),
                initial_OOF=initrisk,endpoint_OOF=oofrisk,last_gradient_norm=completed[-1]['gradient_norm'],
                last_trial_CE_decrease=proposals[-1]['base_CE']-proposals[-1]['trial_CE'],
                terminated_by_budget_not_convergence=True, fit_sha256=sha(folder/'fit.json'))
            stages.append(stage)
            # Recount real saved OOF snapshots; do not create a new classifier call.
            for update in [1,20,50,100,150,196,197,198,199,200]:
                lp = folder/f'accepted{update}_OOF_rows.parquet'
                dd,qqq = check_ledger(lp,frame,None)
                learning_history.append(dict(fold=f,arm=arm,update=update,**risk_rows(dd,qqq)))
    combined = {}
    not_asa = rows.route.ne('asa').to_numpy()
    for arm in 'AB':
        p = full['pred_'+arm].to_numpy()
        assert np.isin(p,[0,1,2]).all()
        assert np.array_equal(p[pos],probability[arm].argmax(1))
        assert np.array_equal(p[pos],asa['pred_'+arm])
        assert np.array_equal(asa[[f'{arm}_p{c}' for c in range(3)]],probability[arm])
        assert np.array_equal(p[not_asa],old.pred_A.to_numpy()[not_asa])
        assert int((p[not_asa] != y[not_asa]).sum()) == 107
        tt = pd.concat(train_parts[arm],ignore_index=True)
        assert tt.equals(pd.read_parquet(RUN/f'{arm}_all_training_role_ledger.parquet'))
        combined[arm] = dict(full=classes(y,p),ASA=classes(y[pos],p[pos]),TRAIN=classes(tt.truth.to_numpy(),tt.pred.to_numpy()))
        assert combined[arm]['full'] == quality[arm]['full_task']['B']
        assert combined[arm]['ASA'] == quality[arm]['ASA']['B']
    yy = asa.truth.to_numpy()
    asa['pred_zero'] = zero.argmax(1)
    paired = {name:changes(yy,asa[before].to_numpy(),asa[after].to_numpy()) for name,before,after in [
        ('zero_vs_V146_A','pred_V146_A','pred_zero'),('B_vs_zero','pred_zero','pred_B'),
        ('A_vs_zero','pred_zero','pred_A'),('B_vs_A','pred_A','pred_B'),
        ('B_vs_A0','pred_A0','pred_B'),('B_vs_V146_A','pred_V146_A','pred_B')]}
    for name in ['B_vs_A','B_vs_A0','B_vs_V146_A']:
        assert paired[name] == delivery['original_pairs'][name]
    fixed = pd.read_parquet(ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet')
    assert np.array_equal(fixed.row_position,asa.row_position)
    masks = dict(all_ASA=np.ones(len(asa),bool),hard578=fixed.known_578_cohort.to_numpy(),
        strict51=fixed.same_family_and_outer_fold_control_S.to_numpy(),
        unknown_or_missing_ports=~fixed.both_ports_observed.to_numpy(),
        ICMP=fixed.facts_json.map(lambda z:str(json.loads(z).get('transport_protocol','')).upper()=='ICMP').to_numpy(),
        all17_strict_S_less_M=bank_blocked & (yy==2) & fixed.known_578_cohort.to_numpy())
    assert sum(masks['hard578']) == 578 and sum(masks['strict51']) == 51 and sum(masks['all17_strict_S_less_M']) == 188
    cohort_results = {}
    for name,mask in masks.items():
        cohort_results[name] = {k:classes(yy[mask],asa.loc[mask,k].to_numpy()) for k in ['pred_A0','pred_V146_A','pred_zero','pred_A','pred_B']}
    fold_results = [dict(fold=f,A_errors=int(asa.loc[asa.fold.eq(f),'pred_A'].ne(asa.loc[asa.fold.eq(f),'truth']).sum()),
        B_errors=int(asa.loc[asa.fold.eq(f),'pred_B'].ne(asa.loc[asa.fold.eq(f),'truth']).sum())) for f in range(3)]
    assert fold_results == quality['matched']['folds']
    source = asa.assign(A_wrong=asa.pred_A.ne(asa.truth), B_wrong=asa.pred_B.ne(asa.truth)).groupby(['root','truth']).agg(
        support=('truth','size'), A_errors=('A_wrong','sum'), B_errors=('B_wrong','sum')).reset_index()
    assert source.equals(pd.read_parquet(RUN/'all_source_class_quality.parquet'))
    source_all = []
    for root,part in asa.groupby('root'):
        yp = part.truth.to_numpy()
        for cl in [1,2]:
            mm = yp==cl
            if not mm.any():continue
            rec = dict(root=int(root),fold=int(part.fold.iloc[0]),truth=cl,support=int(mm.sum()))
            for k in ['pred_A0','pred_V146_A','pred_zero','pred_A','pred_B']:
                rec[k+'_errors'] = int((part.loc[mm,k]!=cl).sum())
            source_all.append(rec)
    source_all = pd.DataFrame(source_all)
    sc = {}
    for k in ['pred_A0','pred_V146_A','pred_zero','pred_A','pred_B']:
        sr = source_all[source_all.truth.eq(2)]
        outside = (yy==2)&~asa.root.isin([21702,20849,29]).to_numpy()
        sc[k]=dict(S_sources=len(sr),S_mean_source_recall=float((1-sr[k+'_errors']/sr.support).mean()),
            S_zero_recall_roots=int(sr[k+'_errors'].eq(sr.support).sum()),
            S_outside_top3_errors=int((asa.loc[outside,k]!=2).sum()))
    capacity = {str(c):dict(original_rows=int((yy==c).sum()),
        all17_strict_wrong_margin_rows=int((bank_blocked&(yy==c)).sum()),
        errors_with_some_correct_expert=int(((yy==c)&asa.pred_B.ne(asa.truth).to_numpy()&bank_has_correct).sum()),
        errors_without_correct_expert=int(((yy==c)&asa.pred_B.ne(asa.truth).to_numpy()&~bank_has_correct).sum())) for c in [1,2]}
    assert sum(s['gradients'] for s in stages)==delivery['fusion_full_gradients']==1200
    assert sum(s['proposals'] for s in stages)==delivery['fusion_proposals']==1200
    assert sum(s['updates'] for s in stages)==delivery['fusion_updates']==1200
    assert sum(s['evaluation_classifier_calls'] for s in stages)==delivery['actual_evaluation_classifier_forward_calls']==480
    assert not delivery['quality_acceptance'] and not delivery['matched_effect_passed'] and not delivery['model_promoted']
    result = dict(status='V158_actual_full_gold_saved_probabilities_roles_costs_independently_recounted',
        official_rows=len(y),ASA_rows=len(asa),physical_fusion_sealed_files=len(seal['source_sha256']),
        own_model_forwards=0,own_feature_calls=0,own_gradients=0,own_fits=0,own_updates=0,
        metrics=combined,zero_ASA=classes(yy,asa.pred_zero.to_numpy()),paired_changes=paired,
        folds=fold_results,cohorts=cohort_results,source_summary=sc,convex_capacity_diagnostic=capacity,
        fusion_stages=stages,matched_gates=quality['matched_gates'],quality_acceptance=False,model_promoted=False,
        limits=['Saved arithmetic is not a new actual scorer replay; executor verification records that replay.',
            'OOF fitting quality, deployment FIT protection, and outer development quality are separate populations.',
            'Convex negative-margin bound is a fixed model-family limit, not raw-data impossibility.',
            'All six fits stopped at registered budget with nonzero terminal gradients; numerical convergence is not claimed.',
            'Inspected source folds are development evidence, not a blind external environment test.'],
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__),RUN/'final_delivery.json',
            RUN/'quality.json',RUN/'verification.json',RUN/'run_seal.json',ROOT/'data/official/train.parquet']})
    OUT.mkdir()
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'OOF_saved_checkpoint_learning.json').write_text(json.dumps(learning_history,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    source_all.to_parquet(OUT/'all_source_initial_learning_and_final_errors.parquet',index=False)
    asa['some_expert_correct'] = bank_has_correct
    asa['strict_convex_wrong_margin'] = bank_blocked
    asa['best_true_vs_other_expert_margin'] = bank_margin
    asa.to_parquet(OUT/'all_original_ASA_initial_endpoint_and_capacity.parquet',index=False)
    print(json.dumps({k:result[k] for k in ['status','official_rows','ASA_rows','zero_ASA','paired_changes','convex_capacity_diagnostic','source_summary']},ensure_ascii=False))


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        OUT.mkdir(exist_ok=True)
        (OUT/'failure.json').write_text(json.dumps(dict(error_type=type(e).__name__,error=str(e),
            traceback=traceback.format_exc(),own_model_forwards=0,own_gradients=0,own_fits=0),ensure_ascii=False,indent=2),encoding='utf-8')
        raise
