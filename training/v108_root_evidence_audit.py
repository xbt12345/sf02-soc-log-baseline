"""No-fit audit of V107: fit-side support, ranking, concentration, geometry.

Every label-conditioned operation here is retrospective diagnosis, never an
inference feature, deployable threshold, or new blind validation.
"""
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize as unit_norm
from threadpoolctl import threadpool_limits

from run_v75 import ROOT, OUT, save, sha
from v75_views import matrix_hashes, BYTE_FEATURES
from v102_root_review import behavior, metrics, bootstrap

PREV = ROOT / 'artifacts/v107_matched_training_20260928'
DEST = ROOT / 'artifacts/v108_root_evidence_review_20260928'
N1 = ROOT / 'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'
IDS = ROOT / 'artifacts/v92_evidence_training_20260928/ASA_input_ids.npy'
FID = ROOT / 'artifacts/v79_execution_20260927/row_feature_id.npy'


def concentration(frame):
    result = {}
    for c, label in ((1, 'M'), (2, 'S')):
        g = frame[frame.truth == c].groupby('root').size().sort_values(ascending=False)
        result[label] = {'rows': int(g.sum()), 'roots': len(g),
            'largest_root_rows': int(g.iloc[0]), 'top3_rows': int(g.iloc[:3].sum()),
            'largest_fraction': float(g.iloc[0] / g.sum()),
            'inverse_mass_concentration_not_statistical_ESS': float(g.sum()**2 / np.square(g).sum())}
    return result


def auc(y, score):
    return float(roc_auc_score(y == 2, score)) if len(np.unique(y)) == 2 else None


def oracle(y, score, budget, target):
    # Whole score ties move together. Threshold chosen on inspected OOF labels:
    # an optimistic diagnostic, NOT a trainable/deployable calibration result.
    order = np.argsort(-score, kind='stable')
    s = score[order]; yy = y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    m = np.r_[0, np.cumsum(yy == 1)[ends]]
    good = np.r_[0, np.cumsum(yy == 2)[ends]]
    eligible = good[m <= budget]
    reach = m[good >= target]
    return {'diagnostic_only': True, 'M_false_S_budget': int(budget),
            'max_S_correct_with_M_budget': int(eligible.max()),
            'target_S_correct': int(target),
            'min_M_false_S_to_reach_target': int(reach.min()) if len(reach) else None}


def conflict(frame, key):
    ct = frame.groupby([key, 'truth']).size().unstack(fill_value=0)
    bad = (ct > 0).sum(axis=1) > 1
    return {'groups': len(ct), 'mixed_label_groups': int(bad.sum()),
            'mixed_group_rows': int(ct.loc[bad].sum().sum()),
            'empirical_deterministic_min_errors_at_this_resolution': int((ct.sum(axis=1)-ct.max(axis=1)).sum()),
            'scope': 'Observed labels and this representation only; not a Bayes error or proof of real-world ambiguity.'}


def main():
    started = time.monotonic()
    assert not DEST.exists(), DEST
    paths = [PREV/'registration.json', PREV/'evaluation.json', PREV/'adversarial_decision.json',
             PREV/'OOF_ASA_ledger.parquet', PREV/'N2_ASA.npz', PREV/'N2_text_pairs.parquet',
             N1, IDS, FID, OUT/'rows.parquet', OUT/'projections.parquet']
    receipts = {str(p.relative_to(ROOT)).replace('\\','/'): sha(p) for p in paths}
    reg = json.loads(paths[0].read_text(encoding='utf-8'))
    assert reg['N2_ASA_sha256'] == sha(PREV/'N2_ASA.npz')
    assert reg['N2_pairs_sha256'] == sha(PREV/'N2_text_pairs.parquet')
    for p in (N1, IDS, FID, OUT/'rows.parquet'):
        assert reg['input_sha256'][str(p.relative_to(ROOT)).replace('\\','/')] == sha(p)
    d = pd.read_parquet(PREV/'OOF_ASA_ledger.parquet')
    ids = np.load(IDS); fid = np.load(FID, mmap_mode='r')
    lookup = np.full(int(fid.max())+1, -1, np.int32); lookup[ids] = np.arange(len(ids))
    d['local'] = lookup[fid[d.row_position.to_numpy()]]
    assert (d.local >= 0).all() and len(d) == 112807
    assert d.groupby('local').fold.nunique().max() == 1
    views = {'N1': sparse.load_npz(N1), 'N2': sparse.load_npz(PREV/'N2_ASA.npz')}
    assert (views['N1'][:, BYTE_FEATURES:] != views['N2'][:, BYTE_FEATURES:]).nnz == 0
    d['changed'] = np.asarray((views['N1'] != views['N2']).getnnz(axis=1) > 0)[d.local]
    rows = pd.read_parquet(OUT/'rows.parquet', columns=['row_position', 'projection_id', 'label_index']).set_index('row_position')
    assert np.array_equal(rows.loc[d.row_position, 'label_index'], d.truth)
    facts = pd.read_parquet(OUT/'projections.parquet', columns=['facts'])
    d['projection_id'] = rows.loc[d.row_position, 'projection_id'].to_numpy()
    fm = {int(p): json.loads(facts.facts.iat[int(p)]) for p in d.projection_id.unique()}
    d['behavior'] = [behavior(fm[int(p)]) for p in d.projection_id]
    def coarse(f):
        # A redacted port makes the exact key incomplete, not its known roles.
        keys = ('action','outcome','transport_protocol','src_role','dst_role')
        if any(k not in f for k in keys): return None
        if any(f[k] not in ('inside','outside','dmz') for k in ('src_role','dst_role')): return None
        if f['transport_protocol'] not in ('tcp','udp','icmp'): return None
        return json.dumps({k:f[k] for k in keys},sort_keys=True)
    d['coarse'] = [coarse(fm[int(p)]) for p in d.projection_id]
    support_cols = []
    for key in ('behavior', 'coarse'):
        for what in ('roots', 'rows'):
            for side in ('same', 'other'):
                col = f'{key}_{side}_{what}'; d[col] = -1; support_cols.append(col)
        for fold in range(3):
            tr = d[(d.fold != fold) & d[key].notna()]
            group = tr.groupby([key, 'truth'])
            for what, counts in (('roots', group.root.nunique().to_dict()), ('rows', group.size().to_dict())):
                mask = d.fold == fold
                for side in ('same', 'other'):
                    d.loc[mask, f'{key}_{side}_{what}'] = [counts.get((b, int(y) if side == 'same' else 3-int(y)), 0) if b else -1
                                                          for b,y in zip(d.loc[mask,key],d.loc[mask,'truth'])]
    y = d.truth.to_numpy(); a = d.N1_TabM25.to_numpy(); b = d.N2_TabM25.to_numpy()
    regressed = (y == 2) & (a == y) & (b != y)
    slices = {'all_S': y == 2, 'N1_S_errors': (y == 2)&(a != y),
              'N2_S_errors': (y == 2)&(b != y), 'S_regressed': regressed,
              'changed_S': d.changed.to_numpy() & (y == 2)}
    slices['ICMP_3_13_S'] = np.array([fm[int(p)].get('transport_protocol') == 'icmp' and fm[int(p)].get('icmp_type') == 3 and fm[int(p)].get('icmp_code') == 13 for p in d.projection_id]) & (y == 2)
    for protocol, port in (('udp',514),('tcp',6514)):
        slices[f'{protocol}_{port}_S'] = np.array([fm[int(p)].get('transport_protocol') == protocol and fm[int(p)].get('dst_port_fixed') == port for p in d.projection_id]) & (y == 2)
    supports = {}
    for name, mask in slices.items():
        t = d.loc[mask]
        s = {'rows': len(t), 'roots': int(t.root.nunique()), 'N1_correct': int((a[mask] == y[mask]).sum()), 'N2_correct': int((b[mask] == y[mask]).sum())}
        for key in ('behavior','coarse'):
            same = t[f'{key}_same_roots']; other = t[f'{key}_other_roots']
            s[key] = {'incomplete': int((same < 0).sum()), 'zero_same_class_support': int((same == 0).sum()),
                'only_other_class_support': int(((same == 0)&(other > 0)).sum()),
                'one_same_class_root': int((same == 1).sum()), 'at_least_two_same_class_roots': int((same >= 2).sum()),
                'both_classes_supported': int(((same > 0)&(other > 0)).sum())}
        supports[name] = s
    fit_results = []; prob_receipts = {}
    for view in ('N1','N2'):
        for fold in range(3):
            folder = PREV/f'fold{fold}_{view}_TabM25_seed10201'
            fit = json.loads((folder/'fit.json').read_text())
            assert fit['prob_sha256'] == sha(folder/'ASA_input_prob.npy')
            assert fit['model_sha256'] == sha(folder/'model.pt')
            prob_receipts[folder.name] = {'prob_sha256':fit['prob_sha256'], 'model_sha256':fit['model_sha256']}
            p = np.load(folder/'ASA_input_prob.npy')[d.local]
            hold = d.fold.to_numpy() == fold
            assert np.array_equal(p[hold].argmax(axis=1), d.loc[hold,f'{view}_TabM25'])
            assert np.array_equal(p[hold,2], d.loc[hold,f'{view}_S_probability'])
            assert int((p[~hold].argmax(axis=1) != y[~hold]).sum()) == fit['train_original_errors']
            for scope,mask in (('train',~hold),('heldout',hold)):
                mm = metrics(y[mask],p[mask].argmax(axis=1))
                mm.update(view=view,fold=fold,scope=scope,S_auc=auc(y[mask],p[mask,2]))
                mm['concentration'] = concentration(d[mask])
                fit_results.append(mm)
    ranking = {}
    for view in ('N1','N2'):
        score = d[f'{view}_S_probability'].to_numpy()
        ranking[view] = {'OOF_S_auc': auc(y,score), **oracle(y,score,int(((y==1)&(a!=1)).sum()),int(((y==2)&(a==2)).sum()))}
    # Test an even more optimistic oracle: arbitrary threshold per existing fold.
    # Budgets allocated to each fold equal its N1 M mistakes, sum = global budget.
    ranking['N2_fold_oracles_at_N1_fold_M_budgets'] = []
    for fold in range(3):
        m = d.fold.to_numpy() == fold
        ranking['N2_fold_oracles_at_N1_fold_M_budgets'].append({'fold':fold,**oracle(y[m],d.loc[m,'N2_S_probability'].to_numpy(),int(((y[m]==1)&(a[m]!=1)).sum()),int(((y[m]==2)&(a[m]==2)).sum()))})
    floor = {}
    for view in ('N1','N2'):
        hashes = np.array(matrix_hashes(views[view]), dtype=object)
        d[f'{view}_hash'] = hashes[d.local]
        assert d.groupby(f'{view}_hash').fold.nunique().max() == 1
        floor[view+'_full_input'] = conflict(d,view+'_hash')
    floor['complete_behavior_key'] = conflict(d[d.behavior.notna()], 'behavior')
    # Counterfactual nearest-neighbor geometry: label known only for diagnostics.
    neighbor_parts = []
    for fold in range(3):
        queries = d[(d.fold==fold)&regressed].groupby('local').size()
        if not len(queries): continue
        for view in ('N1','N2'):
            xx = unit_norm(views[view], norm='l2', axis=1)
            query_ids = queries.index.to_numpy()
            output = {'fold':np.full(len(queries),fold), 'view':np.full(len(queries),view), 'local':query_ids, 'S_rows':queries.to_numpy()}
            for c,label in ((1,'M'),(2,'S')):
                train_ids = np.unique(d.loc[(d.fold!=fold)&(d.truth==c),'local'])
                assert not np.intersect1d(query_ids,train_ids).size
                other = xx[train_ids].T.tocsr(); best=[]; near=[]
                for start in range(0,len(queries),32):
                    sim = (xx[query_ids[start:start+32]]@other).toarray()
                    ix=sim.argmax(axis=1); best.extend(sim[np.arange(len(sim)),ix]);near.extend(train_ids[ix])
                output[f'max_cosine_{label}'] = best; output[f'nearest_{label}_local'] = near
            neighbor_parts.append(pd.DataFrame(output))
    neighbors = pd.concat(neighbor_parts,ignore_index=True)
    geometry = {}
    for view,g in neighbors.groupby('view'):
        geometry[view] = {'unique_queries':len(g), 'regressed_S_rows':int(g.S_rows.sum()),
                         'rows_M_neighbor_strictly_closer':int(g.loc[g.max_cosine_M > g.max_cosine_S,'S_rows'].sum()),
                         'rows_S_neighbor_strictly_closer':int(g.loc[g.max_cosine_M < g.max_cosine_S,'S_rows'].sum()),
                         'rows_tied':int(g.loc[g.max_cosine_M == g.max_cosine_S,'S_rows'].sum()),
                         'note':'Exact cosine in model INPUT representation, not hidden representation or proof of semantic similarity.'}
    result={'status':'no_fit_root_evidence_audit_completed','classifier_fits':0,'calibration_fits':0,
        'scope':'Retrospective official-training-only diagnosis; all OOF labels inspected; no external or private answers.',
        'source_sha256':sha(__file__),'input_sha256':receipts,'model_prediction_receipts':prob_receipts,
        'support_slices':supports,'training_and_holdout':fit_results,'overall_concentration':concentration(d),
        'ranking_diagnostic':ranking,'representation_conflict_floor':floor,'input_neighborhood':geometry,
        'paired_group_bootstrap':bootstrap(y,a,b,d.root.to_numpy(),d.fold.to_numpy()),
        'limits':['Behavior keys are descriptive proxies, not sufficient M/S truth; zero exact support does not rule out transfer.',
                  'Observable source roots are not certified physical actors.',
                  'Threshold oracle uses heldout labels and must never be shipped or treated as unbiased performance.',
                  'Group bootstrap is conditional on existing fits and studied folds; not unseen-domain uncertainty.']}
    DEST.mkdir()
    d.to_parquet(DEST/'support_and_error_ledger.parquet',index=False)
    neighbors.to_parquet(DEST/'regressed_S_nearest_input_neighbors.parquet',index=False)
    result['elapsed_seconds']=time.monotonic()-started
    save(DEST/'root_evidence.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4): main()
