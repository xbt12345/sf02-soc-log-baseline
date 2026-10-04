"""Read saved v101 predictions; no fitting, threshold tuning or outer-label selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/v101_full_input_group_n1_20260928'
DEST = ROOT / 'artifacts/v102_root_review_20260928'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def behavior(f):
    keys = ['action', 'outcome', 'transport_protocol', 'src_role', 'dst_role']
    if any(k not in f for k in keys):
        return None
    if any(f[k] not in ('inside', 'outside', 'dmz') for k in ('src_role', 'dst_role')):
        return None
    if f['transport_protocol'] == 'icmp':
        extra, limit = ['icmp_type', 'icmp_code'], 255
    elif f['transport_protocol'] in ('tcp', 'udp'):
        extra, limit = ['dst_port_fixed'], 65535
    else:
        return None
    if any(type(f.get(k)) is not int or not 0 <= f[k] <= limit for k in extra):
        return None
    return json.dumps({k: f[k] for k in keys + extra}, sort_keys=True)


def metrics(y, p):
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    f1 = [2*cm[c,c]/(cm[c].sum()+cm[:,c].sum()) if cm[c].sum()+cm[:,c].sum() else 0 for c in (1,2)]
    return {'rows': len(y), 'errors': int((y != p).sum()), 'MS_F1': float(np.mean(f1)),
        'M_correct': int(cm[1,1]), 'M_total': int(cm[1].sum()),
        'S_correct': int(cm[2,2]), 'S_total': int(cm[2].sum())}


def bootstrap(y, ref, pred, groups, folds, n=2000, stratified=True):
    u, ix = np.unique(groups, return_inverse=True)
    parts = np.zeros((len(u), 2, 3, 3), dtype=np.int64)
    np.add.at(parts, (ix, 0, y, ref), 1)
    np.add.at(parts, (ix, 1, y, pred), 1)
    strata = [np.unique(ix[folds == k]) for k in range(3)] if stratified else [np.arange(len(u))]
    rng = np.random.default_rng(10201)
    delta = []
    for _ in range(n):
        take = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        cm = parts[take].sum(0)
        f = np.mean([2*cm[:,c,c]/np.maximum(cm[:,c,:].sum(1)+cm[:,:,c].sum(1),1) for c in (1,2)], axis=0)
        delta.append(f[1]-f[0])
    return {'units': len(u), 'resamples': n, 'stratified_by_fold': stratified,
        'delta_95_interval': np.quantile(delta,[.025,.975]).tolist(),
        'resample_positive_fraction_not_posterior': float(np.mean(np.array(delta)>0))}


def main():
    DEST.mkdir(exist_ok=False)
    inputs = [OLD/'source_rows.parquet', OLD/'oof_predictions.parquet', OLD/'N1_ASA_group_map.parquet',
              ROOT/'artifacts/v75_four_arm_20260921_r2/rows.parquet',
              ROOT/'artifacts/v75_four_arm_20260921_r2/projections.parquet', OLD/'selection.json',
              OLD/'source_variant_audit.json']
    receipts = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    source = pd.read_parquet(inputs[0])
    oof = pd.read_parquet(inputs[1])
    asa = source[source.is_ASA].copy().reset_index(drop=True)
    assert np.array_equal(asa.row_position, oof.row_position)
    assert np.array_equal(asa.label_index, oof.truth_official)
    assert sha(inputs[1]) == json.loads(inputs[5].read_text())['oof_sha256']

    # Reconstruct the exact v101 label-free connected units, retaining their IDs.
    components = np.unique(source.component)
    parent = np.arange(len(components))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union_frame(frame, key):
        first = {}
        for c, g in frame[['component', key]].drop_duplicates().itertuples(index=False, name=None):
            ci = int(np.searchsorted(components, c))
            if g in first:
                a, b = find(ci), find(first[g])
                if a != b:
                    parent[max(a,b)] = min(a,b)
            else:
                first[g] = ci
    n1 = pd.read_parquet(inputs[2]).set_index('row_position')
    asa['N1_group'] = n1.loc[asa.row_position, 'N1_group'].to_numpy()
    union_frame(asa, 'N1_group')
    union_frame(source, 'fid')
    roots = components[np.array([find(i) for i in range(len(components))])]
    source['merged_root'] = roots[np.searchsorted(components, source.component)]
    rebuilt = np.array([int(hashlib.sha256(f'9901:{r}'.encode()).hexdigest()[:8],16)%3 for r in source.merged_root])
    assert np.array_equal(rebuilt, source.fold)
    asa['merged_root'] = source.set_index('row_position').loc[asa.row_position,'merged_root'].to_numpy()
    for key in ['component','merged_root','fid']:
        assert source.groupby(key).fold.nunique().max() == 1
    assert asa.groupby('N1_group').fold.nunique().max() == 1
    asa[['row_position','component','merged_root','fold','fid']].to_parquet(DEST/'source_ASA_units.parquet', index=False)

    base = pd.read_parquet(inputs[3], columns=['row_position','projection_id','label_index']).set_index('row_position')
    assert np.array_equal(base.loc[asa.row_position,'label_index'], asa.label_index)
    projections = pd.read_parquet(inputs[4], columns=['facts'])
    pids = base.loc[asa.row_position,'projection_id'].to_numpy()
    fact_map = {int(p):json.loads(projections.facts.iat[int(p)]) for p in np.unique(pids)}
    asa['behavior'] = [behavior(fact_map[int(p)]) for p in pids]
    asa['facts_json'] = [projections.facts.iat[int(p)] for p in pids]
    asa['train_same_class_roots'] = -1
    asa['train_other_class_roots'] = -1
    for k in range(3):
        fit = asa[(asa.fold != k)&asa.behavior.notna()]
        counts = fit.groupby(['behavior','label_index']).merged_root.nunique().to_dict()
        for idx in asa.index[(asa.fold == k)&asa.behavior.notna()]:
            b, c = asa.at[idx,'behavior'], int(asa.at[idx,'label_index'])
            asa.at[idx,'train_same_class_roots'] = counts.get((b,c),0)
            asa.at[idx,'train_other_class_roots'] = counts.get((b,3-c),0)
    def bin_support(n):
        return 'incomplete_behavior' if n < 0 else ('zero_same_class' if n == 0 else ('one_same_class_root' if n == 1 else 'two_or_more_same_class_roots'))
    asa['support_bucket'] = asa.train_same_class_roots.map(bin_support)
    y = asa.label_index.to_numpy()
    names = [c for c in oof if c not in ('row_position','component','fold','truth_official')]
    models = {}
    for name in names:
        pred = oof[name].to_numpy()
        asa[name] = pred
        component = pd.DataFrame({'root': asa.merged_root, 'y':y, 'correct':pred==y}).groupby(['root','y']).correct.agg(['mean','size'])
        m = metrics(y,pred)
        m['class_root_macro_recall'] = {str(c):float(component.xs(c, level='y')['mean'].mean()) for c in (1,2)}
        m['fold_metrics'] = {str(k): metrics(y[asa.fold == k],pred[asa.fold == k]) for k in range(3)}
        m['support_slices'] = []
        for (bucket,c),g in asa.groupby(['support_bucket','label_index']):
            take=g.index.to_numpy()
            m['support_slices'].append({'bucket':bucket,'class':int(c),'rows':len(g),'roots':int(g.merged_root.nunique()),'errors':int((pred[take]!=y[take]).sum())})
        models[name] = m
    ref = oof.R0_teacher.to_numpy()
    candidate = oof.R0_MAG_200.to_numpy()
    n1pred = oof.N1_MAG_200.to_numpy()
    ci = {'R0_MAG200_vs_teacher':bootstrap(y,ref,candidate,asa.merged_root.to_numpy(),asa.fold.to_numpy()),
          'N1_MAG200_vs_R0_MAG200':bootstrap(y,candidate,n1pred,asa.merged_root.to_numpy(),asa.fold.to_numpy())}
    counts = asa.groupby(['fid','label_index']).size().unstack(fill_value=0)
    conflict_ids = counts.index[(counts>0).sum(1)>1]
    asa['source_exact_input_conflict'] = asa.fid.isin(conflict_ids)
    pred_array = oof[names].to_numpy()
    all_wrong = (pred_array != y[:,None]).all(1)
    asa['all_10_states_wrong'] = all_wrong
    asa.to_parquet(DEST/'source_error_and_support_ledger.parquet',index=False)
    top = []
    for (fold,root,c),g in asa[candidate!=y].groupby(['fold','merged_root','label_index']):
        top.append({'fold':int(fold),'root':int(root),'class':int(c),'errors':len(g),
            'rows':int(((asa.merged_root==root)&(y==c)).sum()),
            'support_buckets':g.support_bucket.value_counts().to_dict(),
            'behaviors':g.behavior.value_counts().head(3).to_dict()})
    top.sort(key=lambda a:-a['errors'])
    curves = []
    for k in range(3):
        for view in ('R0','N1'):
            for arm in ('ERM','MAG'):
                p=OLD/f'fold{k}_{view}'/f'{arm}_progress.json'
                h=json.loads(p.read_text())
                short=next(x for x in h if x['step']==200); long=h[-1]
                curves.append({'fold':k,'view':view,'arm':arm,
                    'ASA_fraction_of_training_population':short['ASA_denominator']/short['full_denominator'],
                    'compatibility_coefficient_when_CE_in_ASA_units':.01*short['full_denominator']/short['ASA_denominator'],
                    'errors200':short['ASA_train_errors'],'errors_long':long['ASA_train_errors'],
                    'negative200':short['teacher_correct_negative_flips'],'negative_long':long['teacher_correct_negative_flips'],
                    'last_CE_mag_cos':long['CE_magnitude_cosine'],
                    'last_protection_grad_l2':long['protection_gradient_l2']})
                receipts[str(p.relative_to(ROOT))]=sha(p)
    result={'status':'read_only_root_cause_diagnosis_no_fit', 'classifier_fits':0,'calibration_fits':0,
        'source_rows':len(source),'ASA_rows':len(asa),'old_ASA_components':int(asa.component.nunique()),
        'merged_ASA_units':int(asa.merged_root.nunique()),'merged_source_units':int(source.merged_root.nunique()),
        'S_old_components':int(asa[y==2].component.nunique()),'S_merged_units':int(asa[y==2].merged_root.nunique()),
        'all_folds_reconstructed_exactly':True, 'models':models,'bootstrap_correct_group_unit':ci,
        'source_R0_empirical_conflict_floor':int((counts.sum(1)-counts.max(1)).sum()),
        'all_states_wrong':int(all_wrong.sum()),'all_states_wrong_M':int((all_wrong&(y==1)).sum()),
        'all_states_wrong_S':int((all_wrong&(y==2)).sum()),
        'all_states_wrong_without_exact_input_conflict':int((all_wrong&~asa.source_exact_input_conflict).sum()),
        'label_oracle_union_errors_not_deployable':int(all_wrong.sum()),'top_candidate_error_units':top[:15],
        'training_curves':curves,'input_sha256':receipts,
        'limits':['Source OOF is already observed development data; bootstrap is conditional on fixed fitted models.',
                  'Merged units are duplicate/component proxies, not verified enterprises or independent incidents.',
                  'Behavior keys are partial observable facts, not sufficient M/S rules; missing keys are not pooled as one behavior.',
                  'No new classifier, V selection, official submission or external ground truth.']}
    (DEST/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'receipt.json').write_text(json.dumps({'script_sha256':sha(__file__), 'inputs':receipts,
        'outputs':{p.name:sha(p) for p in DEST.iterdir() if p.is_file()}},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','ASA_rows','old_ASA_components','merged_ASA_units','S_merged_units','bootstrap_correct_group_unit','all_states_wrong','source_R0_empirical_conflict_floor']},ensure_ascii=False))
    print(json.dumps({k:models[k] for k in ['R0_teacher','R0_MAG_200']},ensure_ascii=False))


if __name__ == '__main__':
    main()
