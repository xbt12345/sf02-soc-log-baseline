"""Observed input support and component concentration; diagnostic labels only."""
import json
import hashlib
import numpy as np
import pandas as pd
from run_v75 import ROOT, OUT, read, save, sha
from v79_execute import rows

DEST = ROOT / 'artifacts/v81_diagnosis_20260927'
LAST = ROOT / 'artifacts/v79_execution_20260927'
FIELDS = ['action', 'outcome', 'transport_protocol', 'src_role', 'dst_role', 'icmp_type', 'icmp_code', 'auth_result']


def summarize(ids, y, fit, take, pred, width):
    counts = np.bincount(ids[fit] * 3 + y[fit], minlength=width * 3).reshape(-1, 3)
    target = np.bincount(ids[take] * 3 + y[take], minlength=width * 3).reshape(-1, 3)
    available = counts.sum(1)
    matches = counts[ids[take]]
    yy, pp = y[take], pred[take]
    true_support = matches[np.arange(len(yy)), yy]
    total = matches.sum(1)
    buckets = np.where(total == 0, 'unseen_input', np.where(true_support == 0, 'seen_only_other_labels',
                       np.where((matches > 0).sum(1) == 1, 'seen_true_label_only', 'seen_mixed_labels')))
    tab = []
    for cl in range(3):
        for bucket in np.unique(buckets):
            m = (yy == cl) & (buckets == bucket)
            if m.any():
                tab.append({'class': cl, 'bucket': str(bucket), 'rows': int(m.sum()), 'errors': int((pp[m] != yy[m]).sum())})
    active = target.sum(1) > 0
    conflicting = (target > 0).sum(1) > 1
    return {'support_buckets': tab, 'target_empirical_same_input_error_floor': int((target.sum(1)-target.max(1))[active].sum()),
            'target_conflicting_groups': int((conflicting & active).sum()),
            'fit_empirical_same_input_error_floor': int((counts.sum(1)-counts.max(1)).sum()),
            'scope': 'Floor is empirical for this chosen representation and role, not irreducible raw-data noise or guaranteed learnable excess.'}


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.mkdir()
    r = rows(); y = r.label_index.to_numpy(); fid = np.load(LAST/'row_feature_id.npy')
    pred = np.load(LAST/'P1_B_ovr_feature_prediction.npy')[fid]
    fit = ~r.fold.isin([0,2]).to_numpy()
    p = pd.read_parquet(OUT/'projections.parquet', columns=['facts'])
    facts = [json.loads(s) for s in p.facts]
    canonical = [json.dumps(f, sort_keys=True, separators=(',', ':')) for f in facts]
    fids, funique = pd.factorize(np.asarray(canonical, dtype=object), sort=False)
    coarse = [json.dumps({k:f[k] for k in FIELDS if k in f}, sort_keys=True, separators=(',', ':')) for f in facts]
    cids, cunique = pd.factorize(np.asarray(coarse, dtype=object), sort=False)
    factid = fids[r.projection_id.to_numpy()]; coarseid = cids[r.projection_id.to_numpy()]
    roles = {'fit':fit, 'C':r.fold.eq(2).to_numpy(), 'H':r.fold.eq(0).to_numpy()}
    out = {'new_fits': 0, 'source_sha256':sha(__file__), 'coarse_fields': FIELDS, 'roles':{}}
    for role, mask in roles.items():
        out['roles'][role] = {}
        for route in ['ALL', 'asa', 'cef_fields', 'vpc_v2', 'windows_message']:
            take = mask if route == 'ALL' else mask & r.route.eq(route).to_numpy()
            entry = {'rows': int(take.sum()), 'errors':int((take & (pred != y)).sum())}
            for name, ids, width in [('complete_encoding', fid, int(fid.max())+1), ('facts_only',factid,len(funique)), ('coarse_behavior',coarseid,len(cunique))]:
                entry[name] = summarize(ids,y,fit,take,pred,width)
            errs = r[take & (pred != y)]
            groups = errs.groupby(['component','label_index']).size().rename('errors').sort_values(ascending=False)
            entry['error_components'] = int(errs.component.nunique())
            entry['top_error_components'] = groups.head(10).reset_index().to_dict('records')
            out['roles'][role][route] = entry
    # Rank fit errors by exact feature group. This is model learnability material, never target tuning.
    ix = np.flatnonzero(fit & (pred != y))
    error_rows = r.iloc[ix].copy();error_rows['feature_id']=fid[ix];error_rows['prediction']=pred[ix]
    error_rows.to_parquet(DEST/'fit_error_rows.parquet',index=False)
    grouped = error_rows.groupby(['feature_id','route','label_index','prediction']).agg(rows=('row_position','size'),components=('component','nunique'),example_row=('row_position','first')).sort_values('rows',ascending=False).reset_index()
    grouped.to_csv(DEST/'fit_error_groups.csv',index=False)
    out['fit_error_feature_groups'] = int(error_rows.feature_id.nunique())
    out['all_training_label_counts'] = np.bincount(y,minlength=3).tolist()
    out['input_sha256'] = {'rows':sha(OUT/'rows.parquet'),'feature_ids':sha(LAST/'row_feature_id.npy'), 'predictions':sha(LAST/'P1_B_ovr_feature_prediction.npy')}
    save(DEST/'support_diagnosis.json',out)
    print(json.dumps({'fit':out['roles']['fit']['asa'],'C':out['roles']['C']['asa'],'H':out['roles']['H']['asa'],'fit_error_feature_groups':out['fit_error_feature_groups']},ensure_ascii=False,indent=2))


if __name__ == '__main__':main()
