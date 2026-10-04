"""Canonical-input support versus correct controls; original labels and roles unchanged."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v135_runtime import ROOT,OUT,read,save,sha,require_run_seal,load_data,fit_context

DEST=ROOT/'artifacts/v136_root_review_20260930'

def main():
    require_run_seal(ROOT/'training/v135_train.py');x,d=load_data()
    target=DEST/'canonical_support_controls.json'
    if target.exists():raise FileExistsError(target)
    member=read(DEST/'diagnosis_receipt.json')
    for file,h in member['output_sha256'].items():
        if sha(DEST/file)!=h:raise ValueError('Frozen diagnosis changed '+file)
    ledger_path=OUT/'ASA_prediction_ledger.parquet';ladder_path=ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet'
    ledger=pd.read_parquet(ledger_path);lad=pd.read_parquet(ladder_path)
    if not np.array_equal(ledger.row_position,d.row_position) or not np.array_equal(lad.row_position,d.row_position):raise ValueError('Row identity mismatch')
    residual=pd.read_parquet(DEST/'TRAIN_pure_residuals.parquet')
    controls=[];held_reports=[];bad=[]
    for fold in range(3):
        fit,c,pure,totals,ids=fit_context(d,fold)
        counts=fit.groupby(['canonical_key','truth']).size().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
        sources=fit.groupby(['canonical_key','truth']).root.nunique().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
        mass=counts.reindex(fit.canonical_key).to_numpy();roots=sources.reindex(fit.canonical_key).to_numpy()
        y=fit.truth.to_numpy();ii=fit.local.to_numpy();q=np.load(DEST/f'fold{fold}_R_decay_frozen.npz')['mean_probability']
        fit=fit.copy();fit['same_class_mass']=mass[np.arange(len(fit)),y-1]
        fit['same_class_sources']=roots[np.arange(len(fit)),y-1]
        fit['wrong']=q[ii].argmax(1)!=y;fit['pure']=pure[ii].astype(bool)
        for cl in [1,2]:
            part=fit[fit.truth.eq(cl)&fit.pure];m=part.same_class_mass.le(4)&part.same_class_sources.eq(1)
            controls.append({'fold':fold,'class':cl,'pure_original_role_rows':len(part),
                'sparse_single_source_rows':int(m.sum()),'sparse_single_source_errors':int(part.loc[m,'wrong'].sum()),
                'sparse_single_source_correct':int((~part.loc[m,'wrong']).sum()),'other_errors':int(part.loc[~m,'wrong'].sum())})
        for arm in ['R_const','R_decay','O_const','O_decay']:
            z=residual[residual.fold.eq(fold)&residual.arm.eq(arm)].copy()
            reference=fit[['row_position','canonical_key','same_class_mass','same_class_sources']]
            z=z.merge(reference,on='row_position',how='left',validate='one_to_one');bad.append(z)
        h=d[d.fold.eq(fold)].copy();hm=counts.reindex(h.canonical_key).fillna(0).to_numpy();hy=h.truth.to_numpy()
        same=hm[np.arange(len(h)),hy-1];other=hm[np.arange(len(h)),2-hy]
        h['fit_input_status']=np.select([(same>0)&(other==0),(same>0)&(other>0),(same==0)&(other>0)],['same_only','mixed','opposite_only'],default='unseen')
        h=h.merge(ledger[['row_position','pred_A0','pred_R_decay']],on='row_position',validate='one_to_one')
        h['wrong']=h.pred_R_decay.ne(h.truth)
        for name,m in [('all',np.ones(len(h),bool)),('A0_correct_R_wrong',h.pred_A0.eq(h.truth)&h.wrong),('root2868_M',h.root.eq(2868)&h.truth.eq(1))]:
            part=h[m]
            for (cl,status),z in part.groupby(['truth','fit_input_status']):
                held_reports.append({'fold':fold,'population':name,'class':int(cl),'fit_input_status':status,'rows':len(z),'errors':int(z.wrong.sum())})
    joined=pd.concat(bad,ignore_index=True)
    joined.to_parquet(DEST/'canonical_TRAIN_residuals.parquet',index=False)
    z=lad[lad.root.eq(2868)&lad.truth.eq(1)]
    root_support=z.groupby(['destination_M_rows','destination_S_rows','destination_M_roots','destination_S_roots']).size().reset_index(name='original_rows').to_dict('records')
    save(target,{'status':'canonical_actual_input_support_recomputed','TRAIN_controls':controls,'HELD_diagnostics':held_reports,
        'root2868_M_destination_support':root_support,'actual_input_keys':int(d.canonical_key.nunique()),
        'local_indices':int(d.local.nunique()),'multi_local_actual_keys':int(d.groupby('canonical_key').local.nunique().gt(1).sum()),
        'residuals':{arm:{'original_role_rows':len(part),'independent_original_rows':int(part.row_position.nunique()),
            'canonical_inputs':int(part.canonical_key.nunique()),'all_same_class_mass_2_to_4':bool(part.same_class_mass.between(2,4).all()),
            'all_single_fit_source':bool(part.same_class_sources.eq(1).all())} for arm,part in joined.groupby('arm')},
        'new_fits':0,'new_updates':0,
        'limits':['Numerically unseen is not proof of semantically unseen behavior.',
                  'Source roots are isolation components, not confirmed independent organizations.',
                  'Sparse correct controls refute lack of support as a sufficient explanation.']})
    bindings=[Path(__file__),ledger_path,ladder_path,DEST/'diagnosis_receipt.json',DEST/'TRAIN_pure_residuals.parquet']
    save(DEST/'support_receipt.json',{'new_fits':0,'new_updates':0,'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in bindings},
        'output_sha256':{p.name:sha(p) for p in [target,DEST/'canonical_TRAIN_residuals.parquet']}})
    print(json.dumps({'R_decay':read(target)['residuals']['R_decay'],'sparse_correct_controls':sum(z['sparse_single_source_correct'] for z in controls),'new_updates':0}),flush=True)

if __name__=='__main__':main()
