"""Freeze proven TRAIN mastery only after actual endpoint/window/input replays."""
import pandas as pd
import numpy as np
from v142_runtime import ROOT,OUT,read,save,sha,require_run_seal
from v135_runtime import load_data,fit_context


def main():
    require_run_seal(ROOT/'training/v142_train.py');delivery=read(OUT/'final_delivery.json');verify=read(OUT/'verification.json');learning=read(OUT/'learning_qualification.json')
    if (OUT/'verified_TRAIN_mastery_registry.json').exists():raise FileExistsError('Preserve accepted capability')
    if not delivery['issue_solved'] or not learning['issue_solved']:
        save(OUT/'mastery_freeze_refused.json',{'first_issue_closed':False,'reason':'Actual training/guard/window qualification did not pass','new_fits':0,'new_updates':0});print('Mastery not accepted');return
    if not all(v['exact_zero_and_endpoint_replay'] and v['cache_free_numeric_input_replay_count']==22546 for v in verify['audits']):raise ValueError('Actual numerical function replay incomplete')
    _,d=load_data();scopes=[];independent=[]
    for f in range(3):
        rows=pd.read_parquet(OUT/f'fold{f}_S2/endpoint_original_rows.parquet');frame,_,pure,_,_=fit_context(d,f)
        if not np.array_equal(rows.row_position,frame.row_position) or not np.array_equal(rows.truth,frame.truth):raise ValueError('Official role truth mismatch')
        pr=pure[frame.local].astype(bool)
        if (rows.pred.ne(rows.truth)&pr).any() or (rows.pred.ne(rows.truth)&rows.truth.eq(1)).any():raise ValueError('Classification mastery contradicted')
        if int((rows.pred.ne(rows.truth)&rows.truth.eq(2)).sum())!=[22,6,28][f]:raise ValueError('Mixed majority gate contradicted')
        info=learning['folds'][f]
        if not info['stable_window_mastered'] or len({v['parameter_sha256'] for v in info['last_states']})!=5:raise ValueError('Stable state window missing')
        guard=rows[rows.pred.eq(rows.truth)][['row_position','training_role','truth','pred']].copy()
        path=OUT/f'fold{f}_verified_all_correct_TRAIN_guard.parquet';guard.to_parquet(path,index=False);independent.extend(guard.row_position.to_list())
        cp=OUT/f'fold{f}_S2/endpoint.pt';baseline=OUT/f'fold{f}_S2/endpoint_original_rows.parquet'
        scopes.append({'id':f'TRAIN-MASTERED-fold{f}-V142','training_role':f,'all_correct_original_role_rows':len(guard),'pure_original_role_rows':int(pr.sum()),
                       'guard':path.relative_to(ROOT).as_posix(),'guard_sha256':sha(path),'checkpoint':cp.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(cp),
                       'baseline_rows':baseline.relative_to(ROOT).as_posix(),'baseline_rows_sha256':sha(baseline),'new_errors_allowed':0,
                       'scope':'Registered complete TRAIN role, correct rows including mixed-majority M; not source transfer or formal model adoption.'})
    save(OUT/'verified_TRAIN_mastery_registry.json',{'first_issue_closed':True,'scopes':scopes,'pure_TRAIN_role_rows':sum(s['pure_original_role_rows'] for s in scopes),
          'protected_correct_TRAIN_role_rows':sum(s['all_correct_original_role_rows'] for s in scopes),'independent_protected_official_rows':len(set(independent)),
          'prior_V138_and_V140_guards_preserved':True,'next_guard_entry':'training/v142_retention_check.py','deployable':False,
          'delivery_sha256':sha(OUT/'final_delivery.json'),'verification_sha256':sha(OUT/'verification.json'),'source_sha256':sha(__file__)})
    print({'first_issue_closed':True,'protected_role_rows':sum(s['all_correct_original_role_rows'] for s in scopes),'independent_rows':len(set(independent))},flush=True)


if __name__=='__main__':main()
