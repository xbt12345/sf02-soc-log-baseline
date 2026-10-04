"""Evaluate frozen rotation and locked-role gates; no model/epoch selection."""
from run_v75 import ROOT,read,save,sha
from v81_training_contract import compare

DEST=ROOT/'artifacts/v87_solver_supervision_r2_20260927'


def main():
    assert read(DEST/'verification.json')['status']=='passed'
    primary=read(DEST/'continuation.json');winner=primary['primary_selected'];checks=[]
    if winner is None:
        out={'primary_selected':None,'rotation_passed':False,'locked_regression_passed':False,
             'final_full_fit_authorized':False,'support_withdrawal_refits_authorized':False,
             'reason':'no_primary_candidate','source_sha256':sha(__file__)}
        save(DEST/'completion_gates.json',out);return
    rot=read(DEST/'rotation/continuation.json');assert read(DEST/'rotation/verification.json')['status']=='passed'
    folders=[(1,DEST/'fold1'),(3,DEST/'rotation/fold3'),(4,DEST/'rotation/fold4')]
    for h,folder in folders:
        name=winner['name']
        d=next(d for d in read(folder/'diagnosis.json') if d['name']==name)
        # The paired baseline confusion is reconstructed from the final confusion
        # plus independent row predictions rather than inferred from flip totals.
        import numpy as np
        from v79_execute import rows
        from v82_capacity import LAST
        r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy')
        teacherfolder=ROOT/'artifacts/v85_protection_20260927/fold1' if h==1 else folder
        old=np.load(teacherfolder/'teacher_prediction.npy')[fid]
        for role in ['fit_full','C','H']:
            mask=~r.fold.isin([0,2,h]).to_numpy() if role=='fit_full' else r.fold.eq(2 if role=='C' else 0).to_numpy()
            base=np.bincount(y[mask]*3+old[mask],minlength=9).reshape(3,3)
            actual=next(i for i in d['roles'] if i['role']==role)
            guard=compare(base,np.asarray(actual['cm']),role!='fit_full')
            ok=guard['eligible'] and actual['negative_flips']==0
            checks.append({'fold':h,'name':name,'role':role,'passed':bool(ok),'repairs':actual['positive_flips'],
                           'regressions':actual['negative_flips'],'regressions_by_class':actual['negative_flips_by_class'],'guard':guard})
    passed=rot['rotation_passed'] and all(c['passed'] for c in checks)
    out={'primary_selected':winner,'rotation_passed':rot['rotation_passed'],'locked_regression_checks':checks,
         'locked_regression_passed':all(c['passed'] for c in checks),
         'final_full_fit_authorized':bool(passed),'support_withdrawal_refits_authorized':bool(passed),
         'reason':'all_development_gates_passed' if passed else 'frozen_rotation_or_locked_role_regression_failed',
         'source_sha256':sha(__file__),'new_checkpoint_selection':False}
    save(DEST/'completion_gates.json',out)
    print(__import__('json').dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
