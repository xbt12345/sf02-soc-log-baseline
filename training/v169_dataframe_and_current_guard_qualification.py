"""Real DataFrame lifecycle branch and actual206 mixed guard wiring, zero model calls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,sha
from v169_pair_lifecycle_v3 import run_fit,blocker_count
from v169_pair_lifecycle_qualification_v2 import Oracle
from v165_fixed_endpoint_decision_floor_diagnostic import load_context
from v169_current_correct_context import apply
from v160_fixed_endpoint_diagnostic_v3 import actual_blockers
from v159_boundary_train_v4 import stats

OUT=ROOT/'artifacts/v169_dataframe_and_current_guard_qualification_20261002'

class FrameOracle(Oracle):
    def __init__(self,folder,empty=False):super().__init__(folder);self.empty=empty;self.restorations=0
    def direction_trial(self,*args):
        table=pd.DataFrame(columns=['scope','local','truth','rival']) if self.empty else pd.DataFrame([dict(scope='OOF',local=0,truth=2,rival=1)])
        return dict(accepted=False,blockers=table,parameter_sha256=self.identity())
    def correction_trial(self,ob,gs,trial,active,normals,state,correction):self.restorations+=1;return dict(trial,accepted=True)

def main():
    assert not OUT.exists();OUT.mkdir();cases=[]
    for empty in [False,True]:
        backend=FrameOracle(OUT/('empty_frame' if empty else 'nonempty_frame'),empty);result=run_fit(backend)
        assert backend.closed and backend.counter.log.closed
        if empty:assert result['accepted_updates']==1 and result['status']=='finite_rejection_without_protected_blocker' and backend.restorations==0
        else:assert result['fixed_endpoint_reached'] and result['accepted_updates']==20 and backend.restorations==19
        assert result['exception'] is None
        cases.append(dict(real_dataframe_empty=empty,accepted_updates=result['accepted_updates'],corrections=backend.restorations,closed=True))
    try:blocker_count(np.array([1]))
    except TypeError:pass
    else:raise AssertionError('Unknown blocker container must refuse')
    files=[Path(__file__).resolve(),ROOT/'training/v169_pair_lifecycle_v3.py',ROOT/'training/v169_current_correct_context.py',ROOT/'training/v169_prior_pair_training_entry_v9.py',ROOT/'artifacts/v169_initial_current_correct_protection_20261002/review.json'];roles=[]
    for role in [0,1,2]:
        ctx=load_context(role);initial=ctx['OOF_rows'].protected_correct.to_numpy(bool).copy();info=apply(ctx,role);mask=ctx['OOF_rows'].protected_correct.to_numpy(bool)
        assert np.all(~initial|mask) and int((mask&~initial).sum())==[106,6,94][role]
        reference=ROOT/f'artifacts/v164_short_supervised_trajectory_20261002/role{role}/endpoint';qfile,lfile=reference/'OOF_q.npy',reference/'OOF_logq.npy';q,lp=np.load(qfile),np.load(lfile)
        assert stats(ctx,q,'OOF')['protected_regressions']==0
        # Force the actual added mixed row's legal prediction to change, then
        # run the SAME original-row statistics and blocker function as entry.
        row=ctx['OOF_rows'].loc[mask&~initial].iloc[0];local=int(row.local);truth=int(row.truth);rival=1 if truth==2 else 2;wrong=q.copy();wrong[local]=[0.,0.,0.];wrong[local,rival]=1.
        blocks=actual_blockers(ctx,wrong,lp,'OOF');actual=stats(ctx,wrong,'OOF')
        assert row.row_position in set(blocks.row_position) and actual['protected_regressions']>0
        before=len(ctx['OOF_rows']);assert before==info['complete_original_rows'] and len(ctx['OOF_rows'])==before
        roles.append(dict(role=role,added_mixed_rows=info['additional_mixed_rows'],all_original_rows_retained=before,added_row_regression_in_actual_full_guard_and_blockers=True));files.extend([qfile,lfile])
    report=dict(status='V169_real_DataFrame_and_all206_current_mixed_correct_guard_interfaces_qualified',dataframe_cases=cases,unknown_blocker_type_refused=True,all_current_correct_guard_cases=roles,total_added_mixed_rows=206,old_guard_not_revoked=True,actual_original_row_statistics_and_blocker_code_used=True,modified_saved_q_is_guard_oracle_not_new_model_inference=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    (OUT/'qualification.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status=report['status'],real_dataframe_cases=2,added_mixed_rows=206,official_calls=0)))

if __name__=='__main__':main()
