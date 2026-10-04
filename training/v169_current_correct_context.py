"""Wire the frozen all-current-correct ledger into the original full OOF guard."""
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings

def apply(ctx,role):
    folder=ROOT/'artifacts/v169_initial_current_correct_protection_20261002';report=read(folder/'review.json');check_bindings(report['source_sha256']);spec=report['roles'][role]
    frozen=pd.read_parquet(ROOT/spec['reference']);ledger=pd.read_parquet(ROOT/spec['ledger']);rows=ctx['OOF_rows']
    keys=['row_position','local','truth','root','pure_current_input']
    if not np.array_equal(rows[keys].to_numpy(),frozen[keys].to_numpy()) or ledger.row_position.duplicated().any():raise ValueError('All original row identity must match V164 and frozen ledger')
    correct=frozen.pred.eq(frozen.truth);expected=frozen.loc[correct,keys]
    if not np.array_equal(expected.to_numpy(),ledger[keys].to_numpy()):raise ValueError('Current mixed/pure correct rows omitted from frozen protection')
    old=rows.protected_correct.to_numpy(bool);mask=correct.to_numpy(bool)
    if np.any(old&~mask):raise ValueError('Cannot revoke older protected correct rows')
    extra=mask&~old
    if int(extra.sum())!=[106,6,94][role] or rows.loc[extra,'pure_current_input'].any():raise ValueError('Expected all206 current correct mixed rows must be added')
    ctx['OOF_rows']=rows.copy();ctx['OOF_rows']['protected_correct']=mask
    return dict(all_original_current_correct_protected=True,role=role,complete_original_rows=len(rows),protected_rows=int(mask.sum()),additional_mixed_rows=int(extra.sum()),ledger=spec['ledger'],ledger_sha256=spec['ledger_sha256'],all_mixed_rows_still_scored=True)
