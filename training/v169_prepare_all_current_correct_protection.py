"""Freeze all currently correct V164 original rows including206 mixed controls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,sha

OUT=ROOT/'artifacts/v169_initial_current_correct_protection_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();goldfile=ROOT/'data/official/train.parquet';gold=pd.read_parquet(goldfile,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();files=[Path(__file__).resolve(),goldfile];roles=[]
    for role in [0,1,2]:
        reference=ROOT/f'artifacts/v164_short_supervised_trajectory_20261002/role{role}/endpoint/OOF_original_rows.parquet';rows=pd.read_parquet(reference)
        assert not rows.row_position.duplicated().any() and np.array_equal(rows.truth,gold[rows.row_position])
        correct=rows.pred.eq(rows.truth);old=rows.protected_correct
        assert not (old&~correct).any();extra=correct&~old
        assert int(extra.sum())==[106,6,94][role] and not rows.loc[extra,'pure_current_input'].any()
        ledger=rows.loc[correct,['row_position','local','truth','root','pure_current_input','protected_correct']].copy();ledger=ledger.rename(columns={'protected_correct':'previously_protected'})
        path=OUT/f'role{role}_all_current_correct_original_rows.parquet';ledger.to_parquet(path,index=False)
        extra_path=OUT/f'role{role}_additional_mixed_original_rows.parquet';rows.loc[extra,['row_position','local','truth','root','pure_current_input']].to_parquet(extra_path,index=False)
        roles.append(dict(role=role,all_current_correct_rows=len(ledger),added_current_correct_mixed_rows=int(extra.sum()),reference=reference.relative_to(ROOT).as_posix(),reference_sha256=sha(reference),ledger=path.relative_to(ROOT).as_posix(),ledger_sha256=sha(path),extra=extra_path.relative_to(ROOT).as_posix(),extra_sha256=sha(extra_path)))
        files.extend([reference,path,extra_path])
    report=dict(status='V169_all_V164_current_correct_original_rows_and206_mixed_controls_frozen',roles=roles,added_mixed_rows=[106,6,94],both_arms_same_frozen_protection=True,all_mixed_scoring_and_fixed_targets_preserved=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    (OUT/'review.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status=report['status'],added_mixed_rows=[106,6,94],official_calls=0)))

if __name__=='__main__':main()
