"""Version lossless original blocker masks and same-point q/logq references."""
from experiment_review import ROOT,sha
from v169_prepare_training_entry_v2 import change

def main():
    old=ROOT/'training/v169_prior_pair_training_entry_v5.py';target=ROOT/'training/v169_prior_pair_training_entry_v6.py'
    if target.exists():raise FileExistsError(target)
    text=old.read_text(encoding='utf-8')
    text=change(text,'from v169_saved_state_quality import review as saved_state_quality','from v169_saved_state_quality import review as saved_state_quality\nfrom v169_measurement_output_references import save as save_measurement_outputs')
    text=change(text,"np.savez_compressed(folder/'all_blocking_original_row_identity.npz',row_position=blocks.row_position.to_numpy(),local=blocks.local.to_numpy(),truth=blocks.truth.to_numpy(),rival=blocks.rival.to_numpy(),scope=blocks.scope.to_numpy(dtype=str))", "oo=self.ctx['OOF_rows'];de=self.ctx['deployment_rows']\n            np.savez_compressed(folder/'full_original_protection_and_blocker_masks.npz',OOF_protected=oo.protected_correct.to_numpy(bool),OOF_blocked=(oo.protected_correct.to_numpy(bool)&(q[oo.local.to_numpy()].argmax(1)!=oo.truth.to_numpy())),deployment_blocked=(de.initial_correct.to_numpy(bool)&(dq[de.local.to_numpy()].argmax(1)!=de.truth.to_numpy())))\n            save(folder/'original_reference_identity.json',{scope:dict(path=(PRIOR/f'role{self.role}/endpoint/{scope}_original_rows.parquet').relative_to(ROOT).as_posix(),sha256=sha(PRIOR/f'role{self.role}/endpoint/{scope}_original_rows.parquet')) for scope in ['OOF','deployment']})")
    text=change(text,"np.savez_compressed(target/f'repeat{repetition}_outputs.npz',margin=np.array([value['margin']]),q=value['q'],logq=value['logq']);pairs.append(value)", "save_measurement_outputs(target,repetition,value,trial['folder']/'complete_outputs.npz',scope,chunk);pairs.append(value)")
    target.write_bytes(text.encode('utf-8'));print(dict(status='V169_entry_v6_factorized_all_blocker_rows_and_exact_same_point_measurement_outputs',old_sha256=sha(old),new_sha256=sha(target),official_calls=0))

if __name__=='__main__':main()
