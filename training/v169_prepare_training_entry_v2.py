"""Version the draft after actual root terminal-replay failure, retain v1."""
from experiment_review import ROOT,sha

def change(text,old,new):
    if text.count(old)!=1:raise ValueError('Unique source fragment required: '+old[:80])
    return text.replace(old,new)

def main():
    original=ROOT/'training/v169_prior_pair_training_entry.py';target=ROOT/'training/v169_prior_pair_training_entry_v2.py'
    if target.exists():raise FileExistsError(target)
    text=original.read_text(encoding='utf-8')
    text=change(text,'from v169_pair_lifecycle import Schedule,run_fit,paired_result','from v169_pair_lifecycle_v2 import Schedule,run_fit,paired_result')
    text=text.replace("training/v169_prior_pair_training_entry.py'","training/v169_prior_pair_training_entry_v2.py'")
    text=change(text,"self.last_observation=None;self.current_state=0","self.last_observation=None;self.current_state=0;self.resources_closed=False")
    text=change(text,"            self._persist_observation(folder,ob)\n", "            for scope in ['OOF','deployment']:\n                ids=used if scope=='OOF' else np.arange(22546)\n                for name,kind in [('q','probability'),('logq','log_probability')]:\n                    previous=np.load(PRIOR/f'role{self.role}/endpoint/{scope}_{name}.npy')\n                    if not repeat_values(ob[f'{scope}_{name}'][ids],previous[ids],kind)['passed']:raise RuntimeError('V164 original zero-step risk/argmax/output drift')\n            self._persist_observation(folder,ob)\n")
    text=change(text,"            save(target/'commit.json',dict(state=state,previous_parameter_sha256=origin,parameter_sha256=self.identity(),repairs=int(repair.sum()),cumulative_protected_rows=int((old|repair).sum()),bootstrap_included_in20=True))\n            self.last_observation=trial;self.ctx['baseline_stats']=trial['guard']['OOF'];self.current_state=state;committed=True", "            acknowledged_state=self.snapshot();acknowledged_ledger=self.protection_snapshot()\n            receipt=dict(state=state,previous_parameter_sha256=origin,parameter_sha256=self.identity(),repairs=int(repair.sum()),cumulative_protected_rows=int((old|repair).sum()),bootstrap_included_in20=True)\n            save(target/'commit.json',receipt)\n            self.last_observation=trial;self.ctx['baseline_stats']=trial['guard']['OOF'];self.current_state=state;committed=True\n            return dict(state=acknowledged_state,ledger=acknowledged_ledger,receipt=receipt)")
    text=change(text,"self.counter.close()\n\ndef main():", "self.close_resources()\n    def terminal_failure(self,failure):\n        save(self.folder/'terminal_failure.json',dict(failure,accepted_updates=self.current_state,complete_parameters=width(self.arm),protection_rows=int(self.protection_snapshot().sum())))\n    def close_resources(self):\n        if not self.resources_closed:\n            self.counter.close();self.resources_closed=True\n\ndef main():")
    target.write_bytes(text.encode('utf-8'))
    print(dict(status='V169_entry_v2_prepared_after_root_actual_terminal_replay_counter_close_failure',old_sha256=sha(original),new_sha256=sha(target),official_calls=0))

if __name__=='__main__':main()
