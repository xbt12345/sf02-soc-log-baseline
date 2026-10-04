"""Prepare scoped seal upgrade, phase disk checks, initial risk and complete S audit."""
from experiment_review import ROOT,sha
from v169_prepare_training_entry_v2 import change

def main():
    old=ROOT/'training/v169_prior_pair_training_entry_v2.py';target=ROOT/'training/v169_prior_pair_training_entry_v3.py'
    if target.exists():raise FileExistsError(target)
    text=old.read_text(encoding='utf-8');start=text.index('def require():');end=text.index('class PairCounter:',start)
    text=text[:start]+"def require(phase='running'):\n    from v169_pair_execution_review import require_run_seal\n    return require_run_seal(OUT/'run_seal.json',__file__,phase)\n\n"+text[end:]
    text=change(text,'from v169_pair_lifecycle_v2 import Schedule,run_fit,paired_result','from v169_pair_lifecycle_v2 import Schedule,run_fit,paired_result\nfrom v169_saved_state_quality import review as saved_state_quality')
    text=change(text,"            self._persist_observation(folder,ob)\n", "            accepted=read(PRIOR/f'role{self.role}/fit.json')['permanent_updates']\n            old_risk=PRIOR/f'role{self.role}/parameter_point{accepted}/class1_repeat0'\n            for name in first[0]:\n                if not repeat_values(first[0][name],np.load(old_risk/f'{name}.npy'),'risk')['passed']:raise RuntimeError('V164 original target/full risk zero-step drift')\n            self._persist_observation(folder,ob)\n")
    text=change(text,"        return dict(state=state,parameter_sha256=self.identity(),beta=beta,prior_coefficient=float(np.exp(beta)),OOF=stats(self.ctx,trial['OOF_q'],'OOF'),deployment=stats(self.ctx,trial['deployment_q'],'deployment'))", "        coefficient=float(self.model.beta.exp().detach()) if self.arm=='B' else 1.\n        reference=pd.read_parquet(PRIOR/f'role{self.role}/endpoint/OOF_original_rows.parquet')\n        cohort=pd.read_parquet(ROOT/f'artifacts/v169_learnable_prior_pair_plan_20261002/role{self.role}_all_S_initial_prior_cohort.parquet')\n        prior=np.log(np.maximum(np.asarray(self.ctx['OOF'],np.float64).mean(1),1e-12))\n        quality=saved_state_quality(reference,trial['OOF_q'],trial['OOF_logq'],cohort,prior,self.model.output_weight.detach().cpu().numpy(),coefficient)\n        save(self.folder/f'accepted{state}/full_original_quality_and_fixed_S_slices.json',quality)\n        return dict(state=state,parameter_sha256=self.identity(),beta=beta,prior_coefficient=coefficient,OOF=stats(self.ctx,trial['OOF_q'],'OOF'),deployment=stats(self.ctx,trial['deployment_q'],'deployment'),saved_state_quality=quality)")
    text=change(text,'    plan=require();configure();results={}','    plan=require(\'initial\');configure();results={}')
    target.write_bytes(text.encode('utf-8'));print(dict(status='V169_entry_v3_prepared_dedicated_scope_seal_and_resource_phase_checks',old_sha256=sha(old),new_sha256=sha(target),official_calls=0))

if __name__=='__main__':main()
