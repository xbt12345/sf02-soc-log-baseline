"""Bind completed independent entry review before any new real preflight call."""
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_runtime_v4 import OUT,save,require
def main():
    p=ROOT/'artifacts/v159_independent_numeric_entry_activation_review_20261002/audit.json';q=read(p)
    check_bindings(q['source_sha256']);require(ROOT/'training/v159_boundary_train_v4.py')
    target=OUT/'independent_activation_review_binding.json';assert not target.exists() and not list(OUT.glob('preflight*'))
    save(target,dict(status='completed_independent_numeric_entry_review_bound_before_new_real_preflight',independent_review=p.relative_to(ROOT).as_posix(),independent_review_sha256=sha(p),source_sha256=sha(__file__),cumulative_prior_head_calls=124,cumulative_prior_full_class_gradients=8,new_preflight_head_cap=336,new_preflight_gradient_cap=12,official_new_heads=0,official_new_gradients=0,official_fits=0,official_updates=0,quality_acceptance=False))
    print('actual independent numeric entry review verified and prospectively bound')
if __name__=='__main__':main()
