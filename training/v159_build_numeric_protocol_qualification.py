"""Build no-model protocol cases for new technical-cost-conserving adapter."""
from experiment_review import ROOT
def main():
    old=ROOT/'training/v159_protocol_synthetic_qualification_v2.py';new=old.with_name('v159_protocol_synthetic_qualification_v3.py');assert not new.exists()
    text=old.read_text(encoding='utf-8').replace('experiment_review_v159_v2','experiment_review_v159_v3').replace('v159_boundary_runtime_v3','v159_boundary_runtime_v4').replace('v159_boundary_train_v3','v159_boundary_train_v4').replace('v159_boundary_evaluate_v3','v159_boundary_evaluate_v4').replace("v159_protocol_synthetic_qualification_v2_20261002'","v159_protocol_synthetic_qualification_v3_20261002'")
    needle="    dummy=OUT/'synthetic_dependency.txt'"
    insert="""    p['historical_technical_cost']=dict(classifier_forward_chunks=124,opinion_feature_blocks=124,full_class_gradients=8,fits=0,updates=0)
    p['total_cumulative_caps']=dict(classifier_forward_chunks=78196,opinion_feature_blocks=78196,full_class_gradients=2420,fits=6)
    p['numeric_repeat_policy']=dict(module='training/v159_float64_repeat_policy_v2.py',repeat_eps=8,step_eps=16,argmax_exact=True,gradient_absolute_floor=0,Armijo_relaxation=False)
"""
    assert needle in text;text=text.replace(needle,insert+needle)
    needle="    cases['unsealed_rejected']="
    insert="""    for name,section,key,value in [('forgot_historical_cost_rejected','historical_technical_cost','classifier_forward_chunks',0),('silent_budget_reset_rejected','total_cumulative_caps','full_class_gradients',2412),('arbitrary_numeric_loosen_rejected','numeric_repeat_policy','repeat_eps',1000),('argmax_tolerance_rejected','numeric_repeat_policy','argmax_exact',False),('Armijo_relaxation_rejected','numeric_repeat_policy','Armijo_relaxation',True)]:
        bad=copy.deepcopy(p);bad[section][key]=value;cases[name]=not review_plan(bad)['plan_review_passed'];assert cases[name]
"""
    assert needle in text;text=text.replace(needle,insert+needle)
    text=text.replace("    assert require_checkpoint(p,r)['endpoint_review_passed'];", "    r['termination']='no_resolved_finite_step'\n    assert require_checkpoint(p,r)['endpoint_review_passed'];")
    new.write_text(text,encoding='utf-8');print('new protocol 20 cases source generated; no calls')
if __name__=='__main__':main()
