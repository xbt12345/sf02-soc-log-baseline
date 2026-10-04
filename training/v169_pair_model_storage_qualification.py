"""Actual full-size CPU synthetic forwards/autograd; no official data calls."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment_review import ROOT,sha
from v159_current_input_boundary_v3 import CurrentInputBoundary
from v159_float64_repeat_policy_v2 import repeat_gradient as old_repeat,finite_step_review
from v160_margin_normal import measure
from v169_prior_pair_model import PriorPairBoundary,load_origin,width,validate_schema,gradient_repeat
from v169_exact_gradient_storage import save_vector,load_vector,digest
from v169_prior_pair_training_entry import strict_class_step,require

OUT=ROOT/'artifacts/v169_pair_model_storage_qualification_20261002'

def expect_error(call):
    try:call()
    except (ValueError,FloatingPointError,RuntimeError):return True
    raise AssertionError('Expected refusal')

def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.manual_seed(16901);original=CurrentInputBoundary();a,b=PriorPairBoundary('A'),PriorPairBoundary('B')
    with torch.no_grad():original.output_weight.copy_(torch.randn(16,3,dtype=torch.float64)*.03)
    state={k:v.clone() for k,v in original.state_dict().items()};load_origin(a,state);load_origin(b,state)
    assert sum(p.numel() for p in a.parameters())==1060832 and sum(p.numel() for p in b.parameters())==1060833
    assert validate_schema(b)[-1]==('beta',1060832,1060833)
    # Same actual complete observation width; only synthetic CSR/opinions.
    indptr=torch.tensor([0,211,422],dtype=torch.int64);indices=torch.tensor(list(range(211))+list(range(200,411)),dtype=torch.int64)
    data=torch.linspace(.1,.8,422,dtype=torch.float64);x=torch.sparse_csr_tensor(indptr,indices,data,size=(2,66287))
    p=torch.tensor([[[0.,1.-1e-12,1e-12]]*16,[[.2,.3,.5]]*16],dtype=torch.float64)
    counts=dict(synthetic_model_heads=0,synthetic_opinion_features=0,synthetic_complete_derivatives=0)
    hooks=[]
    for model in [original,a,b]:
        hooks.append(model.register_forward_pre_hook(lambda *args:counts.__setitem__('synthetic_model_heads',counts['synthetic_model_heads']+1)))
        hooks.append(model.opinions.register_forward_pre_hook(lambda *args:counts.__setitem__('synthetic_opinion_features',counts['synthetic_opinion_features']+1)))
    output=[m(x,p) for m in [original,a,b]]
    assert all(torch.equal(output[0][j],o[j]) for o in output[1:] for j in range(3))
    gradients={}
    for arm,model in [('A',a),('B',b)]:
        pairs=[]
        for repetition in range(2):
            model.zero_grad(set_to_none=True);q,lp,_=model(x,p);(-lp[:,2].mean()).backward();counts['synthetic_complete_derivatives']+=1
            assert all(v.grad is not None for v in model.parameters());g=np.concatenate([v.grad.numpy().ravel() for v in model.parameters()]);pairs.append(g)
            meta=save_vector(OUT/f'{arm}_class_repeat{repetition}.npz',g);assert load_vector(OUT/f'{arm}_class_repeat{repetition}.npz',width(arm)).tobytes()==g.tobytes()
        assert gradient_repeat(*pairs,arm)['passed'];gradients[arm]=pairs[0]
    assert np.array_equal(gradients['A'],gradients['B'][:-1]) and gradients['B'][-1]!=0
    assert gradient_repeat(gradients['A'],gradients['A'],'A')==old_repeat(gradients['A'],gradients['A'])
    altered=gradients['B'].copy();altered[-1]*=-1.;assert not gradient_repeat(gradients['B'],altered,'B')['passed']
    altered=gradients['B'].copy();altered[-1]=np.nextafter(altered[-1],np.inf)
    assert gradient_repeat(gradients['B'],altered,'B')['passed']
    assert not gradient_repeat(gradients['A'],gradients['A'],'B')['passed']
    sparse_bound=[]
    for arm,model in [('A',a),('B',b)]:
        value=measure(model,x,p,0,2,1);counts['synthetic_complete_derivatives']+=1
        assert value['gradient'].shape==(width(arm),)
        if arm=='B':assert abs(value['gradient'][-1]-float(p.mean(1).clamp_min(1e-12).log()[0,2]-p.mean(1).clamp_min(1e-12).log()[0,1]))<1e-13
        nonzero=np.flatnonzero(value['gradient'].view(np.uint64)!=0)
        assert len(nonzero)<=211*16+241
        sparse_bound.append(dict(arm=arm,raw_nonzero_bit_patterns=len(nonzero),bound=211*16+241))
        save_vector(OUT/f'{arm}_margin.npz',value['gradient'])
    # Negative zero, smallest subnormal, exact zero, huge/tiny finite values.
    special=np.zeros(1060833,np.float64);special[:6]=[-0.,np.nextafter(0.,1.),np.nextafter(0.,-1.),1e-280,-1e280,.25]
    save_vector(OUT/'signed_zero_subnormal.npz',special);assert load_vector(OUT/'signed_zero_subnormal.npz',1060833).tobytes()==special.tobytes()
    expect_error(lambda:load_vector(OUT/'signed_zero_subnormal.npz',1060832))
    # A may not inherit the old mean-loss gate: mean descends, S worsens.
    before=[1.,1.];after=[.9,1.01];slopes=[-1.,-1.]
    assert finite_step_review(before,after,slopes,1000,1,'A',1.,True)['accepted']
    for arm in ['A','B']:assert not strict_class_step(before,after,slopes,[1000,1],True)['accepted']
    assert strict_class_step(before,[.9,.9],slopes,[1000,1],True)['accepted']
    assert not strict_class_step(before,[.9,.9],slopes,[1000,1],False)['accepted']
    # Full scalar rollback includes beta; no clipping under/overflow.
    initial={k:v.clone() for k,v in b.state_dict().items()}
    for beta in [1000.,-1000.]:
        try:
            with torch.no_grad():b.beta.fill_(beta)
            expect_error(lambda:b(x,p))
        finally:b.load_state_dict(initial)
        assert all(torch.equal(v,initial[k]) for k,v in b.state_dict().items())
    with torch.no_grad():b.beta.fill_(-1.)
    changed=b(x,p);assert not torch.equal(changed[0],output[0][0]) and torch.equal(changed[2],output[0][2]);b.load_state_dict(initial)
    expect_error(require)
    for hook in hooks:hook.remove()
    files=[Path(__file__).resolve(),*[ROOT/'training'/f for f in ['v169_prior_pair_model.py','v169_exact_gradient_storage.py','v169_prior_pair_training_entry.py','v169_dynamic_trial_restoration.py','v169_working_joint_restoration.py']]]
    report=dict(status='V169_complete_CPU_synthetic_model_parameter_repeat_strict_objective_and_lossless_storage_qualified',
      complete_parameters=dict(A=1060832,B=1060833),zero_beta_original_A_B_all_q_logq_delta_bit_exact=True,
      beta_autograd_nonzero_and_own_segment_checked=True,all_base_derivative_components_exact_equal_at_beta0=True,
      no_negative_zero_or_subnormal_loss=True,margin_sparse_support_bound_observed=sparse_bound,
      A_control_cannot_use_old_mean_risk_gate=True,beta_overflow_and_underflow_rejected_without_clipping=True,
      full_scalar_state_rollback_exact=True,unsealed_official_entry_refused=True,counts=counts,
      official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,
      actual_official_zero_step_still_unexecuted=True,execution_authority=False,
      source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    (OUT/'qualification.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status=report['status'],counts=counts,official_calls=0)))

if __name__=='__main__':main()
