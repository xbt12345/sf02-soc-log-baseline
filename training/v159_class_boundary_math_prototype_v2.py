"""Synthetic-only complete-input/current-opinion class-boundary prototype.

No official arrays, classifiers, features, gradients, or training entry are
executed. Constructed parameter witnesses are not fitted parameters.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,softmax
from experiment_review import ROOT,sha,check_bindings

COUNTS=dict(synthetic_boundary_evaluations=0,synthetic_class_risk_evaluations=0,synthetic_analytic_class_gradient_evaluations=0)
OUT=ROOT/'artifacts/v159_class_boundary_math_v2_20261001'

def opinion_features(logits):
    if logits.ndim!=3 or logits.shape[1:]!=(16,3) or not np.isfinite(logits).all():raise ValueError('Exactly16 finite current NN logits required')
    lp=logits-logsumexp(logits,axis=-1,keepdims=True);p=softmax(logits,axis=-1)
    entropy=-(p*lp).sum(-1,keepdims=True)
    common=np.concatenate([p.mean(1),p.var(1),entropy.mean(1)],axis=1)
    return np.concatenate([p,np.broadcast_to(common[:,None,:],(len(p),16,7)),entropy],axis=-1)

def boundary(x,logits,observation_weight,opinion_weight,bias,output_weight):
    COUNTS['synthetic_boundary_evaluations']+=1
    feat=opinion_features(logits);obs=np.asarray(x@observation_weight)
    if obs.shape!=(len(logits),len(bias)) or opinion_weight.shape!=(11,len(bias)) or output_weight.shape!=(len(bias),3):raise ValueError('Head shapes changed')
    hidden=np.tanh(obs[:,None,:]+feat@opinion_weight+bias)
    delta=(hidden@output_weight).mean(1)
    shifted=logits+delta[:,None,:]
    # The origin is the mean of member probabilities, never the probability
    # of mean logits. All three outputs, including N, are retained.
    q=softmax(shifted,axis=-1).mean(1)
    lp=shifted-logsumexp(shifted,axis=-1,keepdims=True)
    logq=logsumexp(lp,axis=1)-np.log(16.)
    assert np.isfinite(q).all() and np.isfinite(logq).all()
    return q,logq,delta

def class_risk_and_delta_gradient(logits,delta,original_counts,cls):
    COUNTS['synthetic_class_risk_evaluations']+=1;COUNTS['synthetic_analytic_class_gradient_evaluations']+=1
    mass=original_counts[:,cls].astype(np.float64);total=mass.sum()
    if total<=0:raise ValueError('Absent class has no risk or descent certificate')
    z=logits+delta[:,None,:];lp=z-logsumexp(z,axis=-1,keepdims=True);p=softmax(z,axis=-1)
    normalizer=logsumexp(lp[:,:,cls],axis=1)
    logq=normalizer-np.log(16.)
    responsibility=np.exp(lp[:,:,cls]-normalizer[:,None])
    per_row=(responsibility[:,:,None]*p).sum(1);per_row[:,cls]-=1
    return float(-(mass@logq)/total),per_row*(mass/total)[:,None]

def main():
    assert not OUT.exists();OUT.mkdir()
    sources=[Path(__file__).resolve(),ROOT/'training/v159_class_boundary_math_prototype.py',ROOT/'artifacts/v159_class_boundary_math_20261001/failure.json',ROOT/'training/v128_train_r3.py',ROOT/'training/v127_model.py',ROOT/'training/v110_layer_probes.py',
        ROOT/'training/v85_protection.py',ROOT/'training/v87_solver.py',ROOT/'training/v159_mgda_synthetic_qualification.py',
        ROOT/'artifacts/v159_mgda_synthetic_qualification_20261001/qualification.json',
        ROOT/'artifacts/v159_independent_OOF_capacity_and_input_review_v2_20261001/audit.json',
        ROOT/'artifacts/v159_current_expert_capacity_review_20261001/audit.json']
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    (OUT/'pre_synthetic_bindings.json').write_text(json.dumps(dict(status='sources_bound_before_synthetic_function_evaluation',source_sha256=bindings),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rng=np.random.default_rng(15901);x=np.array([[-1.,0.],[1.,0.]])
    member=np.empty((16,3));member[:,0]=.05;member[:,1]=np.linspace(.62,.80,16);member[:,2]=.95-member[:,1]
    logits=np.broadcast_to(np.log(member)[None,:,:],(2,16,3)).copy()
    w_x=rng.normal(size=(2,2));w_p=rng.normal(size=(11,2));bias=rng.normal(size=2);zero=np.zeros((2,3))
    base=softmax(logits,axis=-1).mean(1)
    origin,_,delta=boundary(x,logits,w_x,w_p,bias,zero)
    assert np.array_equal(origin,base) and np.array_equal(delta,np.zeros((2,3)))
    assert origin.argmax(1).tolist()==[1,1]
    # Constructed witness: flip the distinguishable S input while preserving
    # the distinguishable M control. No fitting or search selects this state.
    witness_out=np.array([[0.,-3.,3.],[0.,0.,0.]])
    q,_,_=boundary(x,logits,np.eye(2),np.zeros((11,2)),np.zeros(2),witness_out)
    assert q.argmax(1).tolist()==[1,2] and q[1,2]>member[:,2].max()
    perm=np.array([7,4,0,15,3,8,1,2,6,9,10,5,11,14,12,13])
    permq,_,_=boundary(x,logits[:,perm,:],np.eye(2),np.zeros((11,2)),np.zeros(2),witness_out)
    permutation_gap=float(np.abs(q-permq).max());assert permutation_gap<1e-14
    # Show expert opinions can change the learned correction with the same x.
    opinion_w=np.zeros((11,2));opinion_w[0,0]=1.;out=np.array([[0.,0.,1.],[0.,0.,0.]])
    _,_,d1=boundary(np.zeros_like(x),logits,np.zeros((2,2)),opinion_w,np.zeros(2),out)
    changed=logits.copy();changed[:,0,:]=np.log([.2,.6,.2])
    _,_,d2=boundary(np.zeros_like(x),changed,np.zeros((2,2)),opinion_w,np.zeros(2),out)
    assert not np.array_equal(d1,d2)
    counts=np.array([[0,9,0],[0,0,2]],np.float64);test_delta=np.array([[.13,-.07,.19],[-.17,.11,.08]])
    max_fd=0.
    for cls in [1,2]:
        value,g=class_risk_and_delta_gradient(logits,test_delta,counts,cls)
        for row in range(2):
            for col in range(3):
                change=np.zeros_like(test_delta);change[row,col]=1e-5
                pos=class_risk_and_delta_gradient(logits,test_delta+change,counts,cls)[0]
                neg=class_risk_and_delta_gradient(logits,test_delta-change,counts,cls)[0]
                max_fd=max(max_fd,abs((pos-neg)/2e-5-g[row,col]))
    assert max_fd<1e-9
    # Stable original-frequency ensemble probability CE, not member CE.
    heterogeneous=np.log(np.array([[[.05,.94,.01]]*8+[[.05,.15,.80]]*8]))
    cc=np.array([[0,0,5]],np.float64);loss,_=class_risk_and_delta_gradient(heterogeneous,np.zeros((1,3)),cc,2)
    expected=-np.log(softmax(heterogeneous,axis=-1).mean(1)[0,2]);member_ce=float(-(heterogeneous[:,:,2]).mean())
    assert abs(loss-expected)<1e-12 and abs(loss-member_ce)>.1
    # Mean-logit substitution changes the origin and must be rejected.
    wrong_origin=softmax(heterogeneous.mean(1),axis=-1);true_origin=softmax(heterogeneous,axis=-1).mean(1)
    wrong_origin_gap=float(np.abs(wrong_origin-true_origin).max());assert wrong_origin_gap>.05
    # The scalar probability can underflow while stable log-CE still gives a
    # finite, nonzero class derivative. Never clip the training loss to 690.
    huge=np.broadcast_to(np.array([0.,1000.,-1000.]),(1,16,3)).copy()
    huge_loss,huge_grad=class_risk_and_delta_gradient(huge,np.zeros((1,3)),cc,2)
    assert softmax(huge,axis=-1).mean(1)[0,2]==0 and abs(huge_loss-2000)<1e-12 and np.allclose(huge_grad,np.array([[0.,1.,-1.]]),atol=1e-12,rtol=0)
    missing_class_rejected=False
    try:class_risk_and_delta_gradient(logits,np.zeros((2,3)),counts,0)
    except ValueError:missing_class_rejected=True
    assert missing_class_rejected
    # Check the derivative through the zero output layer, with no conditional
    # zero correction branch, and a single functional common-descent probe.
    from v159_mgda_synthetic_qualification import exact_two_gradient_direction
    h0=np.tanh((x@w_x)[:,None,:]+opinion_features(logits)@w_p+bias).mean(1)
    initial_risks=[];head_gradients=[];head_fd=0.
    for cls in [1,2]:
        value,gd=class_risk_and_delta_gradient(logits,np.zeros((2,3)),counts,cls)
        initial_risks.append(value);g=h0.T@gd;head_gradients.append(g.ravel())
        assert np.linalg.norm(g)>0
        for row in range(2):
            for col in range(3):
                step=np.zeros((2,3));step[row,col]=1e-5
                _,lp1,_=boundary(x,logits,w_x,w_p,bias,step)
                _,lp2,_=boundary(x,logits,w_x,w_p,bias,-step)
                mass=counts[:,cls];numerical=-(mass@(lp1[:,cls]-lp2[:,cls]))/mass.sum()/2e-5
                head_fd=max(head_fd,abs(numerical-g[row,col]))
    assert head_fd<1e-9
    alpha,direction=exact_two_gradient_direction(*head_gradients)
    slopes=[float(g@direction) for g in head_gradients];assert max(slopes)<0
    probe=.01*direction.reshape(2,3)/np.linalg.norm(direction)
    probe_q,probe_logq,_=boundary(x,logits,w_x,w_p,bias,probe)
    probe_risks=[float(-(counts[:,cls]@probe_logq[:,cls])/counts[:,cls].sum()) for cls in [1,2]]
    assert all(after<before for after,before in zip(probe_risks,initial_risks)) and probe_q[0].argmax()==1
    check_bindings(bindings)
    result=dict(status='single_current_input_class_boundary_synthetic_function_qualified_not_official_runtime',
        official_layout=dict(current_input_columns=66287,current_text_columns=65792,current_facts=495,current_members=16,
            opinion_columns=11,hidden=16,outputs=3,parameters=66287*16+11*16+16+16*3),
        origin_mean_member_probability_exact_in_synthetic=True,legacy_expert_input=False,legacy_prior=False,
        witness_initial_predictions=origin.argmax(1).tolist(),witness_corrected_predictions=q.argmax(1).tolist(),
        witness_corrected_S_probability=float(q[1,2]),old_convex_S_max=float(member[:,2].max()),
        member_permutation_gap=permutation_gap,expert_opinion_affects_correction=True,
        mean_probability_CE_finite_difference_max_error=max_fd,mean_logit_wrong_origin_gap=wrong_origin_gap,
        underflow_probability_example=dict(stable_true_class_CE=huge_loss,analytic_delta_gradient=huge_grad.tolist()),
        missing_class_rejected=True,zero_head_gradient_finite_difference_max_error=head_fd,zero_head_class_gradient_norms=[float(np.linalg.norm(g)) for g in head_gradients],synthetic_common_descent_probe=dict(M_weight=alpha,class_slopes=slopes,initial_class_risks=initial_risks,probe_class_risks=probe_risks,old_correct_M_retained=True),synthetic_work=COUNTS,official_classifier_calls=0,official_features_calls=0,
        official_gradient_calls=0,official_fits=0,official_parameter_updates=0,official_model_or_data_evaluation=False,
        new_training_entry_registered=False,official_zero_step_replay=False,quality_acceptance=False,model_promoted=False,
        limits=['V128/V85 already allow free class residuals; escaping a convex hull alone is not a novel SOC mechanism.',
            'Synthetic existence and analytic delta gradient do not prove full parameter-gradient implementation or real-data learnability.',
            'Actual CUDA origin equality, all imported runtime dependencies, full input binding, class directions, guards and costs still need formal review and sealing.'],
        source_bindings_sha256=sha(OUT/'pre_synthetic_bindings.json'))
    (OUT/'qualification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
