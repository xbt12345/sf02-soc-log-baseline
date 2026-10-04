"""New wired entry versions; never mutate executed v3 trial/source/seal."""
from experiment_review import ROOT

def write(name,text):
    path=ROOT/'training'/name;assert not path.exists();path.write_text(text,encoding='utf-8')

def replace(text,old,new):
    assert old in text,old[:160];return text.replace(old,new)

def main():
    adapter=(ROOT/'training/experiment_review_v159_v2.py').read_text(encoding='utf-8')
    adapter=adapter.replace('full-OOF-replay-v3','full-OOF-replay-numeric-v4').replace('v159_boundary_train_v3','v159_boundary_train_v4').replace('v159_boundary_evaluate_v3','v159_boundary_evaluate_v4')
    adapter=replace(adapter("" ) if False else adapter,"'no_feasible_step'}","'no_feasible_step','no_resolved_common_descent_no_automatic_fallback','no_resolved_mean_descent_no_automatic_fallback','no_resolved_finite_step'}")
    needle="    if plan.get('shape',{}).get('parameters')"
    addition="""    carry=plan.get('historical_technical_cost',{})
    cumulative=plan.get('total_cumulative_caps',{})
    if carry.get('classifier_forward_chunks')!=124 or carry.get('opinion_feature_blocks')!=124 or carry.get('full_class_gradients')!=8 or carry.get('fits')!=0 or carry.get('updates')!=0:violations.append('historical_cost_not_conserved')
    if cumulative.get('classifier_forward_chunks')!=78196 or cumulative.get('opinion_feature_blocks')!=78196 or cumulative.get('full_class_gradients')!=2420 or cumulative.get('fits')!=6:violations.append('cumulative_technical_budget')
    policy=plan.get('numeric_repeat_policy',{})
    if policy.get('module')!='training/v159_float64_repeat_policy_v2.py' or policy.get('repeat_eps')!=8 or policy.get('step_eps')!=16 or not policy.get('argmax_exact') or policy.get('gradient_absolute_floor')!=0 or policy.get('Armijo_relaxation')!=False:violations.append('fixed_resolved_numeric_policy')
"""
    adapter=replace(adapter,needle,addition+needle);write('experiment_review_v159_v3.py',adapter)
    runtime=(ROOT/'training/v159_boundary_runtime_v3.py').read_text(encoding='utf-8').replace('experiment_review_v159_v2','experiment_review_v159_v3').replace("v159_class_boundary_trial_20261002'","v159_class_boundary_numeric_trial_20261002'").replace('v159_boundary_execution_contract_v3','v159_boundary_execution_contract_v4').replace('v159_boundary_train_v3','v159_boundary_train_v4').replace('v159_boundary_evaluate_v3','v159_boundary_evaluate_v4')
    runtime=replace(runtime,"    assert p['activation_entries']", "    assert p['total_cumulative_caps']['classifier_forward_chunks']==78196 and p['total_cumulative_caps']['full_class_gradients']==2420\n    assert p['activation_entries']")
    write('v159_boundary_runtime_v4.py',runtime)
    train=(ROOT/'training/v159_boundary_train_v3.py').read_text(encoding='utf-8').replace('v159_boundary_runtime_v3','v159_boundary_runtime_v4').replace('v159_boundary_evaluate_v3','v159_boundary_evaluate_v4')
    train=replace(train,'from v159_class_direction import class_direction,finite_armijo','from v159_float64_repeat_policy_v2 import resolved_class_direction as class_direction,finite_armijo,finite_step_review,repeat_values,repeat_gradient,repeat_direction')
    train=replace(train,"    # Warm only a separately counted dummy", "    assert tensor_hash(model.state_dict())==tensor_hash(torch.load(ROOT/'artifacts/v159_class_boundary_trial_20261002/initial.pt',map_location='cpu',weights_only=True)['state'])\n    # Warm only a separately counted dummy")
    train=replace(train,"            assert np.array_equal(oo[ctx['ids']],repeat[ctx['ids']]) and np.array_equal(de,dr)","            for name,array in [('OOF_probability',oo),('OOF_repeated_probability',repeat),('deployment_probability',de),('deployment_repeated_probability',dr)]:np.save(folder/(name+'.npy'),array)\n            assert repeat_values(oo[ctx['ids']],repeat[ctx['ids']],'probability')['passed'] and repeat_values(de,dr,'probability')['passed']")
    train=replace(train,"            if previous is not None:assert np.array_equal(oo[ctx['ids']],previous[0][ctx['ids']]) and np.array_equal(de,previous[1])","            if previous is not None:assert repeat_values(oo[ctx['ids']],previous[0][ctx['ids']],'probability')['passed'] and repeat_values(de,previous[1],'probability')['passed']")
    begin=train.index("                gs=[]\n                for repetition");end=train.index("                joint.append",begin)
    train=train[:begin]+'''                gs=[];observed=[]
                for repetition in range(2):
                    gg=[]
                    for cls in [1,2]:
                        riskv,q,lp,g=risk(model,ctx,'OOF',ctx['ids'],counter,cls)
                        prefix=folder/f'repetition{repetition}_class{cls}'
                        for name,array in [('risk',riskv),('probability',q),('log_probability',lp),('complete_gradient',g)]:np.save(str(prefix)+'_'+name+'.npy',array)
                        assert repeat_values(q[ctx['ids']],oo[ctx['ids']],'probability')['passed']
                        observed.append((riskv,q[ctx['ids']],lp[ctx['ids']]));gg.append(g)
                    gs.append(gg)
                numerical=[]
                for item in observed[1:]:numerical.append({key:repeat_values(observed[0][j],item[j],'probability' if j==1 else key) for j,key in enumerate(['risk','probability','log_probability'])})
                gradients=[repeat_gradient(gs[0][j],gs[1][j]) for j in [0,1]]
                directions=[repeat_direction(*gs,*ctx['mass'][1:],a) for a in ['A','B']]
                save(folder/'repeat_review.json',dict(same_point=numerical,gradient_comparisons=gradients,direction_comparisons=directions))
                assert all(r['passed'] for p in numerical for r in p.values()) and all(r['passed'] for r in gradients+directions)
                grad_report=dict(class_norms=[float(np.linalg.norm(g)) for g in gs[0]],repeated_complete_class_gradients_within_fixed_scale=True,repeat_review_sha256=sha(folder/'repeat_review.json'),hidden_initial_gradients_zero=all(np.count_nonzero(g[:-48])==0 for g in gs[0]),output_gradient_nonzero=all(np.linalg.norm(g[-48:])>0 for g in gs[0]))
                assert grad_report['hidden_initial_gradients_zero'] and grad_report['output_gradient_nonzero']
''' +train[end:]
    train=replace(train,"    assert np.array_equal(qo[ctx['ids']],np.load(OUT/f'preflight{f}_{arm}/OOF_probability.npy')[ctx['ids']]) and np.array_equal(de[ctx['ids']],np.load(OUT/f'preflight{f}_{arm}/deployment_probability.npy')[ctx['ids']])","    assert repeat_values(qo[ctx['ids']],np.load(OUT/f'preflight{f}_{arm}/OOF_probability.npy')[ctx['ids']],'probability')['passed'] and repeat_values(de[ctx['ids']],np.load(OUT/f'preflight{f}_{arm}/deployment_probability.npy')[ctx['ids']],'probability')['passed']")
    train=replace(train,'iterations+=1;gs=[];base_risks=None','iterations+=1;gs=[];base_risks=None;base_probability=None;base_log_probability=None;same_point_review=None')
    train=replace(train,"            if base_risks is not None:assert np.array_equal(value,base_risks)\n            base_risks=value","            if base_risks is not None:\n                same_point_review=dict(risk=repeat_values(value,base_risks,'risk'),probability=repeat_values(q[ctx['ids']],base_probability,'probability'),log_probability=repeat_values(lp[ctx['ids']],base_log_probability,'log_probability'))\n                assert all(r['passed'] for r in same_point_review.values())\n            else:base_risks=value;base_probability=q[ctx['ids']].copy();base_log_probability=lp[ctx['ids']].copy()")
    train=replace(train,"base_risks=base_risks.tolist(),class_norms=","base_risks=base_risks.tolist(),same_point_review=same_point_review,slope_resolution=direction['slope_resolution'],class_norms=")
    train=replace(train,"            ok=finite_armijo(base_risks,trial,direction['class_slopes'],*ctx['mass'][1:],arm,step,guard)","            finite_review=finite_step_review(base_risks,trial,direction['class_slopes'],*ctx['mass'][1:],arm,step,guard);ok=finite_review['accepted']")
    train=replace(train,"classification_guard=guard,OOF_stats=","classification_guard=guard,finite_numeric_review=finite_review,OOF_stats=")
    train=replace(train,"termination='proposal_budget' if proposals>=600 else 'no_feasible_step';break","termination='proposal_budget' if proposals>=600 else ('no_resolved_finite_step' if finite_review['reason']=='below_numeric_resolution' else 'no_feasible_step');break")
    write('v159_boundary_train_v4.py',train)
    evaluate=(ROOT/'training/v159_boundary_evaluate_v3.py').read_text(encoding='utf-8').replace('v159_boundary_runtime_v3','v159_boundary_runtime_v4').replace('v159_boundary_train_v3','v159_boundary_train_v4').replace('experiment_review_v159_v2','experiment_review_v159_v3')
    evaluate=replace(evaluate,'from v159_class_direction import finite_armijo','from v159_float64_repeat_policy_v2 import finite_armijo,repeat_values')
    evaluate=replace(evaluate,'assert gap<=2e-12',"assert repeat_values(actual[['p0','p1','p2']].to_numpy(),stored[['p0','p1','p2']].to_numpy(),'probability')['passed']")
    evaluate=replace(evaluate,'assert oof_gap<=2e-12',"assert repeat_values(actual_oo[['p0','p1','p2']].to_numpy(),oo[['p0','p1','p2']].to_numpy(),'probability')['passed']")
    evaluate=replace(evaluate,'assert log_gap<=1e-10',"assert repeat_values(actual_oo[['logp0','logp1','logp2']].to_numpy(),oo[['logp0','logp1','logp2']].to_numpy(),'log_probability')['passed']")
    evaluate=replace(evaluate,"np.allclose(ov,accepted_history['stable_class_risks'],atol=1e-10,rtol=0)","repeat_values(ov,accepted_history['stable_class_risks'],'risk')['passed']")
    evaluate=replace(evaluate,"pd.testing.assert_frame_equal(source,saved_source,check_exact=False,rtol=0,atol=1e-8)","assert np.array_equal(source[['root','truth','support','errors']].to_numpy(),saved_source[['root','truth','support','errors']].to_numpy())\n                assert repeat_values(source[['stable_CE_sum','probability_clip_CE_sum']].to_numpy(),saved_source[['stable_CE_sum','probability_clip_CE_sum']].to_numpy(),'source_CE_sum')['passed']")
    evaluate=replace(evaluate,"assert np.array_equal(q,np.load(folder/'endpoint_deployment_probability.npy'))","assert repeat_values(q,np.load(folder/'endpoint_deployment_probability.npy'),'probability')['passed']")
    evaluate=replace(evaluate,"assert np.array_equal(qo[ctx['ids']],np.load(folder/'endpoint_OOF_probability.npy')[ctx['ids']]) and np.allclose(value,r['final_stable_class_risks'],atol=1e-10,rtol=0)","assert repeat_values(qo[ctx['ids']],np.load(folder/'endpoint_OOF_probability.npy')[ctx['ids']],'probability')['passed'] and repeat_values(value,r['final_stable_class_risks'],'risk')['passed']")
    evaluate=replace(evaluate,"        validation_scope=", "        historical_technical_cost=plan['historical_technical_cost'],cumulative_actual_head_calls=124+pre['actual_head_forward_calls']+sum(r['counts']['head_attempts'] for r in receipts)+sum(c['head_attempts'] for c in evaluation_counts),cumulative_actual_full_class_gradients=8+pre['actual_full_class_gradients']+sum(r['full_class_gradients'] for r in receipts),total_cumulative_caps=plan['total_cumulative_caps'],\n        validation_scope=")
    write('v159_boundary_evaluate_v4.py',evaluate)
    print('new numeric v4 entries/adapter created; no official calls')
if __name__=='__main__':main()
