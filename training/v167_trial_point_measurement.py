"""All covered full-width margin Jacobians at the actual temporary trial."""
import numpy as np
import torch
from v159_boundary_train_v4 import inputs,tensor_hash
from v159_float64_repeat_policy_v2 import repeat_values,repeat_gradient
from v160_margin_normal import input_identity,measure
from v161_fixed_error_endpoint_diagnostic_v2 import save

def measure_all(model,ctx,records,counter,folder,origin,trial,stage,origin_logs,trial_logs,trial_q):
    assert origin!=trial and tensor_hash(model.state_dict())==trial and 0<len(records)<=25;counter.set_trial_point(trial,stage);normals=[];new_records={};old_state=tuple(p.detach().clone() for p in model.parameters())
    for identity,old in records.items():
        scope,local,truth,rival=[old[k] for k in ['scope','local','truth','rival']];ids=ctx['ids'] if scope=='OOF' else np.arange(22546);pos=int(np.searchsorted(ids,local));assert ids[pos]==local;chunk=ids[pos//2048*2048:pos//2048*2048+2048];query=pos%2048
        assert input_identity(ctx['fold'],scope,chunk,ctx['x'][chunk],np.asarray(ctx[scope][chunk],np.float64),query,truth,rival)==identity;assert old['base_parameter_sha256']==origin
        target=folder/identity;target.mkdir(parents=True);np.save(target/'chunk_local_ids.npy',chunk);meta=dict(input_identity=identity,role=ctx['fold'],scope=scope,local=local,query=query,truth=truth,rival=rival,base_parameter_sha256=trial,origin_parameter_sha256=origin,linearization_parameter_sha256=trial,correction_stage=stage,head_chunk_matches_protection_scope=True);save(target/'input_binding.json',meta);observed=[]
        for repetition in range(2):
            assert tensor_hash(model.state_dict())==trial;counter.margin_before(identity);result=measure(model,*inputs(ctx,scope,chunk),query,truth,rival);counter.margin_after(identity)
            for key,value in result.items():np.save(target/f'repeat{repetition}_{key}.npy',value)
            observed.append(result)
        first,second=observed;repeat=dict(gradient=repeat_gradient(first['gradient'],second['gradient']),q=repeat_values(first['q'],second['q'],'probability'),logq=repeat_values(first['logq'],second['logq'],'log_probability'),margin=repeat_values([first['margin']],[second['margin']],'margin'),same_trial_q=repeat_values(first['q'],trial_q[scope][chunk],'probability'),same_trial_logq=repeat_values(first['logq'],trial_logs[scope][chunk],'log_probability'),same_trial_margin=repeat_values([first['margin']],[trial_logs[scope][local,truth]-trial_logs[scope][local,rival]],'margin'));save(target/'measurement_repeat_review.json',repeat);assert all(v['passed'] for v in repeat.values())
        assert tensor_hash(model.state_dict())==trial and all(torch.equal(p,v) for p,v in zip(model.parameters(),old_state));new_records[identity]=meta;normals.append(first['gradient'])
    assert list(new_records)==list(records);save(folder/'trial_point_identity.json',dict(origin_parameter_sha256=origin,linearization_parameter_sha256=trial,stage=stage,full_functions=len(records),complete_margin_derivatives=2*len(records),all_original_functions_retained=True));return new_records,normals

def blocker_identities(ctx,blockers):
    result=set()
    for (scope,local,truth,rival),group in blockers.groupby(['scope','local','truth','rival'],sort=True):
        ids=ctx['ids'] if scope=='OOF' else np.arange(22546);pos=int(np.searchsorted(ids,local));assert ids[pos]==local;chunk=ids[pos//2048*2048:pos//2048*2048+2048]
        result.add(input_identity(ctx['fold'],scope,chunk,ctx['x'][chunk],np.asarray(ctx[scope][chunk],np.float64),pos%2048,int(truth),int(rival)))
    return result
