"""Measure only all registered, previously unmeasured V165 blockers at origin."""
import numpy as np
import torch
from v159_boundary_train_v4 import inputs,tensor_hash
from v159_float64_repeat_policy_v2 import repeat_values,repeat_gradient
from v160_margin_normal import input_identity,measure
from v161_fixed_error_endpoint_diagnostic_v2 import save

MAX_NORMALS=25

def add_observed_functions(model,ctx,base,blockers,records,normals,counter,folder,expected_fresh,origin_logs,origin_probabilities):
    pending={};origin=tensor_hash(model.state_dict());assert origin==counter.parameter_point
    for (scope,local,truth,rival),group in blockers.groupby(['scope','local','truth','rival'],sort=True):
        ids=ctx['ids'] if scope=='OOF' else np.arange(22546);pos=int(np.searchsorted(ids,local));assert ids[pos]==local;start=pos//2048*2048;chunk=ids[start:start+2048];query=pos-start
        identity=input_identity(ctx['fold'],scope,chunk,ctx['x'][chunk],np.asarray(ctx[scope][chunk],np.float64),query,int(truth),int(rival))
        assert identity not in records,'Current V165 observed blockers must all be new functions'
        pending[identity]=(scope,chunk,query,int(truth),int(rival),group)
    assert len(pending)==expected_fresh and len(records)+len(pending)<=MAX_NORMALS
    for identity,(scope,chunk,query,truth,rival,group) in pending.items():
        target=folder/identity;target.mkdir(parents=True);group.to_parquet(target/'blocking_original_rows.parquet',index=False);np.save(target/'chunk_local_ids.npy',chunk)
        meta=dict(input_identity=identity,role=ctx['fold'],scope=scope,query=query,truth=truth,rival=rival,local=int(chunk[query]),base_parameter_sha256=origin,head_chunk_matches_protection_scope=True);save(target/'input_binding.json',meta);results=[]
        for repetition in range(2):
            assert tensor_hash(model.state_dict())==origin;counter.margin_before(identity);result=measure(model,*inputs(ctx,scope,chunk),query,truth,rival);counter.margin_after(identity)
            for key,value in result.items():np.save(target/f'repeat{repetition}_{key}.npy',value)
            results.append(result)
        review=dict(gradient=repeat_gradient(results[0]['gradient'],results[1]['gradient']),q=repeat_values(results[0]['q'],results[1]['q'],'probability'),logq=repeat_values(results[0]['logq'],results[1]['logq'],'log_probability'),margin=repeat_values([results[0]['margin']],[results[1]['margin']],'margin'),same_origin_q=repeat_values(results[0]['q'],origin_probabilities[scope][chunk],'probability'),same_origin_logq=repeat_values(results[0]['logq'],origin_logs[scope][chunk],'log_probability'),same_origin_margin=repeat_values([results[0]['margin']],[origin_logs[scope][chunk[query],truth]-origin_logs[scope][chunk[query],rival]],'margin'))
        save(target/'measurement_repeat_review.json',review);assert all(v['passed'] for v in review.values())
        assert all(torch.equal(p,b) for p,b in zip(model.parameters(),base)) and tensor_hash(model.state_dict())==origin
        records[identity]=meta;normals.append(results[0]['gradient'])
    return dict(fresh_function_identities=list(pending),fresh_functions=len(pending),joint_functions=len(records),new_complete_margin_gradients=2*len(pending),parameter_sha256=origin)
