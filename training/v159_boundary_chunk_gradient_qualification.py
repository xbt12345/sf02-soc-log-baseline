"""Real entry risk accumulator on toy rows only, including empty-class chunks."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiment_review import ROOT,sha,check_bindings
import v159_boundary_train as entry
from v159_current_input_boundary_v3 import CurrentInputBoundary

OUT=ROOT/'artifacts/v159_boundary_chunk_gradient_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(4)
    paths=[Path(__file__).resolve(),ROOT/'training/v159_boundary_train.py',ROOT/'training/v159_boundary_runtime.py',ROOT/'training/v159_current_input_boundary_v3.py',ROOT/'training/v159_class_direction.py']
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    (OUT/'pre_synthetic_bindings.json').write_text(json.dumps(dict(status='bound_before_toy_entry_risk_function_and_autograd',source_sha256=bindings),indent=2)+'\n',encoding='utf-8')
    model=CurrentInputBoundary(width=2,hidden=2);rng=np.random.default_rng(15901)
    x=torch.tensor(rng.normal(size=(2049,2)),dtype=torch.float64);p=torch.tensor(np.broadcast_to([.05,.71,.24],(2049,16,3)).copy(),dtype=torch.float64)
    c=np.zeros((22546,3),np.int64);c[:2048,1]=np.arange(2048)%3+1;c[2048,2]=7
    ctx=dict(mass=c.sum(0),counts=c);ids=np.arange(2049)
    def toy_inputs(context,scope,ii):return x[ii],p[ii]
    counter=entry.Counter(model,4,OUT/'synthetic_entry_calls.jsonl',2);actual=[];risk_values=[]
    with patch.object(entry,'inputs',toy_inputs):
        for cls in [1,2]:
            risks,q,lp,g=entry.risk(model,ctx,'OOF',ids,counter,cls);actual.append(g);risk_values.append(risks)
    counts=counter.counts();counter.close()
    q,lp,_=model(x,p);reference=[]
    for cls in [1,2]:
        mass=torch.tensor(c[:2049,cls],dtype=torch.float64);loss=-(mass*lp[:,cls]).sum()/float(c[:,cls].sum())
        grads=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True);reference.append(torch.cat([g.flatten() for g in grads]).numpy())
    errors=[float(np.abs(a-b).max()) for a,b in zip(actual,reference)];assert max(errors)<1e-12
    expected=[float(-(torch.tensor(c[:2049,k],dtype=torch.float64)*lp[:,k]).sum()/c[:,k].sum()) for k in [1,2]]
    assert all(np.allclose(v,expected,atol=1e-12,rtol=0) for v in risk_values)
    assert counts==dict(head_attempts=4,head_completed=4,feature_attempts=4,feature_completed=4,gradient_attempts=2,gradient_completed=2)
    events=[json.loads(s) for s in (OUT/'synthetic_entry_calls.jsonl').read_text().splitlines()]
    assert all(e['original_class_mass']==ctx['mass'].tolist() for e in events if e['kind']=='full_class_gradient')
    check_bindings(bindings)
    result=dict(status='actual_entry_chunk_gradient_matches_full_original_mass_toy_reference',toy_rows=2049,toy_width=2,original_class_mass=ctx['mass'].tolist(),
        first_chunk_S_mass=0,last_chunk_M_mass=0,complete_class_gradient_max_errors=errors,stable_class_risks=expected,synthetic_counted_entry_work=counts,
        additional_reference_head_calls=1,additional_reference_full_parameter_class_gradients=2,
        official_classifier_calls=0,official_features_calls=0,official_gradients=0,official_fits=0,official_updates=0,
        formal_runtime_ready=False,official_zero_step_replay=False,quality_acceptance=False,source_sha256=bindings)
    (OUT/'qualification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
