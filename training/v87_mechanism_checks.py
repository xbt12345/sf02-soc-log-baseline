"""CPU theorem-sized checks plus real architecture/device and full-P preflight."""
import gc
import json
import unittest
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from run_v75 import read, save, sha
from test_v87_solver import SolverChecks
from v87_execute import DEST, OLD, data, Protection, SEED
from v87_solver import MarginJacobian
from v85_protection import Residual, csr_tensor


def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SolverChecks)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful()
    r,y,fid,x,cc,full,vc,fit,inner,selected=data()
    used=np.flatnonzero(cc.sum(1));ids=used[[0,len(used)//2,len(used)-1]]
    model=Residual().cuda()
    model.load_state_dict(torch.load(OLD/'fold1/C_epoch060.pt',map_location='cuda',weights_only=True))
    true=np.array([0,1,2]);other=np.array([1,2,0])
    j=MarginJacobian(model,x[ids],true,other)
    z=model(csr_tensor(x[ids]));gradients=[]
    for i in range(3):
        g=torch.autograd.grad(z[i,true[i]]-z[i,other[i]],tuple(model.parameters()),retain_graph=True)
        gradients.append(torch.cat([v.flatten() for v in g]))
    dense=torch.stack(gradients).double();actual=j.gram();expected=(dense@dense.T).cpu().numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-5)
    weights=np.array([.4,-.2,.8])
    factored=torch.cat([v.flatten() for v in j.transpose(weights)])
    direct=(torch.as_tensor(weights,device='cuda')@dense).float()
    torch.testing.assert_close(factored,direct,atol=2e-7,rtol=2e-4)
    maxdiff=float((factored-direct).abs().max())
    del j,z,dense,factored,direct,gradients,model;gc.collect();torch.cuda.empty_cache()
    z0=np.load(OLD/'fold1/teacher_scores.npy',mmap_mode='r');old=np.load(OLD/'fold1/teacher_prediction.npy')
    torch.cuda.reset_peak_memory_stats()
    p=Protection(x,z0,old,DEST/'fold1');torch.manual_seed(SEED);model=Residual().cuda()
    check,_,_=p.evaluate(model)
    assert check['protected_negative_flips']==check['violating_input_groups']==0
    out={'passed':True,'CPU_mathematical_tests':result.testsRun,'real_architecture_CUDA_jacobian_inputs':len(ids),
         'factorized_transpose_max_abs_difference':maxdiff,'full_P_preflight':check,
         'protected_unique_inputs':len(p.ids),'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),
         'source_sha256':sha(__file__),'solver_source_sha256':sha(__import__('v87_solver').__file__),
         'new_classifier_fits':0,'scope':'No optimization of new classifier. Mechanism arithmetic and memory only; no quality acceptance.',
         'pre_registration_repairs':[{'issue':'Sparse CSR .T uses unsupported as_strided in torch 2.7.1 CPU',
                                     'repair':'Use explicit two-dimensional transpose; then real CUDA checked.'}]}
    save(DEST/'mechanism_tests.json',out);print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=4):main()
