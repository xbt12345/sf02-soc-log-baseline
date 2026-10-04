"""Full model/max batch synthetic CUDA buffer measurement, no SOC forwards."""
import ctypes,json
from pathlib import Path
import numpy as np
import torch
from experiment_review import ROOT,sha
from v169_prior_pair_model import PriorPairBoundary,width,gradient_repeat
from v160_margin_normal import measure
from v169_factorized_resource_budget_v3 import memory

OUT=ROOT/'artifacts/v169_cuda_host_buffer_qualification_20261002'

class ProcessMemory(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in ['peak_working','working','peak_paged_pool','paged_pool','peak_nonpaged_pool','nonpaged_pool','pagefile','peak_pagefile']]

def process_memory():
    current=ctypes.windll.kernel32.GetCurrentProcess;current.restype=ctypes.c_void_p
    query=ctypes.windll.psapi.GetProcessMemoryInfo;query.argtypes=[ctypes.c_void_p,ctypes.POINTER(ProcessMemory),ctypes.c_ulong]
    value=ProcessMemory();value.cb=ctypes.sizeof(value)
    if not query(current(),ctypes.byref(value),value.cb):raise OSError('Actual process memory unavailable')
    return {name:int(getattr(value,name)) for name in ['peak_working','working','pagefile','peak_pagefile']}

def main():
    assert not OUT.exists();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    imported=dict(process=process_memory(),free_RAM_bytes=memory());torch.cuda.mem_get_info();torch.cuda.synchronize();initialized=dict(process=process_memory(),free_RAM_bytes=memory())
    # Same full observation width, maximum registered batch and row support.
    batch=2048;indptr=torch.arange(0,(batch+1)*211,211,dtype=torch.int64,device='cuda');indices=torch.arange(211,dtype=torch.int64,device='cuda').repeat(batch);data=torch.linspace(.1,.8,211,dtype=torch.float64,device='cuda').repeat(batch)
    x=torch.sparse_csr_tensor(indptr,indices,data,size=(batch,66287),device='cuda');p=torch.tensor([.2,.3,.5],dtype=torch.float64,device='cuda').repeat(batch,16,1)
    records=[];counts=dict(synthetic_heads=0,synthetic_features=0,synthetic_complete_derivatives=0)
    for arm in ['A','B']:
        torch.cuda.reset_peak_memory_stats();model=PriorPairBoundary(arm).cuda()
        with torch.no_grad():model.output_weight.copy_(torch.randn_like(model.output_weight)*.03)
        hooks=[model.register_forward_pre_hook(lambda *a:counts.__setitem__('synthetic_heads',counts['synthetic_heads']+1)),model.opinions.register_forward_pre_hook(lambda *a:counts.__setitem__('synthetic_features',counts['synthetic_features']+1))]
        for cls in [1,2]:
            pairs=[]
            for repetition in range(2):
                model.zero_grad(set_to_none=True);q,lp,_=model(x,p);(-lp[:,cls].mean()).backward();counts['synthetic_complete_derivatives']+=1
                g=np.concatenate([v.grad.detach().cpu().numpy().ravel() for v in model.parameters()]);assert g.shape==(width(arm),);pairs.append(g)
            assert gradient_repeat(*pairs,arm)['passed'];del pairs,g
        pairs=[]
        for repetition in range(2):value=measure(model,x,p,0,2,1);counts['synthetic_complete_derivatives']+=1;pairs.append(value)
        assert gradient_repeat(pairs[0]['gradient'],pairs[1]['gradient'],arm)['passed'];torch.cuda.synchronize()
        records.append(dict(arm=arm,complete_parameters=width(arm),max_batch=batch,row_nnz=211,process=process_memory(),free_RAM_bytes=memory(),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        for h in hooks:h.remove()
        del pairs,value,q,lp,model;torch.cuda.empty_cache()
    assert counts==dict(synthetic_heads=12,synthetic_features=12,synthetic_complete_derivatives=12)
    # The import baseline is already unavailable when require checks free RAM.
    # Do not subtract it twice; record complete peaks and incremental deltas.
    final=process_memory();report=dict(status='V169_complete_A_B_max_batch_CUDA_synthetic_host_and_device_buffers_measured',imported=imported,CUDA_initialized=initialized,cases=records,final_process=final,peak_commit_increment_after_import=max(0,final['peak_pagefile']-imported['process']['pagefile']),peak_resident_increment_after_import=max(0,final['peak_working']-imported['process']['working']),counts=counts,all_model_inputs_synthetic=True,full_parameter_widths_preserved=True,maximum_registered_batch_and_row_support_used=True,no_official_SOC_forward_or_gradient=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,not_full_CUDA_training_fit_peak=True,RAM_registration_unchanged=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),ROOT/'training/v169_prior_pair_model.py',ROOT/'training/v160_margin_normal.py',ROOT/'training/v169_factorized_resource_budget_v3.py']})
    OUT.mkdir();(OUT/'qualification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],counts=counts,peak_host_increment=report['peak_commit_increment_after_import'],GPU_max_reserved=max(r['GPU_peak_reserved_bytes'] for r in records),official_calls=0)))

if __name__=='__main__':main()
