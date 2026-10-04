"""Full 64-function old/new byte-exact solver proof and live memory peaks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import json,gc
from pathlib import Path
import numpy as np
from experiment_review import ROOT,sha
import v169_working_joint_restoration as old
import v169_working_joint_restoration_v2 as new
from v169_cuda_host_buffer_qualification_v2 import process_memory

OUT=ROOT/'artifacts/v169_memory_exact_solver_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();cases=[]
    for arm,width in [('A',1060832),('B',1060833)]:
        u=np.zeros(width);u[:2]=1.;m=np.zeros(width);s=np.zeros(width);m[0]=-1.;s[1]=-1.
        if arm=='B':m[-1]=.1;s[-1]=.2
        a=np.zeros((64,width));a[np.arange(64),np.arange(2,66)]=1.;b=np.full(64,.2);c=np.full(64,-.1)
        # Signed zero cases preserve the exact original negation/division bits.
        a[0,100]=-0.;m[100]=-0.;s[101]=-0.
        scale=np.max(np.abs(np.vstack([a,-m,-s])),axis=1);expected=np.vstack([a,-m,-s])/scale[:,None]
        actual=np.empty_like(expected);np.divide(a,scale[:-2,None],out=actual[:-2]);np.divide(-m,scale[-2],out=actual[-2]);np.divide(-s,scale[-1],out=actual[-1]);assert actual.tobytes()==expected.tobytes();del actual,expected;gc.collect()
        before=process_memory();first=old.propose(u,a,b,c,m,s);old_peak=process_memory();second=new.propose(u,a,b,c,m,s);new_peak=process_memory()
        assert first['status']==second['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard'
        assert first['displacement'].tobytes()==second['displacement'].tobytes() and first['correction'].tobytes()==second['correction'].tobytes()
        assert {k:v for k,v in first.items() if k not in ['displacement','correction']}=={k:v for k,v in second.items() if k not in ['displacement','correction']}
        cases.append(dict(arm=arm,complete_parameters=width,current_functions=64,SVD_input_bits_exact=True,full_displacement_and_correction_bits_exact=True,all_original_unit_metadata_exact=True,one_full_matrix_copy_removed_bytes=66*width*8,before=before,after_old=old_peak,after_new=new_peak,shared_process_peak_is_not_separate_new_peak=True))
        del u,m,s,a,b,c,first,second;gc.collect()
    source=[Path(__file__).resolve(),Path(old.__file__),Path(new.__file__),ROOT/'training/v169_dynamic_trial_restoration_v2.py']
    report=dict(status='V169_same_full_SVD_inputs_full_displacement_and_original_unit_certificates_byte_exact_with_one_matrix_copy_removed',cases=cases,actual_independent_CPU_QP_calls=4,scientific_algorithm_and_dimensions_unchanged=True,no_coordinate_or_normal_drop=True,signed_zero_SVD_input_preserved=True,separate_new_actual_memory_peak_still_requires_measurement=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source})
    (OUT/'qualification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],full_QPs=4,official_calls=0)))

if __name__=='__main__':main()
