"""Actual saved full vectors/rows and adversarial bit-exact reconstruction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from experiment_review import ROOT,read,sha
from v169_structural_vector_storage import save_vector,load_vector,reconstruction,digest
from v169_factorized_row_evidence import save_scope,load_scope

OUT=ROOT/'artifacts/v169_factorized_storage_qualification_v2_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();xfile=ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz';x=load_npz(xfile);actual_columns=np.unique(x.indices);assert len(actual_columns)==412;global_columns=np.unique(np.r_[actual_columns,np.arange(5)])
    files=[Path(__file__).resolve(),xfile,ROOT/'training/v169_structural_vector_storage.py',ROOT/'training/v169_factorized_row_evidence.py',ROOT/'training/v169_prior_pair_training_entry_v5.py'];vectors=[];rows=[]
    for role in [0,1,2]:
        targetfile=ROOT/f'artifacts/v161_independent_frozen_error_cohort_review_20261002/role{role}/target_original_counts.npy';counts=np.load(targetfile);files.append(targetfile)
        state=read(ROOT/f'artifacts/v164_short_supervised_trajectory_20261002/role{role}/fit.json')['permanent_updates']
        for cls in [1,2]:
            file=ROOT/f'artifacts/v164_short_supervised_trajectory_20261002/role{role}/parameter_point{state}/class{cls}_repeat0/complete_fixed_error_target_gradient.npy';g=np.load(file);cols=np.unique(x[np.flatnonzero(counts[:,cls])].indices)
            target=OUT/f'role{role}_class{cls}.npz';meta=save_vector(target,g,cols);assert load_vector(target,1060832).tobytes()==g.tobytes();vectors.append(dict(role=role,class_id=cls,**meta));files.append(file)
        safe=ROOT/f'artifacts/v167_trial_point_restoration_diagnostic_20261002/role{role}/probe0' if role!=1 else ROOT/'artifacts/v168_decision_floor_diagnostic_20261002/role1/probe0'
        ufile=safe/'direction.npy';u=np.load(ufile);stored=OUT/f'role{role}_safe_displacement.npz';meta=save_vector(stored,u,global_columns);assert load_vector(stored,1060832).tobytes()==u.tobytes()
        scaled=u*2.**-3;recipe=dict(kind='scaled',source=stored.relative_to(ROOT).as_posix(),source_sha256=sha(stored),step_hex=float(2.**-3).hex(),width=1060832,dense_sha256=digest(scaled));assert reconstruction(recipe,ROOT).tobytes()==scaled.tobytes()
        corrected=scaled+u;recipe2=dict(kind='added',parent=recipe,correction=stored.relative_to(ROOT).as_posix(),correction_sha256=sha(stored),width=1060832,dense_sha256=digest(corrected));assert reconstruction(recipe2,ROOT).tobytes()==corrected.tobytes()
        bad=dict(recipe,dense_sha256='0'*64)
        try:reconstruction(bad,ROOT)
        except ValueError:pass
        else:raise AssertionError('Wrong recipe reconstruction digest must refuse')
        for scope in ['OOF','deployment']:
            reference=ROOT/f'artifacts/v164_short_supervised_trajectory_20261002/role{role}/endpoint/{scope}_original_rows.parquet';actual=safe/f'{scope}_original_rows.parquet';rr=pd.read_parquet(actual);qfile,lfile=safe/f'{scope}_q.npy',safe/f'{scope}_logq.npy';q,lp=np.load(qfile),np.load(lfile)
            folder=OUT/f'role{role}_{scope}';spec=save_scope(folder,ROOT,reference,q,lp,rr.protected_correct.to_numpy(bool));rebuilt=load_scope(folder,ROOT)
            pd.testing.assert_frame_equal(rebuilt[rr.columns],rr,check_exact=True)
            rows.append(dict(role=role,scope=scope,original_rows=len(rr),all_dataframe_values_exact=True,reference_bound=True,payload_bound=True));files.extend([reference,actual,qfile,lfile])
        files.append(ufile)
    # Explicit arbitrary sign zeros/subnormals inside and uniform -zero outside.
    v=np.full(1060833,-0.,np.float64);v[0]=np.nextafter(0.,1.);v[1]=np.nextafter(0.,-1.);v[2]=0.;v[-1]=-.375
    path=OUT/'negative_zero_subnormal_B.npz';meta=save_vector(path,v,np.array([0]));assert load_vector(path,1060833).tobytes()==v.tobytes() and meta['default_zero_bits']==str(1<<63)
    # Mixed off-support zero signs are not discarded; dense fallback is kept
    # and the storage bound rejected. Tiny nonzero outside also cannot vanish.
    failures=[]
    for name,value in [('mixed_off_support_negative_zero',-0.),('off_support_smallest_subnormal',np.nextafter(0.,1.))]:
        v=np.zeros(1060833,np.float64);v[1000]=value;path=OUT/f'{name}.npz'
        try:save_vector(path,v,np.array([0]))
        except RuntimeError:pass
        else:raise AssertionError('Off-support bit violation must preserve-and-refuse')
        assert load_vector(path,1060833).tobytes()==v.tobytes();failures.append(name)
    report=dict(status='V169_saved_full_vectors_signed_zero_subnormals_recipes_and_all_original_rows_bit_exact_qualified',global_observation_columns=412,encoded_SVD_prefix_observation_columns=5,old_tiny_off_input_coordinates_preserved=True,full_parameter_dimensions_retained=True,vectors=vectors,full_original_row_reconstruction=rows,negative_zero_subnormal_B_exact=True,wrong_reconstruction_hash_refused=True,off_support_violation_keeps_actual_full_vector_and_refuses=failures,lossless_factorization_not_model_coordinate_reduction=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,new_official_runtime_support_qualification_still_required=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    (OUT/'qualification.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status=report['status'],global_observation_columns=412,full_row_scopes=6,official_calls=0)))

if __name__=='__main__':main()
