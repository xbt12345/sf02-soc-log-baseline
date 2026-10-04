"""Actual V168 full correction/displacement tiny coordinates retained."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from experiment_review import ROOT,sha
from v169_structural_vector_storage import save_vector,load_vector

def main():
    out=ROOT/'artifacts/v169_saved_correction_storage_qualification_20261002';assert not out.exists();out.mkdir()
    xfile=ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz';x=load_npz(xfile);actual=np.unique(x.indices);cols=np.unique(np.r_[actual,np.arange(5)]);rows=[]
    source=[Path(__file__).resolve(),xfile,ROOT/'training/v169_structural_vector_storage.py']
    mask=np.ones(1060592,bool);mask[(actual[:,None]*16+np.arange(16)).ravel()]=False
    for name in ['correction','displacement']:
        path=ROOT/f'artifacts/v168_decision_floor_diagnostic_20261002/role1/correction0/{name}.npy';v=np.load(path);packed=out/f'{name}.npz';meta=save_vector(packed,v,cols)
        assert load_vector(packed,1060832).tobytes()==v.tobytes();off=np.flatnonzero(mask&(v[:1060592].view(np.uint64)!=0));assert len(off)==26 and np.array_equal(off,np.arange(26))
        rows.append(dict(name=name,tiny_off_actual_input_coordinates=off.tolist(),maximum_absolute=float(np.abs(v[off]).max()),full_bits_exact=True,packed=meta));source.append(path)
    report=dict(status='V169_actual_V168_correction_and_displacement_prefix_tiny_bits_qualified',vectors=rows,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source})
    (out/'qualification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],official_calls=0)))

if __name__=='__main__':main()
