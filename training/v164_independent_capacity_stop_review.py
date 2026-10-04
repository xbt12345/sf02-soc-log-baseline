"""Rebuild final capped function sets from saved real input identities, CPU only."""
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from v160_independent_fixed_diagnostic_review import read, sha
from v160_margin_normal import input_identity

ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
OUT=ROOT/'artifacts/v164_independent_capacity_stop_review_20261002'


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    assert not OUT.exists()
    OUT.mkdir()
    xfile=ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    sources={Path(__file__).resolve(),ROOT/'training/v160_margin_normal.py',
             ROOT/'training/v160_fixed_endpoint_diagnostic_v3.py',xfile}
    prepared=[]
    for role in range(3):
        folder=TRIAL/f'role{role}'
        fit=read(folder/'fit.json')
        assert fit['exception'] is None and fit['status']=='margin_normal_cap_stop'
        point=folder/f"parameter_point{fit['permanent_updates']}"
        restorations=sorted(point.glob('restoration*/finite_probe/probe.json'),
                            key=lambda p:int(p.parent.parent.name.replace('restoration','')))
        assert restorations
        last=restorations[-1].parent
        files=[folder/'fit.json',point/'parameter_identity.json',
               folder/'baseline/OOF_original_rows.parquet',last/'probe.json',
               last/'actual_blocking_original_rows.parquet']
        known_paths=list((point/'normals').glob('*/input_binding.json'))
        opaths={s:ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{s}_probabilities.npy'
                for s in ['OOF','deployment']}
        sources.update(files+known_paths+list(opaths.values()))
        prepared.append((role,folder,point,last,known_paths,opaths))
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)}
    save(OUT/'source_bindings.json',dict(source_sha256=bindings,official_calls=0))
    x=load_npz(xfile).tocsr();x.sort_indices()
    results=[]
    for role,folder,point,last,known_paths,opaths in prepared:
        origin=read(point/'parameter_identity.json')['parameter_sha256']
        known={p.parent.name:read(p) for p in known_paths}
        assert all(m['input_identity']==identity and m['base_parameter_sha256']==origin
                   for identity,m in known.items())
        ids0=np.sort(pd.read_parquet(folder/'baseline/OOF_original_rows.parquet').local.unique())
        opinions={s:np.asarray(np.load(p,mmap_mode='r')[:,:16],np.float64) for s,p in opaths.items()}
        blockers=pd.read_parquet(last/'actual_blocking_original_rows.parquet')
        functions={}
        for (scope,local,truth,rival),group in blockers.groupby(['scope','local','truth','rival'],sort=True):
            ids=ids0 if scope=='OOF' else np.arange(22546)
            pos=int(np.searchsorted(ids,local));assert ids[pos]==local
            chunk=ids[pos//2048*2048:pos//2048*2048+2048];query=pos%2048
            identity=input_identity(role,scope,chunk,x[chunk],opinions[scope][chunk],query,int(truth),int(rival))
            functions[identity]=dict(scope=scope,local=int(local),truth=int(truth),rival=int(rival),original_rows=len(group))
        fresh=set(functions)-set(known)
        combined=len(known)+len(fresh)
        assert combined>24 and all(not (point/'normals'/identity).exists() for identity in fresh)
        result=dict(role=role,parameter_point=point.relative_to(ROOT).as_posix(),
                    last_actual_proposal=last.relative_to(ROOT).as_posix(),blocking_original_rows=len(blockers),
                    known_measured_functions=len(known),last_blocking_functions=len(functions),
                    fresh_unmeasured_functions=len(fresh),combined_functions=combined,registered_cap=24,
                    cap_exceeded_before_new_derivatives=True,no_local_or_global_infeasibility_claim=True)
        save(OUT/f'role{role}_function_sets.json',dict(review=result,known=known,last_blocking=functions,
                                                     fresh_unmeasured=sorted(fresh)))
        results.append(result)
    assert all(sha(ROOT/p)==value for p,value in bindings.items())
    report=dict(status='all_three_actual_capacity_stops_reconstructed_from_saved_complete_function_inputs',
                roles=results,official_calls=0,parameter_updates=0,
                scope='Recorded capacity exhaustion, not an infeasibility or classification-capacity certificate.')
    save(OUT/'review.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),
                                   traceback=traceback.format_exc(),official_calls=0))
        raise
