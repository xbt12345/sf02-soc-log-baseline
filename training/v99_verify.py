"""Independently recompute saved grouping populations and preserve prior receipts."""
import json
import time
import numpy as np
import pandas as pd
from v89_common import ROOT,read,save,sha,data
from v99_normalization_feasibility import DEST,normalize


def main():
    assert not (ROOT/'evidence/2026-09-28/v99_research_plan/delivery.json').exists()
    started=time.monotonic();audit=read(DEST/'normalization_feasibility.json')
    assert audit['source_sha256']==sha(ROOT/'training/v99_normalization_feasibility.py')
    gradient=read(DEST/'objective_scale_audit.json')
    assert gradient['source_sha256']==sha(ROOT/'training/v99_objective_scale_audit.py')
    assert gradient['parameter_updates']==0
    assert abs(gradient['full_over_ASA']-753709/31289)<1e-12
    # The stored gradient vector is float32; rescaling it rounded once on device.
    assert np.isclose(gradient['total_gradient_inf_full_normalized']*gradient['full_over_ASA'],gradient['total_gradient_inf_equivalent_ASA_units'],rtol=2e-7,atol=1e-12)
    r,y,fid,_,_,_,_=data();row=np.flatnonzero(r.route.eq('asa').to_numpy())
    roles=pd.read_parquet(ROOT/'artifacts/v95_cross_component_training_20260928/full_format_roles.parquet',columns=['role']).role.to_numpy().astype(object)
    for name,fold in [('inner',1),('C',2),('H',0)]:roles[r.fold.eq(fold).to_numpy()]=name
    masks={'AB':np.isin(roles[row],['A','B']),'V':roles[row]=='V','inner':roles[row]=='inner','C':roles[row]=='C','H':roles[row]=='H','all':np.ones(len(row),bool)}
    groups={}
    for mode in audit['modes']:
        table=pd.read_parquet(DEST/f'{mode}_group_map.parquet')
        assert np.array_equal(table.row_position,row) and np.array_equal(table.old_R0,fid[row])
        for name,mask in masks.items():
            grouped=pd.crosstab(table.new_group[mask].to_numpy(),y[row][mask])
            floor=int((grouped.sum(axis=1)-grouped.max(axis=1)).sum())
            assert floor==audit['modes'][mode]['populations'][name]['empirical_conflict_floor']
            assert floor==audit['baseline'][name]['empirical_conflict_floor']
        assert read(DEST/f'{mode}_mixed_merge_groups.json')==[]
        groups[mode]=len(table)
    # Semantic regression sentinels: preserve complete ports and untouched message content.
    for mode in audit['modes']:
        for text in ['Deny tcp src outside:host/53158 dst dmz-2:host/9200','type 3, code 13; denied=false','command="echo  x"']:
            assert normalize(text,mode)==text
    for original in [' /1CRED- <IDENTITY> dst dmz-1: <IDENTITY> /6514',' /1 <IDENTITY> dst dmz-1: <IDENTITY> /6514']:
        assert normalize(original,'placeholder_cluster')==' /1 <IDENTITY> dst dmz-1: <IDENTITY> /6514'
    latest='evidence/2026-09-28/v98_boundary_audit/delivery_r2.json'
    previous=read(ROOT/latest);receipts=previous['verification']['receipt_sha256'].copy();receipts[latest]=sha(ROOT/latest)
    prior={}
    for p,h in receipts.items():
        assert sha(ROOT/p)==h,p
        for name,digest in read(ROOT/p)['artifact_sha256'].items():
            assert name not in prior or prior[name]==digest
            prior[name]=digest
    dirty=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not dirty
    result={'status':'passed','source_sha256':sha(__file__),'normalization_row_groups_recomputed':groups,
        'conflict_floor_by_role_recomputed':True,'known_numeric_fact_sentinels_preserved':True,
        'gradient_unit_conversion_verified':True,'prior_bound_files_rehashed':len(prior),
        'prior_bound_files_changed':dirty,'receipt_sha256':receipts,'actual_classifier_fits':0,'actual_calibration_fits':0,
        'quality_acceptance':False,'model_promoted':False,'elapsed_seconds':time.monotonic()-started,
        'scope':'No-fit candidate input collision/units checks and historical integrity. No training benefit, blind test, external security ground truth or full task replay.'}
    save(DEST/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='receipt_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
