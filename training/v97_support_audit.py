"""Postselection V errors versus A+B observed behavior and exact-input support."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT, OUT, read, save, sha
from v89_common import data, raw_counts
from v96_behavior_contract import observed_behavior
from v97_prepare import DEST


def main():
    assert not (DEST/'support_audit.json').exists()
    assert read(DEST/'selection.json')['status']=='V_only_selection_frozen_before_historic_diagnosis'
    r,y,fid,_,old,_,fit=data()
    roles=pd.read_parquet(ROOT/'artifacts/v95_cross_component_training_20260928/full_format_roles.parquet',columns=['role']).role.to_numpy()
    facts=[json.loads(s) for s in pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts]
    keys=[observed_behavior(f)[0] for f in facts]
    asa=r.route.eq('asa').to_numpy();ab=asa&np.isin(roles,['A','B']);v=asa&(roles=='V')
    train=r.loc[ab,['row_position','component','projection_id','label_index']].copy()
    train['behavior']=[keys[i] for i in train.projection_id]
    support=train[train.behavior.notna()].groupby(['behavior','label_index']).agg(
        rows=('row_position','size'),components=('component','nunique')).reset_index()
    support.to_csv(DEST/'AB_behavior_class_support.csv',index=False)
    tmap={(t.behavior,t.label_index):(int(t.rows),int(t.components)) for t in support.itertuples()}
    full=raw_counts(fid,y,ab,int(fid.max())+1)
    V=r.loc[v,['row_position','component','projection_id','label_index']].copy()
    V['R0']=fid[V.row_position.to_numpy()]
    V['behavior']=[keys[i] for i in V.projection_id]
    for cls in [1,2]:
        V[f'AB_class{cls}_same_behavior_rows']=[tmap.get((k,cls),(0,0))[0] for k in V.behavior]
        V[f'AB_class{cls}_same_behavior_components']=[tmap.get((k,cls),(0,0))[1] for k in V.behavior]
        V[f'AB_class{cls}_same_R0_rows']=full[V.R0.to_numpy(),cls]
    teacher=np.load(DEST/'teacher_prediction.npy');erm=np.load(DEST/'ERM_prediction.npy');mag=np.load(DEST/'MAG_prediction.npy')
    for name,pred in [('teacher',teacher),('ERM',erm),('MAG',mag)]:
        V[name+'_correct']=pred[V.R0.to_numpy()]==V.label_index.to_numpy()
    V.to_parquet(DEST/'V_ASA_row_support.parquet',index=False)
    def group(x):
        out={'rows':len(x),'components':int(x.component.nunique()),
            'distinct_R0':int(x.R0.nunique()),
            'known_behavior_rows':int(x.behavior.notna().sum()),
            'same_label_AB_behavior_supported_rows':int((x.apply(lambda z:z[f'AB_class{int(z.label_index)}_same_behavior_rows']>0,axis=1)).sum()),
            'same_label_AB_exact_R0_supported_rows':int((x.apply(lambda z:z[f'AB_class{int(z.label_index)}_same_R0_rows']>0,axis=1)).sum()),
            'AB_both_M_S_behavior_supported_rows':int(((x.AB_class1_same_behavior_components>0)&
                                                       (x.AB_class2_same_behavior_components>0)).sum())}
        return out
    failed=V[~V.MAG_correct];M=failed[failed.label_index==1];S=failed[failed.label_index==2]
    port=V[V.R0.eq(2809)&V.label_index.eq(1)]
    assert len(port)==4 and len(M)==4
    assert int(port.AB_class1_same_behavior_rows.max())==int(port.AB_class2_same_behavior_rows.max())==0
    icmp=pd.read_parquet(ROOT/'artifacts/v96_protocol_and_experiment_audit_20260928/icmp_rows.parquet')
    rows=icmp[(icmp.role=='V')&(icmp.icmp_type==3)&(icmp.icmp_code==13)&(icmp.label==2)].row_position
    target=V[V.row_position.isin(rows)]
    assert len(target)==685 and not target.MAG_correct.any()
    assert int(target.AB_class2_same_behavior_rows.max())==0
    report={'status':'postselection_support_audit','new_classifier_fits':0,
        'V_ASA_rows':len(V),'MAG_errors':len(failed),'MAG_M_errors':group(M),'MAG_S_errors':group(S),
        'four_new_M_one_component_port5046':group(port),'type3_code13_685_S':group(target),
        'source_sha256':sha(__file__),
        'scope':'Training support describes observed facts only; absent exact behavior labels do not prove prediction impossible. V already inspected.'}
    save(DEST/'support_audit.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
