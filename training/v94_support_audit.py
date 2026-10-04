"""Necessary support checks for a prospective cross-component training objective."""
import json,hashlib
import numpy as np
import pandas as pd
from v89_common import ROOT,DEST as V89,read,save,sha,data
from v94_mechanism_audit import DEST
from v93_regression_audit import FIELDS

def main():
    assert not (DEST/'episode_support.json').exists()
    r,y,fid,z,old,_,fit=data();a=r[fit&r.route.eq('asa').to_numpy()].copy();obs=np.load(V89/'row_fact_code.npy')
    dic=pd.read_parquet(V89/'row_fact_dictionary.parquet').observation_json;coarse=[]
    for text in dic:
        p=json.loads(text);k={n:v for n,v in p['facts'].items() if p['states'].get(n)=='known' and n in FIELDS};coarse.append(json.dumps(k,sort_keys=True))
    a['behavior']=np.array(coarse,object)[obs[a.index]]
    counts=a.groupby(['behavior','label_index']).size().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
    comps=a.groupby(['behavior','label_index']).component.nunique().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
    table=pd.DataFrame({'M_rows':counts[1],'S_rows':counts[2],'M_components':comps[1],'S_components':comps[2]})
    table['three_components_each_class']=(table.M_components>=3)&(table.S_components>=3)
    # Only inspect a prospective fixed partition; do not search seeds for better accuracy.
    def role(c):
        u=int(hashlib.sha256(('v94:9301:'+str(c)).encode()).hexdigest()[:16],16)%10
        return 'A' if u<6 else ('B' if u<8 else 'V')
    a['proposed_role']=a.component.map(role)
    support=a.groupby(['behavior','proposed_role','label_index']).component.nunique().unstack(['proposed_role','label_index'],fill_value=0)
    for ro in ['A','B','V']:
        for cl in [1,2]:table[f'{ro}_{cl}_components']=support.get((ro,cl),pd.Series(0,index=table.index))
    table['both_classes_in_A_B_V']=table[[f'{ro}_{cl}_components' for ro in ['A','B','V'] for cl in [1,2]]].gt(0).all(axis=1)
    table.reset_index().to_csv(DEST/'behavior_support.csv',index=False)
    a[['row_position','component','label_index','behavior','proposed_role']].to_parquet(DEST/'prospective_ASA_roles.parquet',index=False)
    summary={'scope':'Necessary support for coarse-behavior episode design, not causal domains, balanced data, split optimization or model training.',
        'new_classifier_fits':0,'new_calibration_fits':0,'source_sha256':sha(__file__),'seed':9301,'partition_rule':'sha256(v94:9301:component) first16 hex modulo10;0..5 A,6..7 B,8..9 V',
        'fit_ASA_rows':len(a),'behavior_groups':len(table),'groups_at_least_three_components_both_classes':int(table.three_components_each_class.sum()),
        'groups_both_classes_observed_in_A_B_V':int(table.both_classes_in_A_B_V.sum()),'class_coverage':{}}
    for c in [1,2]:
        b=a[a.label_index==c];summary['class_coverage'][str(c)]={'rows':len(b),'in_three_component_groups':int(b.behavior.isin(table.index[table.three_components_each_class]).sum()),
            'in_actual_A_B_V_both_class_groups':int(b.behavior.isin(table.index[table.both_classes_in_A_B_V]).sum()),'roles':b.proposed_role.value_counts().to_dict()}
    save(DEST/'episode_support.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
