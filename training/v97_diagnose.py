"""Open historic development labels ONLY after the frozen V selection receipt."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT, OUT, read, save, sha
from v89_common import data, raw_counts
from v85_protection import changes, cm_from_counts
from v81_training_contract import compare
from v97_prepare import DEST
from v97_select import class_table


def main():
    assert not (DEST/'diagnosis.json').exists()
    selection=read(DEST/'selection.json')
    assert selection['status']=='V_only_selection_frozen_before_historic_diagnosis'
    assert selection['source_sha256']==sha(ROOT/'training/v97_select.py')
    assert selection['actual_classifier_fits']==3 and selection['selected_arm'] is None or selection['selected_arm'] in ['ERM','MAG']
    for p,h in selection['artifact_bindings'].items():assert sha(ROOT/p)==h
    r,y,fid,z,old,sel,fit=data()
    roles=pd.read_parquet(ROOT/'artifacts/v95_cross_component_training_20260928/full_format_roles.parquet',columns=['role']).role.to_numpy()
    assert len(r)==len(y)==len(fid)==len(roles)
    teacher=np.load(DEST/'teacher_prediction.npy')
    trials={a:np.load(DEST/f'{a}_prediction.npy') for a in ['ERM','MAG']}
    isa=r.route.eq('asa').to_numpy()
    blended={'AB_teacher_ASA_plus_old_other':np.where(isa,teacher[fid],old[fid])}
    for arm,p in trials.items():blended[f'{arm}_ASA_plus_old_other']=np.where(isa,p[fid],old[fid])
    oldrows=old[fid]
    masks={'A':roles=='A','B':roles=='B','V':roles=='V',
        'inner':r.fold.eq(1).to_numpy(),'C':r.fold.eq(2).to_numpy(),
        'H':r.fold.eq(0).to_numpy(),'old_full_fit':fit}
    reports={};route_details=[]
    for role,mask in masks.items():
        original=y[mask];n=len(original)
        baseline=np.zeros((3,3),np.int64);np.add.at(baseline,(original,oldrows[mask]),1)
        report={'historic_v85':class_table(baseline)}
        for name,pred in blended.items():
            pm=pred[mask];cm=np.zeros((3,3),np.int64);np.add.at(cm,(original,pm),1)
            table=class_table(cm)
            table.update(negative_flips_vs_historic=int(((oldrows[mask]==original)&(pm!=original)).sum()),
                positive_flips_vs_historic=int(((oldrows[mask]!=original)&(pm==original)).sum()),
                guard_vs_historic=compare(baseline,cm,True))
            report[name]=table
        reports[role]=report
        if role in ['inner','C','H']:
            for route,g in r[mask].groupby('route'):
                ix=g.row_position.to_numpy();oldright=oldrows[ix]==y[ix]
                rec={'role':role,'route':route,'rows':len(ix),'historic_errors':int((~oldright).sum())}
                for name,pred in blended.items():
                    rec[name+'_errors']=int((pred[ix]!=y[ix]).sum())
                    rec[name+'_new_error_vs_old']=int((oldright&(pred[ix]!=y[ix])).sum())
                route_details.append(rec)
    save(DEST/'postselection_role_metrics.json',reports)
    pd.DataFrame(route_details).to_csv(DEST/'postselection_route_metrics.csv',index=False)
    icmp=pd.read_parquet(ROOT/'artifacts/v96_protocol_and_experiment_audit_20260928/icmp_rows.parquet')
    target=icmp[(icmp.role=='V')&(icmp.icmp_type==3)&(icmp.icmp_code==13)]
    assert target.groupby('label').size().to_dict()=={1:180,2:685}
    target_rows={}
    for label,g in target.groupby('label'):
        ix=g.row_position.to_numpy();rec={'support':len(ix),'components':int(g.component.nunique()),'R0_inputs':int(g.R0.nunique()),
            'historic_correct_exposed_to_V':int((oldrows[ix]==label).sum()),
            'AB_teacher_correct':int((teacher[fid[ix]]==label).sum())}
        for arm,p in trials.items():rec[arm+'_correct']=int((p[fid[ix]]==label).sum())
        target_rows[str(label)]=rec
    save(DEST/'type3_code13_readout.json',{'status':'postselection_readout_only','role':'V','slice':target_rows,
        'A_B_training_type3_code13_rows':0,'historic_reference_saw_V':True,
        'scope':'Existing labels and predictions only; no label-derived feature/rule or refit.'})
    vmask=masks['V']&isa&(y==1)
    regress=np.flatnonzero(vmask&(teacher[fid]==1)&(trials['MAG'][fid]!=1))
    facts=pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts
    mrows=[]
    for i in regress:
        mrows.append({'row_position':int(i),'component':int(r.component.iloc[i]),'R0':int(fid[i]),
            'projection_facts':json.loads(facts.iloc[r.projection_id.iloc[i]]),
            'old_prediction':int(old[fid[i]]),'teacher_prediction':int(teacher[fid[i]]),
            'ERM_prediction':int(trials['ERM'][fid[i]]),'MAG_prediction':int(trials['MAG'][fid[i]])})
    save(DEST/'MAG_V_new_M_errors.json',mrows)
    assert len(mrows)==4
    safety={}
    for name in ['ERM_ASA_plus_old_other','MAG_ASA_plus_old_other']:
        safety[name]={role:{'errors':reports[role][name]['errors'],
            'negative_flips_vs_historic':reports[role][name]['negative_flips_vs_historic'],
            'positive_flips_vs_historic':reports[role][name]['positive_flips_vs_historic'],
            'class_correct':reports[role][name]['correct'],
            'guard_passed':reports[role][name]['guard_vs_historic']['eligible']}
            for role in ['inner','C','H','old_full_fit']}
    any_pass=any(all(s['guard_passed'] and s['negative_flips_vs_historic']==0 for s in rec.values()) for rec in safety.values())
    result={'status':'postselection_historic_diagnosis_complete',
        'selection_sha256':sha(DEST/'selection.json'),'selected_arm':selection['selected_arm'],
        'historic_v85_safety':safety,'any_historic_safety_passed':any_pass,
        'quality_acceptance':False,'model_promoted':False,'new_full_task_replay':False,
        'classifier_fits_executed':3,'calibration_fits_executed':0,
        'new_MAG_V_M_errors':len(mrows),
        'type3_code13_V_M_correct_MAG':target_rows['1']['MAG_correct'],
        'type3_code13_V_S_correct_MAG':target_rows['2']['MAG_correct'],
        'source_sha256':sha(__file__),
        'scope':'Historical C/H/inner and fit are already inspected development; original V was selection. No blind or official score claim.'}
    save(DEST/'diagnosis.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
