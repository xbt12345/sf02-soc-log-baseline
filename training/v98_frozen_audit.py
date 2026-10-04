"""Audit evidence, teacher margins and population gates without fitting any parameter."""
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from v89_common import ROOT, OUT, data, read, save, sha
from v92_train import Branch, csr
from v96_behavior_contract import observed_behavior
from v75_views import view
from v75_corrective import stable

DEST=ROOT/'artifacts/v98_boundary_and_gate_audit_20260928'
V97=ROOT/'artifacts/v97_matched_teacher_20260928'
V92=ROOT/'artifacts/v92_evidence_training_20260928'


def main():
    assert not DEST.exists();DEST.mkdir()
    delivery=read(ROOT/'evidence/2026-09-28/v97_matched_training/delivery.json')
    assert delivery['actual_classifier_fits']==3 and delivery['selected_arm'] is None
    r,y,fid,_,old,_,fit=data()
    roles=pd.read_parquet(ROOT/'artifacts/v95_cross_component_training_20260928/full_format_roles.parquet',columns=['role']).role.to_numpy()
    isa=r.route.eq('asa').to_numpy();ab=isa&np.isin(roles,['A','B']);v=isa&(roles=='V')
    facts=[json.loads(s) for s in pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts]
    keys=[observed_behavior(f)[0] for f in facts]
    rowkeys=np.array([keys[p] for p in r.projection_id],dtype=object)
    ids=np.load(V92/'ASA_input_ids.npy');x=sparse.load_npz(V92/'ASA_R0.npz')
    lookup=np.full(int(fid.max())+1,-1,np.int32);lookup[ids]=np.arange(len(ids))
    teacher=np.load(V97/'teacher_prediction.npy');tz=np.load(V97/'teacher_scores.npy')[ids]
    logits={'teacher':tz};predictions={'teacher':teacher}
    for arm in ['ERM','MAG']:
        state=torch.load(V97/f'{arm}_model.pt',map_location='cpu',weights_only=True)
        net=Branch(x.shape[1]);net.load_state_dict(state['state_dict']);net.eval();zz=np.empty_like(tz)
        with torch.no_grad():
            for start in range(0,len(ids),2048):
                end=min(start+2048,len(ids));zz[start:end]=tz[start:end]+net(csr(x[start:end],'cpu')).double().numpy()
        logits[arm]=zz;predictions[arm]=np.load(V97/f'{arm}_prediction.npy')
        assert np.array_equal(zz.argmax(1),predictions[arm][ids])
    np.savez_compressed(DEST/'frozen_ASA_scores.npz',ASA_ids=ids,teacher=tz,ERM=logits['ERM'],MAG=logits['MAG'])
    V=pd.read_parquet(V97/'V_ASA_row_support.parquet')
    supported=V[(V.label_index==2)&(~V.MAG_correct)&(V.AB_class2_same_behavior_rows>0)]
    assert len(supported)==12
    records=[];representatives={};group_records=[]
    def row_record(i,reason):
        return {'row_position':int(i),'reason':reason,'role':str(roles[i]),'label':int(y[i]),
            'component':int(r.component.iloc[i]),'R0':int(fid[i]),'facts':facts[r.projection_id.iloc[i]],
            'teacher_MS_margin':float(tz[lookup[fid[i]],2]-tz[lookup[fid[i]],1]),
            'MAG_MS_margin':float(logits['MAG'][lookup[fid[i]],2]-logits['MAG'][lookup[fid[i]],1])}
    for behavior,g in supported.groupby('behavior'):
        train=np.flatnonzero(ab&(rowkeys==behavior))
        for cls in [1,2]:
            ix=train[y[train]==cls]
            rec={'behavior':behavior,'class':cls,'AB_rows':len(ix),'AB_components':int(r.component.iloc[ix].nunique()),
                 'V_failed_S_rows':len(g),'V_failed_S_components':int(g.component.nunique())}
            for name,p in predictions.items():rec[name+'_AB_correct']=int((p[fid[ix]]==cls).sum())
            group_records.append(rec)
            for comp,sub in r.iloc[ix].groupby('component'):
                if cls==2 or len(ix)<=40:
                    i=int(sub.row_position.iloc[0]);representatives[i]=row_record(i,'AB_same_behavior_support')
        for _,sub in g.groupby('R0'):
            i=int(sub.row_position.iloc[0]);rec=row_record(i,'V_supported_but_wrong_S');rec['repeated_rows']=len(sub)
            records.append(rec);representatives[i]=rec
    pd.DataFrame(group_records).to_csv(DEST/'supported_behavior_fit_and_transfer.csv',index=False)
    save(DEST/'twelve_supported_S_score_trace.json',records)
    vi=np.flatnonzero(v);local=lookup[fid[vi]];truth=y[vi]
    tt=tz[local];mm=logits['MAG'][local];delta=mm-tt
    teacher_pred=tt.argmax(1);new_pred=mm.argmax(1)
    repaired=(teacher_pred!=truth)&(new_pred==truth)
    lost=(teacher_pred==truth)&(new_pred!=truth)
    crossings=[]
    for label,mask in [('S_repaired',(truth==2)&repaired),('M_lost',(truth==1)&lost)]:
        for f in np.unique(fid[vi[mask]]):
            q=np.flatnonzero(mask&(fid[vi]==f));j=int(q[0]);den=delta[j,2]-delta[j,1]
            crossing=-(tt[j,2]-tt[j,1])/den if abs(den)>1e-14 else None
            i=int(vi[j]);rec=row_record(i,label);rec.update(rows=len(q),crossing_scale_M_S=crossing,
                residual_MS_change=float(den),R0=int(f))
            crossings.append(rec)
            if label=='M_lost':representatives[i]=rec
    save(DEST/'gain_loss_crossing_scales.json',crossings)
    frontier=[]
    for scale in [0.,.1,.25,.5,.75,1.]:
        pred=(tt+scale*delta).argmax(1)
        frontier.append({'scale':scale,'errors':int((pred!=truth).sum()),
            'M_correct':int(((truth==1)&(pred==1)).sum()),'S_correct':int(((truth==2)&(pred==2)).sum()),
            'repairs_vs_teacher':int(((teacher_pred!=truth)&(pred==truth)).sum()),
            'new_errors_vs_teacher':int(((teacher_pred==truth)&(pred!=truth)).sum())})
    save(DEST/'frozen_residual_scale_diagnostic.json',{'status':'retrospective_sensitivity_only_no_scale_selected',
        'frontier':frontier,'scope':'V labels already observed. No coefficient fit, candidate promotion or independent gain claim.'})
    # Gate exposure: the historical fit set includes V, which the historic teacher fitted.
    gp={};op=old[fid];bp=teacher[fid];mp=predictions['MAG'][fid]
    for role,mask in [('A',roles=='A'),('B',roles=='B'),('V',roles=='V'),('inner',r.fold.eq(1).to_numpy()),
                      ('C',r.fold.eq(2).to_numpy()),('H',r.fold.eq(0).to_numpy())]:
        o=op==y;b=bp==y;m=mp==y
        take=mask&isa
        gp[role]={'ASA_rows':int(take.sum()),'historic_wrong':int((take&~o).sum()),
            'teacher_wrong':int((take&~b).sum()),'MAG_wrong':int((take&~m).sum()),
            'historical_correct_final_wrong':int((take&o&~m).sum()),
            'historical_correct_teacher_lost_still_wrong':int((take&o&~b&~m).sum()),
            'historical_correct_teacher_kept_branch_lost':int((take&o&b&~m).sum()),
            'historical_correct_teacher_lost_branch_restored':int((take&o&~b&m).sum())}
    save(DEST/'teacher_branch_exposure_attribution.json',gp)
    # Trace concentrated ICMP classes too; their exact type/code already exists in model facts.
    icmp=pd.read_parquet(ROOT/'artifacts/v96_protocol_and_experiment_audit_20260928/icmp_rows.parquet')
    target=icmp[(icmp.role=='V')&(icmp.icmp_type==3)&(icmp.icmp_code==13)]
    for (cl,f),g in target.groupby(['label','R0']):
        i=int(g.row_position.iloc[0]);rec=row_record(i,'V_ICMP_type3_code13');rec['repeated_rows']=len(g);representatives[i]=rec
    raw={};offset=0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=32768,columns=['message_sanitized'],use_threads=False):
        positions=[i for i in representatives if offset<=i<offset+batch.num_rows]
        if positions:
            strings=batch.column(0).to_pylist()
            for i in positions:
                text=strings[i-offset];rec=representatives[i]
                rec['raw_sha256']=hashlib.sha256(text.encode()).hexdigest()
                rec['normalized_message']=stable(view(text)[0]);raw[i]=rec
        offset+=batch.num_rows
    assert len(raw)==len(representatives)
    save(DEST/'raw_normalized_evidence.json',list(raw.values()))
    result={'status':'frozen_boundary_support_and_exposure_audited','source_sha256':sha(__file__),
        'classifier_fits':0,'calibration_fits':0,'supported_S_rows':12,
        'supported_S_distinct_behaviors':int(supported.behavior.nunique()),
        'each_supported_S_behavior_has_one_S_training_component':all(g['AB_components']==1 for g in group_records if g['class']==2),
        'AB_supported_S_rows':sum(g['AB_rows'] for g in group_records if g['class']==2),
        'AB_supported_S_MAG_correct':sum(g['MAG_AB_correct'] for g in group_records if g['class']==2),
        'MAG_old_fit_negative_flips_by_role':{k:gp[k]['historical_correct_final_wrong'] for k in 'ABV'},
        'V_old_correct_teacher_lost_still_wrong':gp['V']['historical_correct_teacher_lost_still_wrong'],
        'V_old_correct_teacher_kept_branch_lost':gp['V']['historical_correct_teacher_kept_branch_lost'],
        'source_representatives':len(raw),'scope':'Retrospective original-source/score diagnostics, no label-derived rule or new model claim.'}
    save(DEST/'audit.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
