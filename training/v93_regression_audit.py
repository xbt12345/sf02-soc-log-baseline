"""Frozen v92 audit. No optimization, fitted gate, calibration or model promotion."""
import json,time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.special import softmax
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from v89_common import ROOT,OUT,OLD,read,save,sha,data,raw_counts,DEST as V89

PREV=ROOT/'artifacts/v92_evidence_training_20260928'
DEST=ROOT/'artifacts/v93_regression_audit_20260928'
FIELDS=['action','outcome','transport_protocol','src_role','dst_role','dst_port_fixed']

def quant(a):
    a=np.asarray(a)
    return {str(q):float(np.quantile(a,q)) for q in [0,.25,.5,.75,1]} if len(a) else {}

def forward(st,x,hidden=False):
    gelu=lambda a:.5*a*(1+np.tanh(np.sqrt(2/np.pi)*(a+.044715*a**3)))
    h=gelu(gelu(np.asarray(x@st['first.weight'].T))@st['second.weight'].T+st['second.bias'])
    return h if hidden else h@st['out.weight'].T+st['out.bias']

def counts_metrics(cc,old,pred):
    a=np.arange(len(cc));return {
        'errors':int(cc.sum()-cc[a,pred].sum()),
        'PF': [int(cc[(old!=c)&(pred==c),c].sum()) for c in range(3)],
        'NF': [int(cc[(old==c)&(pred!=c),c].sum()) for c in range(3)],
        'correct':[int(cc[pred==c,c].sum()) for c in range(3)]}

def main():
    start=time.monotonic();assert not DEST.exists();DEST.mkdir()
    save(DEST/'registration.json',{'scope':'Frozen model mechanical diagnosis only; all roles are observed development, no model or gate selection.',
        'classifier_fits':0,'calibration_fits':0,'selected':None,'source_sha256':sha(__file__),
        'methods':['Exact score/margin replay','All-row change traces and component/fact coverage','Exact scalar residual path frontier (oracle diagnostic only)',
            'Fixed coordinate-block/output-bias ablations (off-manifold sensitivity, not semantic intervention)',
            'Class-conditional nearest TRAINING input in R0 and learned representation (descriptive, not ground truth)'],
        'fixed_ablations':['remove_output_bias','zero_byte_block_in_residual','zero_fact_block_in_residual','zero_metadata_block_in_residual'],
        'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [PREV/'U0R_model.pt',PREV/'U0_model.pt',PREV/'ASA_R0.npz',PREV/'ASA_fit_counts.npy',PREV/'ASA_input_ids.npy',OLD/'teacher_scores.npy']}})
    r,y,fid,zall,oldall,sel,fit=data();ids=np.load(PREV/'ASA_input_ids.npy');x=sparse.load_npz(PREV/'ASA_R0.npz').astype(float)
    z=np.asarray(zall[ids]);old=z.argmax(1);s=np.load(PREV/'U0R_ASA_scores.npy');new=s.argmax(1);delta=s-z
    state=torch.load(PREV/'U0R_model.pt',map_location='cpu',weights_only=True)['state_dict'];st={k:v.numpy() for k,v in state.items()}
    replay=z+forward(st,x);assert np.allclose(s,replay,rtol=0,atol=1e-10)
    asa=r.route.eq('asa').to_numpy();ar=np.flatnonzero(asa);aid=np.searchsorted(ids,fid[ar]);assert np.array_equal(ids[aid],fid[ar])
    counts={role:raw_counts(fid,y,mask,len(oldall))[ids] for role,mask in [('fit',fit),('inner',r.fold.eq(1)),('C',r.fold.eq(2)),('H',r.fold.eq(0))]}
    assert np.array_equal(counts['fit'],np.load(PREV/'ASA_fit_counts.npy'))
    frame=r.iloc[ar].copy();frame['R0_id']=fid[ar];frame['asa_id']=aid;frame['role']=np.where(fit[ar],'fit',np.where(r.fold.iloc[ar].eq(1),'inner',np.where(r.fold.iloc[ar].eq(2),'C','H')))
    frame['old']=old[aid];frame['new']=new[aid]
    frame['kind']=np.where((frame.old==frame.label_index)&(frame.new!=frame.label_index),'regression',np.where((frame.old!=frame.label_index)&(frame.new==frame.label_index),'repair',np.where(frame.new==frame.label_index,'retained_correct','retained_error')))
    frame['old_M_minus_S']=z[aid,1]-z[aid,2];frame['residual_S_minus_M']=delta[aid,2]-delta[aid,1];frame['new_M_minus_S']=s[aid,1]-s[aid,2]
    frame['new_softmax_max_uncalibrated']=softmax(s,axis=1).max(1)[aid]
    frame['fit_same_R0_M']=counts['fit'][aid,1];frame['fit_same_R0_S']=counts['fit'][aid,2]
    # Known full facts and deliberately coarser behavior are different audit levels.
    obs=np.load(V89/'row_fact_code.npy');d=pd.read_parquet(V89/'row_fact_dictionary.parquet').observation_json
    known=[];coarse=[]
    for text in d:
        p=json.loads(text);k={a:b for a,b in p['facts'].items() if p['states'].get(a)=='known'}
        known.append(json.dumps(k,sort_keys=True));coarse.append(json.dumps({a:b for a,b in k.items() if a in FIELDS},sort_keys=True))
    for name,values in [('known',known),('coarse',coarse)]:
        frame[name]=np.asarray(values,dtype=object)[obs[ar]]
        fitframe=frame[frame.role=='fit']
        ct=fitframe.groupby([name,'label_index']).size().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
        comp=fitframe.groupby([name,'label_index']).component.nunique().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
        for c in [1,2]:
            frame[f'fit_{name}_{c}_rows']=frame[name].map(ct[c]).fillna(0).astype(int)
            frame[f'fit_{name}_{c}_components']=frame[name].map(comp[c]).fillna(0).astype(int)
    summaries=[]
    for (role,kind,cls),b in frame.groupby(['role','kind','label_index']):
        rec={'role':role,'kind':kind,'class':int(cls),'rows':len(b),'inputs':b.R0_id.nunique(),'components':b.component.nunique(),
            'old_M_minus_S':quant(b.old_M_minus_S),'residual_S_minus_M':quant(b.residual_S_minus_M),'new_M_minus_S':quant(b.new_M_minus_S),
            'uncalibrated_new_softmax_ge_99':int((b.new_softmax_max_uncalibrated>=.99).sum()),
            'same_R0_seen_fit':int(((b.fit_same_R0_M+b.fit_same_R0_S)>0).sum())}
        for level in ['known','coarse']:
            m=b[f'fit_{level}_1_rows'];ss=b[f'fit_{level}_2_rows']
            rec[level+'_support']={'none':int(((m==0)&(ss==0)).sum()),'M_only':int(((m>0)&(ss==0)).sum()),'S_only':int(((m==0)&(ss>0)).sum()),'both':int(((m>0)&(ss>0)).sum())}
        summaries.append(rec)
    save(DEST/'population_summary.json',summaries)
    # Follow the entire linear residual segment exactly, with every decision crossing.
    changed=np.flatnonzero(old!=new);assert set(old[changed])<=set([1,2]) and set(new[changed])<=set([1,2])
    tau=(z[changed,1]-z[changed,2])/(delta[changed,2]-delta[changed,1]);assert np.all((tau>0)&(tau<1))
    at=z[changed]+tau[:,None]*delta[changed];assert np.all(at[:,0]<np.minimum(at[:,1],at[:,2])),'B wins inside segment; must handle three-class crossings'
    fulltau=np.full(len(ids),np.nan);fulltau[changed]=tau;frame['critical_alpha']=fulltau[aid]
    curves=[];oracle={}
    for role,cc in counts.items():
        events=pd.DataFrame({'tau':tau})
        for c in [0,1,2]:
            events['PF'+str(c)]=cc[changed,c]*((old[changed]!=c)&(new[changed]==c))
            events['NF'+str(c)]=cc[changed,c]*((old[changed]==c)&(new[changed]!=c))
        grouped=events.groupby('tau').sum().cumsum();grouped.loc[0.]=0;grouped=grouped.sort_index()
        grouped['errors']=int(cc.sum()-cc[np.arange(len(cc)),old].sum())+grouped[['NF0','NF1','NF2']].sum(1)-grouped[['PF0','PF1','PF2']].sum(1)
        grouped['role']=role;grouped.index.name='alpha_crossing';curves.append(grouped.reset_index())
        noM=grouped[grouped.NF1==0];noAny=grouped[grouped[['NF0','NF1','NF2']].sum(1)==0]
        allpf=grouped[grouped.PF2==grouped.PF2.max()]
        oracle[role]={'endpoint':counts_metrics(cc,old,new),'S_repairs_at_zero_M_regression':int(noM.PF2.max()),
            'S_repairs_at_zero_all_class_regression':int(noAny.PF2.max()),'min_M_regressions_preserving_all_endpoint_S_repairs':int(allpf.NF1.min()),
            'min_total_errors_on_scalar_path':int(grouped.errors.min()),'scope':'Uses this role labels to describe oracle frontier; NOT a selected threshold or new model.'}
    pd.concat(curves,ignore_index=True).to_csv(DEST/'scalar_path_frontier.csv',index=False);save(DEST/'scalar_path_diagnosis.json',oracle)
    # Fixed frozen perturbations are sensitivity tests only, not deployable candidates.
    ablations=[]
    for name,beg,end in [('remove_output_bias',None,None),('zero_byte_block_in_residual',0,65792),('zero_fact_block_in_residual',65792,66269),('zero_metadata_block_in_residual',66269,66287)]:
        if beg is None:ss=s-st['out.bias']
        else:
            weights=st['first.weight'].copy();weights[:,beg:end]=0
            ss=z+forward({**st,'first.weight':weights},x)
        pp=ss.argmax(1);np.save(DEST/(name+'_ASA_predictions.npy'),pp.astype(np.int8))
        ablations.append({'name':name,'roles':{role:counts_metrics(cc,old,pp) for role,cc in counts.items()},
            'original_inner_500_remaining':int(counts['inner'][(old==1)&(new!=1)&(pp!=1),1].sum()),
            'original_inner_58_preserved':int(counts['inner'][(old!=2)&(new==2)&(pp==2),2].sum()),
            'scope':'Off-manifold frozen residual sensitivity; teacher unchanged. Does not prove raw field causality or justify deletion.'})
    save(DEST/'frozen_ablations.json',ablations)
    # Query every distinct inner changed input against each TRAINING class.
    target=frame[(frame.role=='inner')&frame.kind.isin(['repair','regression'])]
    queries=np.unique(target.asa_id);nearest=[]
    h=forward(st,x,hidden=True)
    for space,vectors in [('R0',normalize(x)),('hidden16',normalize(h))]:
        for c in [1,2]:
            candidates=np.flatnonzero(counts['fit'][:,c]>0)
            for beg in range(0,len(queries),128):
                qq=queries[beg:beg+128];sim=vectors[qq]@vectors[candidates].T
                sim=sim.toarray() if sparse.issparse(sim) else sim;ii=sim.argmax(1)
                for j,n in enumerate(qq):nearest.append({'asa_id':int(n),'R0_id':int(ids[n]),'space':space,'training_class':c,
                    'nearest_asa_id':int(candidates[ii[j]]),'nearest_R0_id':int(ids[candidates[ii[j]]]),'cosine':float(sim[j,ii[j]])})
    near=pd.DataFrame(nearest);near.to_parquet(DEST/'nearest_training_inputs.parquet',index=False)
    for space in ['R0','hidden16']:
        nc=near[near.space==space].pivot(index='asa_id',columns='training_class',values='cosine')
        for c in [1,2]:frame['nearest_'+space+'_'+str(c)]=frame.asa_id.map(nc[c])
    changes=frame[frame.kind.isin(['regression','repair'])];changes.to_parquet(DEST/'all_changed_rows.parquet',index=False)
    inner=changes[changes.role=='inner'];inner.to_parquet(DEST/'inner_500M_58S_trace.parquet',index=False)
    inner.groupby(['kind','label_index','component']).size().rename('rows').reset_index().to_csv(DEST/'inner_component_changes.csv',index=False)
    near_summary=[]
    for (kind,cls),b in inner.groupby(['kind','label_index']):
        item={'kind':kind,'class':int(cls),'rows':len(b)}
        for space in ['R0','hidden16']:
            a=b['nearest_'+space+'_1'];c=b['nearest_'+space+'_2']
            item[space]={'M_closer_rows':int((a>c+1e-10).sum()),'S_closer_rows':int((c>a+1e-10).sum()),'tie_rows':int((np.abs(a-c)<=1e-10).sum()),'cosine_M':quant(a),'cosine_S':quant(c)}
        near_summary.append(item)
    save(DEST/'nearest_summary.json',near_summary)
    norms={}
    for name in ['U0','U0R']:
        sd=torch.load(PREV/(name+'_model.pt'),map_location='cpu',weights_only=True)['state_dict'];sc=np.load(PREV/(name+'_ASA_scores.npy'));dd=sc-z
        norms[name]={'parameter_l2':float(np.sqrt(sum(float(t.double().square().sum()) for t in sd.values()))),
            'layer_frobenius':{k:float(t.double().norm()) for k,t in sd.items()},'output_bias':sd['out.bias'].tolist(),
            'fit_abs_MS_residual':quant(np.repeat(np.abs(dd[:,1]-dd[:,2]),counts['fit'].sum(1))),
            'inner_abs_MS_residual':quant(np.repeat(np.abs(dd[:,1]-dd[:,2]),counts['inner'].sum(1)))}
    save(DEST/'score_scale_diagnosis.json',norms)
    groups=[]
    for (kind,coarse),b in inner.groupby(['kind','coarse']):
        row=b.iloc[0];groups.append({'kind':kind,'known_behavior':json.loads(coarse),'rows':len(b),'components':b.component.nunique(),
            'fit_M_rows':int(row.fit_coarse_1_rows),'fit_S_rows':int(row.fit_coarse_2_rows),
            'fit_M_components':int(row.fit_coarse_1_components),'fit_S_components':int(row.fit_coarse_2_components),
            'median_old_M_minus_S':float(b.old_M_minus_S.median()),'median_residual_S_minus_M':float(b.residual_S_minus_M.median())})
    save(DEST/'inner_behavior_groups.json',sorted(groups,key=lambda a:-a['rows']))
    receipt={'status':'completed_frozen_diagnosis','new_classifier_fits':0,'new_calibration_fits':0,'selected':None,'quality_acceptance':False,
        'inner_regression_M':int(((inner.kind=='regression')&(inner.label_index==1)).sum()),'inner_repair_S':int(((inner.kind=='repair')&(inner.label_index==2)).sum()),
        'model_replay_max_abs':float(np.max(np.abs(replay-s))),'roles':'Existing repeatedly observed development; no new blind evaluation.',
        'source_sha256':sha(__file__),'seconds':time.monotonic()-start}
    save(DEST/'audit_summary.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
