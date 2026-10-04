"""No-fit decomposition of v85 malicious regression and residual specificity."""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,OUT,read,save,sha,load_sparse
from v79_execute import rows
from v82_capacity import LAST
from v85_protection import Residual,inference,csr_tensor,changes,raw_counts

OLD=ROOT/'artifacts/v85_protection_20260927'
DEST=ROOT/'artifacts/v86_boundary_review_20260927'


def quantile(a):return np.quantile(a,[0,.1,.5,.9,1]).tolist() if len(a) else []


def main():
    if (DEST/'review_receipt.json').exists():raise FileExistsError('Frozen review')
    DEST.mkdir(exist_ok=True);r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X')
    folder=OLD/'fold1';sel=np.zeros(len(r),bool);sel[np.load(folder/'selected_rows.npy')]=True
    fit=~r.fold.isin([0,1,2]).to_numpy();roles={'selected_fit':sel,'fit_full':fit,'inner':r.fold.eq(1).to_numpy(),'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    z=np.load(folder/'teacher_scores.npy',mmap_mode='r');old=np.load(folder/'teacher_prediction.npy');counts=raw_counts(fid,y,sel,x.shape[0]);used=np.flatnonzero(counts.sum(1))
    allids=np.arange(x.shape[0]);models=[];points=[];delta_C=None
    for arm in 'ABC':
        model=Residual().cuda();model.load_state_dict(torch.load(folder/(arm+'_epoch060.pt'),map_location='cuda',weights_only=True))
        delta=inference(model,x,allids);pred=(z+delta).argmax(1)
        assert np.array_equal(pred,np.load(folder/(arm+'_epoch060_all_prediction.npy')))
        np.save(DEST/(arm+'_residual_logits.npy'),delta)
        weighted_mean=np.average(delta[used].astype(float),axis=0,weights=counts[used].sum(1))
        constant=(z+weighted_mean).argmax(1)
        b0=inference(model,sparse.csr_matrix((1,x.shape[1]),dtype=np.float32),np.array([0]))[0]
        noc=(z+delta-b0).argmax(1);roledata=[]
        for role,mask in roles.items():
            cc=raw_counts(fid,y,mask,x.shape[0]);genuine=changes(cc,old,pred);uniform=changes(cc,old,constant)
            oldr=old[fid[mask]];newr=pred[fid[mask]];unir=constant[fid[mask]];yr=y[mask]
            neg=(oldr==yr)&(newr!=yr);ms=(y[mask]>0)&r.route.to_numpy()[mask].astype(str).__eq__('asa')
            ds=(delta[fid[mask],2]-delta[fid[mask],1])[ms]
            roledata.append({'role':role,'trained_residual':genuine,'train_mean_constant_control':uniform,
                'remove_zero_input_output_diagnostic':changes(cc,old,noc),
                'negative_flips_reproduced_by_constant':int((neg&(unir==newr)).sum()),
                'ASA_MS_residual_S_minus_M_quantiles':quantile(ds)})
            for i in np.flatnonzero(mask&(y==1)&(old[fid]==1)&(pred[fid]!=1)):
                row=r.iloc[i];j=fid[i];points.append({'arm':arm,'role':role,'row_position':int(i),'fid':int(j),'component':int(row.component),'route':row.route,
                   'new_class':int(pred[j]),'old_M_minus_S':float(z[j,1]-z[j,2]),'residual_M_minus_S':float(delta[j,1]-delta[j,2]),
                   'new_M_minus_S':float(z[j,1]+delta[j,1]-z[j,2]-delta[j,2])})
        ms=sel&(r.route.to_numpy()=='asa')&(y>0);ds=delta[fid[ms],2]-delta[fid[ms],1]
        models.append({'arm':arm,'train_weighted_mean_residual':weighted_mean.tolist(),'zero_input_residual':b0.tolist(),
          'ASA_MS_constant_offset_squared_energy_fraction':float(np.mean(ds,dtype=float)**2/np.mean(ds.astype(float)**2)),
          'ASA_MS_symmetric_class_details':[{'class':cls,'rows':int((ms&(y==cls)).sum()),'S_minus_M_residual_quantiles':quantile((delta[fid,2]-delta[fid,1])[ms&(y==cls)])} for cls in [1,2]],
          'roles':roledata})
        if arm=='C':delta_C=delta
    p=pd.DataFrame(points);p.to_parquet(DEST/'malicious_regression_rows.parquet',index=False)
    # Closest same/opposite-class training examples for each distinct C-regression input,
    # plus the train-only protected blocker. No neighbours are used for prediction.
    targets=sorted(set(p.loc[(p.arm=='C')&(p.role.isin(['inner','C','H'])),'fid'].tolist()+[2128]))
    projections=pd.read_parquet(OUT/'projections.parquet',columns=['facts']);facts=[json.loads(v) for v in projections.facts]
    selected_first={}
    for i in np.flatnonzero(sel):selected_first.setdefault((int(fid[i]),int(y[i])),int(i))
    records=[];rawpos=set()
    for target in targets:
        reps=np.flatnonzero(fid==target);targetpos=int(reps[0]);rawpos.add(targetpos)
        entry={'fid':target,'original_rows_by_fold_class':r.iloc[reps].groupby(['fold','label_index']).size().rename('rows').reset_index().to_dict('records'),
          'target_row':targetpos,'target_facts':facts[int(r.projection_id.iloc[targetpos])],'old_logits':z[target].tolist(),'C_delta':delta_C[target].tolist(),'nearest_training':[]}
        for cls in [1,2]:
            ids=used[counts[used,cls]>0];sims=cosine_similarity(x[target],x[ids]).ravel();order=np.argsort(-sims)[:5]
            for o in order:
                near=int(ids[o]);row=selected_first[(near,cls)];rawpos.add(row)
                entry['nearest_training'].append({'class':cls,'fid':near,'cosine':float(sims[o]),'original_row':row,'component':int(r.component.iloc[row]),
                      'fit_count':int(counts[near,cls]),'old_class':int(old[near]),'new_class':int((z[near]+delta_C[near]).argmax()),
                      'old_logits':z[near].tolist(),'C_delta':delta_C[near].tolist(),'facts':facts[int(r.projection_id.iloc[row])]})
        records.append(entry)
    save(DEST/'regression_neighbors.json',records)
    # Only selected original raw rows are retained locally, never MCP-whitelisted.
    offset=0;raw=[];wanted=sorted(rawpos)
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=8192,use_threads=False,
         columns=['event_id','message_sanitized','src_port','product_name','vendor_name','label_binary']):
        ix=[i-offset for i in wanted if offset<=i<offset+len(batch)]
        if ix:
            for k in ix:
                record={name:batch.column(j)[k].as_py() for j,name in enumerate(batch.schema.names)};record['row_position']=offset+k;raw.append(record)
        offset+=len(batch)
    save(DEST/'local_raw_cases.json',raw)
    # Quantify representation scale and directional diversity on a fixed selection,
    # no transform fitting and no candidate training.
    rng=np.random.default_rng(8601);sample=np.sort(rng.choice(used,min(2048,len(used)),replace=False));layers=[]
    for name in ['C_epoch000','C_epoch060']:
        m=Residual().cuda();m.load_state_dict(torch.load(folder/(name+'.pt'),map_location='cuda',weights_only=True));xx=csr_tensor(x[sample])
        with torch.no_grad():
            pre=torch.sparse.mm(xx,m.first.weight.T);a=torch.nn.functional.gelu(pre,approximate='tanh');pre2=m.second(a);b=torch.nn.functional.gelu(pre2,approximate='tanh')
            centered=b-b.mean(0);gram=centered.T@centered;ev=torch.linalg.eigvalsh(gram).clamp_min(0)
            layers.append({'name':name,'first_abs_preactivation_quantiles':quantile(pre.abs().cpu().numpy().ravel()),
                 'first_abs_lt_point1_fraction':float((pre.abs()<.1).float().mean()),
                 'hidden_centered_energy_fraction':float(centered.square().sum()/b.square().sum()),
                 'hidden_covariance_participation_rank':float(ev.sum().square()/ev.square().sum()),
                 'sample_rows':len(sample),'sampling_scope':'uniform unique actual-selected fit inputs, no labels used in sample'})
    np.save(DEST/'layer_probe_feature_ids.npy',sample)
    prior={}
    for file in ['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json',
       'evidence/2026-09-27/v81_diagnosis/delivery.json','evidence/2026-09-27/v82_capacity/delivery.json',
       'evidence/2026-09-27/v83_root_review/delivery.json','evidence/2026-09-27/v84_preservation/delivery.json','evidence/2026-09-27/v85_protection/delivery.json']:
        prior.update(read(ROOT/file)['artifact_sha256'])
    changed=[k for k,v in prior.items() if sha(ROOT/k)!=v];assert not changed,changed
    output={'source_sha256':sha(__file__),'new_classifier_fits':0,'new_calibration_fits':0,'models':models,'layer_diagnostics':layers,
      'C_regression_unique_inputs':int(p.loc[(p.arm=='C')&(p.role.isin(['inner','C','H'])),'fid'].nunique()),
      'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,
      'scope':'No-fit frozen v85 score decomposition, constant/zero-input controls, raw-case neighbourhood and sampled layer diagnosis. No new classifier selection, transfer acceptance or raw label causal truth claim.'}
    save(DEST/'diagnosis.json',output)
    print(json.dumps({'layers':layers,'constant_fractions':[(m['arm'],m['ASA_MS_constant_offset_squared_energy_fraction']) for m in models],
        'C_targets':targets,'old_files':len(prior)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=4):main()
