"""Independent original-row and saved-weight verification for v86 no-fit audit."""
import json
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,OUT,read,save,sha,load_sparse
from v79_execute import rows
from v82_capacity import LAST
from v85_verify import cpu_manual,direct
from v86_boundary_diagnosis import DEST,OLD


def main():
    d=read(DEST/'diagnosis.json');support=read(DEST/'support_and_solver.json')
    assert d['source_sha256']==sha(ROOT/'training/v86_boundary_diagnosis.py')
    assert support['source_sha256']==sha(ROOT/'training/v86_support_diagnosis.py')
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X');folder=OLD/'fold1'
    chosen=np.load(folder/'selected_rows.npy');sel=np.zeros(len(r),bool);sel[chosen]=True
    roles={'selected_fit':sel,'fit_full':~r.fold.isin([0,1,2]).to_numpy(),'inner':r.fold.eq(1).to_numpy(),'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    z=np.load(folder/'teacher_scores.npy',mmap_mode='r');old=np.load(folder/'teacher_prediction.npy');targets=read(DEST/'regression_neighbors.json')
    sample=np.load(DEST/'layer_probe_feature_ids.npy');sample=np.unique(np.r_[sample,[t['fid'] for t in targets]])
    maxdiff=0.;checks=0
    for m in d['models']:
        arm=m['arm'];delta=np.load(DEST/(arm+'_residual_logits.npy'));state=torch.load(folder/(arm+'_epoch060.pt'),weights_only=True,map_location='cpu')
        actual=cpu_manual(x[sample],state);diff=float(np.max(abs(actual-delta[sample])));maxdiff=max(maxdiff,diff);assert diff<1e-4
        mean=np.asarray(delta[fid[chosen]],np.float64).mean(0);assert np.allclose(mean,m['train_weighted_mean_residual'],atol=1e-10)
        zero=cpu_manual(sparse.csr_matrix((1,x.shape[1]),dtype=np.float32),state)[0];assert np.allclose(zero,m['zero_input_residual'],atol=1e-6)
        full=(z+delta).argmax(1);constant=(z+mean).argmax(1);nobias=(z+delta-np.array(m['zero_input_residual'],np.float32)).argmax(1)
        assert np.array_equal(full,np.load(folder/(arm+'_epoch060_all_prediction.npy')))
        for q in m['roles']:
            mask=roles[q['role']];yr=y[mask];a=old[fid[mask]];b=full[fid[mask]];c=constant[fid[mask]]
            for candidate,key in [(b,'trained_residual'),(c,'train_mean_constant_control'),(nobias[fid[mask]],'remove_zero_input_output_diagnostic')]:
                stat=direct(yr,a,candidate)
                for k,v in stat.items():assert v==q[key][k],(arm,q['role'],key,k)
            assert int(((a==yr)&(b!=yr)&(b==c)).sum())==q['negative_flips_reproduced_by_constant'];checks+=1
    # Rebuild support independently as joins over original selected rows.
    pf=pd.read_parquet(OUT/'projections.parquet',columns=['facts']);facts=[json.loads(v) for v in pf.facts]
    frame=pd.DataFrame([facts[int(pid)] for pid in r.projection_id.iloc[chosen]])
    frame['class']=y[chosen];frame['component']=r.component.to_numpy()[chosen];frame['fid']=fid[chosen]
    for rec in support['train_only_support']:
        target=next(t for t in targets if t['fid']==rec['fid'])['target_facts']
        for q in rec['levels']:
            if q['name']=='all_parsed_facts':mask=np.array([facts[int(pid)]==target for pid in r.projection_id.iloc[chosen]])
            else:
                keys=['action','outcome','transport_protocol','src_role','dst_role']+(['dst_port_fixed'] if q['name']=='behavior_and_destination_port' else [])
                mask=np.ones(len(frame),bool)
                for key in keys:mask&=frame[key].eq(target.get(key)).to_numpy()
            part=frame.loc[mask];assert len(part)==q['rows']
            for c in q['by_class_components']:
                a=part[part['class']==c['class']]
                assert (len(a),a.component.nunique(),a.fid.nunique())==(c['rows'],c['components'],c['unique_encoded_inputs'])
    # Recheck auxiliary-pair candidate coverage by explicit component sets.
    for rec in support['train_only_contrastive_candidate_coverage']:
        keys=['action','outcome','transport_protocol','src_role','dst_role']+(['dst_port_fixed'] if rec['level']=='behavior_and_destination_port' else [])
        pool=chosen[(r.route.to_numpy()[chosen]=='asa')&(y[chosen]>0)];comp={};classes={};ks=[]
        for i in pool:
            f=facts[int(r.projection_id.iloc[i])];key=tuple((k,f.get(k),k in f) for k in keys);ks.append(key)
            comp.setdefault((key,int(y[i])),set()).add(int(r.component.iloc[i]));classes.setdefault(key,set()).add(int(y[i]))
        for item in rec['by_class']:
            eligible=[j for j,i in enumerate(pool) if int(y[i])==item['class'] and bool(old[fid[i]]!=y[i])==item['old_wrong']]
            pos=sum(len(comp[(ks[j],item['class'])])>=2 for j in eligible)
            both=sum(len(comp[(ks[j],item['class'])])>=2 and len(classes[ks[j]])==2 for j in eligible)
            assert (len(eligible),pos,both)==(item['rows'],item['positive_crosscomponent_available'],item['both_candidate_conditions'])
    from v75_views import view
    raw={q['row_position']:q for q in read(DEST/'local_raw_cases.json')};casechecks=[]
    for target in targets:
        item=raw[target['target_row']];f=target['target_facts'];text,ledger=view(item['message_sanitized'])
        assert str(f['src_port_fixed']) in text and str(f['dst_port_fixed']) in text
        assert int(item['src_port'])==f['src_port_fixed']
        assert 'Deny tcp src outside:' in text and 'dst dmz-1:' in text
        assert ''.join(item['message_sanitized'][a:b] for a,b,_ in ledger['spans'])==item['message_sanitized']
        casechecks.append({'fid':target['fid'],'action_protocol_direction_ports_retained':True,'independent_src_port_matches':True,
            'ledger_reconstructs_raw':True,'scope':'These observed fields only; not proof of complete semantic sufficiency.'})
    # Independent CPU hidden layer diagnostic on exactly the registered probe ids.
    from v85_verify import gelu_numpy
    probe=np.load(DEST/'layer_probe_feature_ids.npy');layer_checks=[]
    for item in d['layer_diagnostics']:
        state=torch.load(folder/(item['name']+'.pt'),weights_only=True,map_location='cpu');w={k:v.numpy() for k,v in state.items()}
        pre=np.asarray(x[probe]@w['first.weight'].T);h=gelu_numpy(gelu_numpy(pre)@w['second.weight'].T+w['second.bias']).astype(float)
        centered=h-h.mean(0);sing=np.linalg.svd(centered,compute_uv=False);ev=sing**2;rank=float(ev.sum()**2/(ev**2).sum())
        assert abs(rank-item['hidden_covariance_participation_rank'])<1e-4
        energy=float((centered**2).sum()/(h**2).sum());assert abs(energy-item['hidden_centered_energy_fraction'])<1e-5
        layer_checks.append({'name':item['name'],'CPU_SVD_participation_rank':rank,'CPU_centered_energy_fraction':energy})
    prog=read(folder/'C_progress.json');assert sum(s['aux_before_step']!=0 for s in prog)==support['C_aux_nonzero_epochs']==0
    assert [s['epoch'] for s in prog if s['fully_rejected']]==list(range(33,61))
    for path,digest in read(OLD/'registration.json')['input_sha256'].items():assert sha(ROOT/path)==digest
    out={'status':'passed','source_sha256':sha(__file__),'diagnosis_sha256':sha(DEST/'diagnosis.json'),
       'support_sha256':sha(DEST/'support_and_solver.json'),'no_new_classifier_fits':True,'no_new_calibration_fits':True,
       'all_role_original_row_replays_checked':checks,'CPU_logit_probe_inputs':len(sample),'CPU_max_logit_difference':maxdiff,
       'CPU_layer_checks':layer_checks,'raw_official_inputs_hashes_unchanged':True,
       'pair_candidate_coverage_checked':True,'raw_case_checks':casechecks,
       'scope':'Original-row controls, train-only matched support, saved-weight CPU probes and CPU SVD. Layer rank is a sample statistic, not a capacity impossibility proof; support keys do not establish true attack intent.'}
    save(DEST/'verification.json',out);print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
