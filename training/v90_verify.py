"""Independent original-label, literal-token, frozen-decision and receipt verification."""
import collections,json,time
import numpy as np,pandas as pd
from sklearn.utils import murmurhash3_32
from threadpoolctl import threadpool_limits
from v89_common import ROOT,OUT,DEST as V89,OLD,PRIOR,read,save,sha
DEST=ROOT/'artifacts/v90_root_review_20260928'

def main():
    start=time.monotonic();assert not (DEST/'verification.json').exists()
    d=read(DEST/'diagnosis.json');ds=read(DEST/'score_decomposition.json');cf=read(DEST/'fixed_model_counterfactuals.json')
    assert d['source_sha256']==sha(ROOT/'training/v90_diagnose.py') and ds['source_sha256']==sha(ROOT/'training/v90_score_decomposition.py')
    r=pd.read_parquet(OUT/'rows.parquet');y=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert np.array_equal(y,r.label_index) and np.array_equal(r.row_position,np.arange(len(y)))
    fit=~r.fold.isin([0,1,2]).to_numpy();inner=r.fold.eq(1).to_numpy()
    obs=np.load(V89/'row_fact_code.npy');facts=pd.read_parquet(V89/'row_fact_dictionary.parquet').observation_json
    tt=[];coarse=[]
    for text in facts:
        a=json.loads(text);k={key:value for key,value in a['facts'].items() if a['states'].get(key)=='known'}
        asa=k.get('action')=='deny';prefix='asa:' if asa else 'windows:'
        terms={prefix+key+'='+str(value) for key,value in k.items()}
        names=['action','outcome','transport_protocol','src_role','dst_role']
        if asa and set(names).issubset(k):
            ordered='asa:ordered:'+'|'.join(key+'='+str(k[key]) for key in names);terms.add(ordered)
            for key in ['src_port_fixed','dst_port_fixed']:
                if key in k:terms.add(ordered+'|'+key+'='+str(k[key]))
        elif not asa and 'event_code_observed' in k:
            terms.update('windows:event='+str(k['event_code_observed'])+'|'+key for key in k if key.startswith('privilege_'))
        tt.append(tuple(sorted(terms)));coarse.append(json.dumps({key:value for key,value in k.items() if key!='src_port_fixed'},sort_keys=True))
    train_vocab=set(t for o in set(obs[fit]) for t in tt[o]);m=np.load(V89/'F00_model.npz');theta=m['theta'];coef=theta[:768].reshape(3,256).T/m['scale'][:,None]
    delta=np.zeros((len(tt),3));unseen_delta=delta.copy();source_delta=delta.copy();buckets=[];missing=[];parts=np.zeros((len(tt),4,3))
    for o,terms in enumerate(tt):
        bb=[abs(murmurhash3_32(t,seed=0,positive=False))%128+(0 if t.startswith('asa:') else 128) for t in terms]
        counter=collections.Counter(bb);norm=np.sqrt(sum(n*n for n in counter.values())) or 1.
        buckets.append(tuple(sorted((b,n/norm) for b,n in counter.items())));missing.append(any(t not in train_vocab for t in terms))
        for t,b in zip(terms,bb):
            contribution=coef[b]/norm;delta[o]+=contribution
            if t not in train_vocab:unseen_delta[o]+=contribution
            if 'src_port_fixed=' in t:source_delta[o]+=contribution
            category=0 if 'src_port_fixed=' in t else 1 if 'dst_port_fixed=' in t else 2 if t.startswith('asa:') else 3
            parts[o,category]+=contribution
    gid=np.load(V89/'F00_row_group.npy');pairs=np.load(V89/'F00_group_keys.npy');z=np.load(OLD/'teacher_scores.npy',mmap_mode='r')[pairs[:,0]]
    old=z.argmax(1)[gid];new=(z+delta[pairs[:,1]]).argmax(1)[gid];assert np.array_equal(new,np.load(V89/'F00_all_prediction.npy')[gid])
    roles={'fit_full':fit,'inner':inner,'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    for state in cf['states']:
        label=state['state'];dd=delta-unseen_delta if label.startswith('unseen_') else float(label[len('fixed_scale_'):])*delta
        pp=(z+dd[pairs[:,1]]).argmax(1)[gid];take=roles[state['role']]
        assert np.bincount(3*y[take]+pp[take],minlength=9).reshape(3,3).tolist()==state['cm']
        assert int((take&(pp!=y)).sum())==state['errors']
        for stat in state['classwise']:
            mask=take&(y==stat['class'])
            assert [int(mask.sum()),int((mask&(pp==y)).sum()),int((mask&(old!=y)&(pp==y)).sum()),int((mask&(old==y)&(pp!=y)).sum())]==[stat[k] for k in ['support','correct','repairs','regressions']]
    literal_mass=collections.defaultdict(lambda:np.zeros(3,dtype=np.int64));hash_mass=collections.defaultdict(lambda:np.zeros(3,dtype=np.int64));aliases=collections.defaultdict(set)
    mass=np.bincount(obs[fit]*3+y[fit],minlength=len(tt)*3).reshape(-1,3)
    for i,t in enumerate(tt):literal_mass[t]+=mass[i];hash_mass[buckets[i]]+=mass[i];aliases[buckets[i]].add(t)
    assert len(set(tt))==d['canonical_token_sets'] and len(aliases)==d['distinct_full_hashed_vectors']
    for kind,mask in [('M_regression',inner&(y==1)&(old==y)&(new!=y)),('S_repair',inner&(y==2)&(old!=y)&(new==y))]:
        os=obs[mask];ec=np.array([literal_mass[tt[o]] for o in os]);hc=np.array([hash_mass[buckets[o]] for o in os]);saved=d['target_counts'][kind]
        actual={'rows':int(mask.sum()),'observation_groups':len(set(os)),'components':int(r.component[mask].nunique()),'whole_vector_alias_rows':sum(len(aliases[buckets[o]])>1 for o in os),
            'unseen_token_rows':sum(missing[o] for o in os),'same_exact_tokens_M_support_rows':int((ec[:,1]>0).sum()),'same_exact_tokens_S_support_rows':int((ec[:,2]>0).sum()),
            'same_exact_tokens_no_training_support_rows':int((ec.sum(1)==0).sum()),'false_hash_only_S_support_rows':int(((ec[:,2]==0)&(hc[:,2]>0)).sum())}
        assert actual==saved,(actual,saved)
    no_src=(z+(delta-source_delta)[pairs[:,1]]).argmax(1)[gid];nf=inner&(y==1)&(old==y)&(new!=y);prot=fit&(y==1)&(old==y)
    assert int((nf&(no_src!=1)).sum())==ds['inner_M_regressions_remaining_without_source_terms']
    assert int((prot&(new==1)&(no_src!=1)).sum())==ds['all_fit_protected_M_that_need_source_terms_to_stay_M']
    coarse_rows=np.asarray(coarse)[obs]
    for s in ds['coarse_groups']:
        selected=coarse_rows==json.dumps(s['coarse_behavior'],sort_keys=True);bad=selected&nf;train=selected&prot
        assert int(bad.sum())==s['M_regression_rows'] and np.bincount(y[fit&selected],minlength=3).tolist()==s['training_class_counts']
        assert int((bad&(no_src!=1)).sum())==s['M_regressions_remaining_without_source_terms']
        assert int((train&(new==1)&(no_src!=1)).sum())==s['fit_protected_M_that_need_source_terms_to_stay_M']
        assert np.allclose((parts[obs[bad],:,2]-parts[obs[bad],:,1]).mean(0),list(s['mean_S_minus_M_shift_by_part'].values()),atol=1e-11)
    selected=np.load(PRIOR/'fold1/selected_rows.npy');true=y[selected]
    for stat,dd in zip(ds['main_risk'],[delta*0,delta]):
        score=z[gid[selected]]+dd[obs[selected]];bce=np.logaddexp(0,score).sum(1)-score[np.arange(len(true)),true]
        from scipy.special import logsumexp
        ce=logsumexp(score,axis=1)-score[np.arange(len(true)),true]
        assert abs(bce.mean()-stat['OVR_BCE'])<1e-12 and abs(ce.mean()-stat['softmax_CE'])<1e-12
    e=m['E'];truth=m['truth'];gap=[]
    for i,t in zip(e,truth):gap.append(max(0,max(.01+z[i,c]-z[i,t] for c in range(3) if c!=t)))
    assert abs(m['weights']@gap-ds['zero_correction_weighted_minimum_E_slack'])<1e-12
    assert float(m['slack_cap'])==ds['locked_weighted_E_slack_cap']
    prior={};paths=['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json']
    for ver,key in [(81,'diagnosis'),(82,'capacity'),(83,'root_review'),(84,'preservation'),(85,'protection'),(86,'boundary_review'),(87,'solver_supervision'),(88,'root_review'),(89,'readout_support')]:paths.append(f'evidence/2026-09-27/v{ver}_{key}/delivery.json')
    for p in paths:
        for path,h in read(ROOT/p)['artifact_sha256'].items():
            assert path not in prior or prior[path]==h,path
            prior[path]=h
    changed=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not changed,changed
    result={'status':'passed','source_sha256':sha(__file__),'new_fits':0,'original_labels_checked':len(y),'all_F00_row_predictions_replayed':len(y),
        'frozen_counterfactual_role_results_replayed':len(cf['states']),'manual_literal_hash_normalization_and_coefficient_replay':True,
        'support_unseen_and_alias_counts_recomputed':True,'score_decomposition_groups_recomputed':len(ds['coarse_groups']),'risk_and_slack_recomputed':True,
        'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,'receipt_hashes':{p:sha(ROOT/p) for p in paths},
        'verified_output_sha256':{p.name:sha(p) for p in DEST.glob('*.json')},'seconds':time.monotonic()-start,
        'scope':'Independent original labels, literal-token construction, Murmur bucket arithmetic and frozen row decisions/support/decompositions. Reuses preserved partial-fact dictionary; no renewed full raw parsing or external validation.'}
    save(DEST/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['receipt_hashes','verified_output_sha256']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
