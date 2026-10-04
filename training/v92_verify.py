"""Independent original-label, numpy model replay and evidence preservation checks."""
import json,time,numpy as np,pandas as pd,torch,pyarrow.parquet as pq
from scipy import sparse
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from v92_common import *
from v75_corrective import stable
from v75_views import byte_matrix,BYTE_FEATURES

def forward(state,x):
    st={k:v.numpy() for k,v in state.items()};x=x.astype(float)
    if 'first.weight' not in st:return np.asarray(x@st['out.weight'].T)+st['out.bias']
    gelu=lambda z:.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)))
    return gelu(gelu(np.asarray(x@st['first.weight'].T))@st['second.weight'].T+st['second.bias'])@st['out.weight'].T+st['out.bias']

def main():
    start=time.monotonic();reg=check();assert not (DEST/'verification.json').exists()
    r,y,fid,z,old,sel,fit=data();original=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();assert np.array_equal(original,y)
    ids=np.load(DEST/'ASA_input_ids.npy');x=sparse.load_npz(DEST/'ASA_R0.npz');cc=np.load(DEST/'ASA_fit_counts.npy')
    counts=pd.DataFrame({'fid':fid[fit],'y':original[fit]}).groupby(['fid','y']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
    npcc=np.zeros((len(z),3),np.int64);npcc[counts.index]=counts.to_numpy();assert np.array_equal(npcc,np.load(DEST/'fit_counts.npy')) and np.array_equal(cc,npcc[ids]);assert cc.sum()==38771 and npcc.sum()==922324
    outside=np.ones(len(z),bool);outside[ids]=False;names=reg['arms']+read(DEST/'numerical_completion_registration.json')['arms'];maxdiff={};nonASAchanges={}
    for name in names:
        rec=read(DEST/(name+'_fit.json'));saved=np.load(DEST/(name+'_ASA_scores.npy'));pred=np.load(DEST/(name+'_prediction.npy'))
        if name=='L0R':
            model=np.load(DEST/(name+'_model.npz'));replay=np.asarray(z[ids])+x@model['coef']+model['intercept']
        else:
            model=torch.load(DEST/(name+'_model.pt'),map_location='cpu',weights_only=True);replay=np.asarray(z[ids])+forward(model['state_dict'],x)
        difference=float(np.abs(saved-replay).max());maxdiff[name]=difference;assert difference<1e-4,(name,difference)
        assert np.array_equal(replay.argmax(1),saved.argmax(1)),('decision_replay_diff',name)
        assert np.array_equal(pred[ids],replay.argmax(1)) and np.array_equal(pred[outside],old[outside]);nonASAchanges[name]=int((pred[outside]!=old[outside]).sum())
        for role,mask in [('fit',fit),('inner',r.fold.eq(1).to_numpy()),('C',r.fold.eq(2).to_numpy()),('H',r.fold.eq(0).to_numpy())]:
            if role not in rec['roles']:continue
            yy=original[mask];a=old[fid[mask]];b=pred[fid[mask]];cm=pd.crosstab(pd.Series(yy),pd.Series(b)).reindex(index=[0,1,2],columns=[0,1,2],fill_value=0).to_numpy()
            rr=rec['roles'][role];assert np.array_equal(cm,rr['cm']) and int((b!=yy).sum())==rr['errors']
            assert int(((a!=yy)&(b==yy)).sum())==rr['positive_flips'] and int(((a==yy)&(b!=yy)).sum())==rr['negative_flips']
        roi=float((cc.sum(1)@logsumexp(replay,axis=1)-(cc*replay).sum())/cc.sum());assert abs(roi-rec.get('last_ASA_ce',rec.get('ASA_ce')))<1e-7,(name,roi)
    trace=pd.read_parquet(DEST/'error_row_trace.parquet');assert len(trace)==720 and trace.row_position.nunique()==720 and trace.kind.value_counts().to_dict()=={'F00_M_regression':668,'support_S':52}
    targets={int(a.row_position):a for a in trace.itertuples()};offset=0;rawchecks=0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=16384,columns=['message_sanitized'],use_threads=False):
        for j,raw in enumerate(batch.column(0).to_pylist()):
            i=offset+j
            if i in targets:assert __import__('hashlib').sha256(raw.encode()).hexdigest()==targets[i].source_raw_sha256;rawchecks+=1
        offset+=batch.num_rows
    texts=pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text;groups=read(DEST/'input_group_trace.json')
    for g in groups:
        i=g['representative_row'];assert stable(texts.loc[r.new_text_id.iloc[i]])==g['current_text']
        assert np.all(fid[np.asarray(g['error_rows'])]==g['R0_id']);diff=x[np.searchsorted(ids,fid[i]),:BYTE_FEATURES]-byte_matrix([g['current_text']]);assert diff.nnz==0 or np.abs(diff.data).max()<1e-7
    cert=read(DEST/'constructive_fit_certificate.json');s=np.load(DEST/'U0R_ASA_scores.npy');op=old[ids];mass=cc[np.arange(len(ids)),op];P=mass>0
    other=s.copy();other[np.arange(len(ids)),op]=-np.inf;mm=s[np.arange(len(ids)),op]-other.max(1);tt=z[ids].copy();tt[np.arange(len(ids)),op]=-np.inf;eps=np.minimum(.001,(z[ids][np.arange(len(ids)),op]-tt.max(1))/2)
    assert np.maximum(eps[P]-mm[P],0).max()==0 and cert['all_training_P_feasible'];table=cc[cc.sum(1)>0];floor=int((table.sum(1)-table.max(1)).sum());assert floor==22 and int((cc.sum(1)-cc[np.arange(len(ids)),s.argmax(1)]).sum())==22
    pred=np.load(DEST/'U0R_prediction.npy');es=trace[trace.kind=='support_S'].row_position.to_numpy();assert np.all(pred[fid[es]]==2)
    rare=read(DEST/'rare_rotation_registration.json');assert rare['source_sha256']==sha(ROOT/'training/v92_rare_rotation.py');rare_verified=[]
    for spec in rare['conditions']:
        name=spec['name'];roles=np.load(DEST/(name+'_roles.npz'));tr=roles['train'];te=roles['test'];assert not set(r.component.iloc[tr])&set(r.component.iloc[te])
        model=np.load(DEST/(name+'_model.npz'));score=x[np.searchsorted(ids,fid[te])]@model['coef']+model['intercept'];savepred=np.load(DEST/(name+'_predictions.npz'));assert np.allclose(score,savepred['scores'],rtol=0,atol=1e-12) and np.array_equal(score.argmax(1),savepred['prediction'])
        report=read(DEST/(name+'_fit.json'));target=np.isin(te,roles['target']);assert int((score.argmax(1)[target]==2).sum())==report['target_S_correct'];rare_verified.append(name)
    assert read(DEST/'selection.json')['selected'] is None
    ledger=read(DEST/'execution_ledger.json');started=[a for a in ledger if a['stage']=='classifier_fit' and a['status']=='started'];finished=[a for a in ledger if a['stage']=='classifier_fit' and a['status']!='started'];assert len(started)==len(finished)==9 and {a['name'] for a in started}==set(names+rare_verified)
    prior={};receipt_paths=read(ROOT/'artifacts/v91_first_principles_20260928/verification.json')['receipt_sha256'];receipt_paths=list(receipt_paths)+['evidence/2026-09-28/v91_first_principles/delivery.json']
    for p in receipt_paths:
        for path,h in read(ROOT/p)['artifact_sha256'].items():assert path not in prior or prior[path]==h;prior[path]=h
    changed=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not changed,changed
    result={'status':'passed','source_sha256':sha(__file__),'actual_classifier_fits':9,'actual_calibration_fits':0,'original_labels_verified':len(original),'full_parameter_population_verified':922324,'ASA_population_verified':38771,'predefined_error_raw_rows_verified':rawchecks,'actual_text_matrix_groups_verified':len(groups),'model_replays':names,'max_numpy_score_replay_difference':maxdiff,'non_ASA_changed_input_groups':nonASAchanges,'rare_component_rotation_replays':rare_verified,'ASA_empirical_minimum_errors_attained':floor,'constructive_P_endpoint_feasible':True,'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,'selected_model':None,'quality_acceptance':False,'seconds':time.monotonic()-start,'scope':'Saved-model prediction and original-row class/error receipts verified on existing development roles. Does not certify generalization, model quality or cloud/platform state.'}
    save(DEST/'verification.json',result);event('verification','passed',receipt=result);emit(**result)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
