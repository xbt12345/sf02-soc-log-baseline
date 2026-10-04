"""Legal TRAIN pair geometry after relaxing only observed source-port value.

No synthetic rows, input edits, losses, gradients, fits or selection. Root means
retain original frequency; pair eligibility never changes classification mass.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json
import numpy as np
import pandas as pd
import torch
from v142_runtime import ROOT,OUT as RUN,save,sha,require_run_seal
from v142_train import tensors
from v138_train import configure
from v135_model import tensor_hash
from v135_runtime import load_data,fit_context

OUT=ROOT/'artifacts/v143_pair_geometry_20261001'


def observed_key(f):
    p=f.get('transport_protocol')
    if p not in ['tcp','udp'] or not all(0<=f.get(k,65536)<=65535 for k in ['src_port_fixed','dst_port_fixed']):return None
    # Pair lookup alone relaxes exact ephemeral source value; range and state stay.
    value={k:v for k,v in f.items() if k!='src_port_fixed'}
    value['src_port_observed']=True;value['dst_port_observed']=True
    return json.dumps(value,sort_keys=True,separators=(',',':'))


def geometry(frame,z):
    u=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    result=[]
    for key,g in frame.groupby('pair_key'):
        for c,a in g.groupby('truth'):
            roots=a.root.unique()
            if len(roots)<2 or a.canonical_key.nunique()<2:continue
            v=u[a.local].sum(0);den=len(a)**2;num=float(v@v)
            for _,r in a.groupby('root'):
                vr=u[r.local].sum(0);den-=len(r)**2;num-=float(vr@vr)
            same_distance=1-num/den
            other=g[g.truth.ne(c)]
            contrast_distance=None
            if len(other):
                w=u[other.local].sum(0);contrast_distance=1-float(v@w)/(len(a)*len(other))
            result.append(dict(key=key,truth=int(c),rows=len(a),roots=len(roots),distinct_actual_inputs=a.canonical_key.nunique(),
                opposite_rows=len(other),opposite_roots=other.root.nunique(),cross_root_positive_pairs=den,
                mean_cross_root_same_class_cosine_distance=same_distance,
                mean_opposite_class_cosine_distance=contrast_distance,
                class_distance_gap=None if contrast_distance is None else contrast_distance-same_distance))
    return pd.DataFrame(result)


def main():
    if OUT.exists():raise FileExistsError('Keep completed probe')
    require_run_seal(ROOT/'training/v142_train.py');configure();_,d=load_data()
    source=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    facts=pd.read_parquet(source,columns=['row_position','facts_json']);keys=facts.facts_json.map(json.loads).map(observed_key)
    d['pair_key']=keys.to_numpy();reports=[];allgroups=[]
    with torch.no_grad():
        for fold in range(3):
            frame,_,pure,_,_=fit_context(d,fold)
            eligible=frame[frame.pair_key.notna() & pure[frame.local].astype(bool)].copy()
            assert not(set(eligible.root)&set(d[d.fold.eq(fold)].root))
            h,f,m=tensors(fold);initial_hash=tensor_hash(m.state_dict());base=h[:,:,128:].mean(1).cpu().numpy()
            state=torch.load(RUN/f'fold{fold}_S2/endpoint.pt',map_location='cpu',weights_only=True);m.load_state_dict(state['state'])
            before=tensor_hash(m.state_dict());representations=[]
            for i in range(0,len(h),1024):representations.append(m.features(h[i:i+1024]).mean(1).cpu().numpy())
            actual=np.concatenate(representations);assert tensor_hash(m.state_dict())==before
            a=geometry(eligible,base);b=geometry(eligible,actual)
            if len(a):
                a=a.rename(columns={c:'C_'+c for c in ['mean_cross_root_same_class_cosine_distance','mean_opposite_class_cosine_distance','class_distance_gap']})
                b=b.rename(columns={c:'S2_'+c for c in ['mean_cross_root_same_class_cosine_distance','mean_opposite_class_cosine_distance','class_distance_gap']})
                merged=a.merge(b[['key','truth','S2_mean_cross_root_same_class_cosine_distance','S2_mean_opposite_class_cosine_distance','S2_class_distance_gap']],on=['key','truth'],validate='one_to_one')
                merged['training_role']=fold;allgroups.append(merged)
                summaries=[]
                for c,g in merged.groupby('truth'):
                    both=g[g.opposite_rows.gt(0)]
                    summaries.append(dict(truth=int(c),groups=len(g),original_positive_rows=int(g.rows.sum()),
                        groups_with_opposite_class=len(both),original_rows_with_opposite=int(both.rows.sum()),
                        C_equal_group_mean_positive_distance=float(g.C_mean_cross_root_same_class_cosine_distance.mean()),
                        S2_equal_group_mean_positive_distance=float(g.S2_mean_cross_root_same_class_cosine_distance.mean()),
                        C_equal_group_mean_class_gap=None if both.empty else float(both.C_class_distance_gap.mean()),
                        S2_equal_group_mean_class_gap=None if both.empty else float(both.S2_class_distance_gap.mean()),
                        C_nonpositive_class_gap_groups=int(both.C_class_distance_gap.le(0).sum()),
                        S2_nonpositive_class_gap_groups=int(both.S2_class_distance_gap.le(0).sum())))
            else:summaries=[]
            reports.append(dict(fold=fold,legal_original_rows=len(frame),eligible_known_both_ports_pure_rows=len(eligible),
                excluded_from_pair_diagnostic_not_classification=len(frame)-len(eligible),groups=summaries,
                initialized_state_sha256=initial_hash,endpoint_state_sha256=before,endpoint_checkpoint_sha256=sha(RUN/f'fold{fold}_S2/endpoint.pt')))
            del h,f,m,base,actual;gc.collect();torch.cuda.empty_cache()
    OUT.mkdir();table=pd.concat(allgroups,ignore_index=True) if allgroups else pd.DataFrame()
    table.to_parquet(OUT/'legal_TRAIN_pair_geometry.parquet',index=False)
    report=dict(status='frozen_endpoint_legal_TRAIN_geometry_only',latest_actual_training='V142',new_fits=0,new_updates=0,new_gradient_evaluations=0,
        folds=reports,pair_lookup='Full body facts except exact observed src_port_fixed; source range, both observation states, every destination/other fact retained.',
        second_issue_solved=False,new_auxiliary_fit_qualified=False,
        limits=['Mean-member 128D cosine geometry is a diagnostic projection; not a deployed decision boundary or causal guarantee.',
                'Cross-root pair distances use original-row frequencies inside each key; summary group means are descriptive, not training weights.',
                'Pure-input/known-port eligibility applies only to pair diagnosis; all TRAIN rows and frequencies remain in classification.',
                'Source-port variation may change fine semantics; relaxed lookup is a hypothesis requiring contrast/risk qualification, not label-preserving synthesis.',
                'No held predictions or labels enter eligibility or geometry; no gradients, auxiliary coefficient, objective or new classifier selection.'],
        source_sha256=sha(__file__),facts_sha256=sha(source),run_seal_sha256=sha(RUN/'run_seal.json'),output_sha256=sha(OUT/'legal_TRAIN_pair_geometry.parquet'))
    save(OUT/'probe.json',report);print(json.dumps(reports,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
