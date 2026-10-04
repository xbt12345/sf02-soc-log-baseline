"""Finalize actual R0 stress checks and feasible matched support controls before fits."""
import collections
import json
import re
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
from scipy import sparse
from sklearn.preprocessing import normalize
from v89_common import *
from v89_prepare import keyhash
from v89_partial_facts import parse
from v75_views import view,byte_matrix
from v75_corrective import stable
from v75_metadata import encode
from run_v75 import adapter


def main():
    assert not (DEST/'registration.json').exists()
    r,y,fid,z,old,sel,fit=data();facts=[json.loads(s) for s in pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts]
    fields=['action','outcome','transport_protocol','src_role','dst_role','dst_port_fixed']
    keys=[tuple(str(f[k]) for k in fields) if all(k in f for k in fields) else None for f in facts]
    rowkey=pd.Series(keys,dtype=object).to_numpy()[r.projection_id.to_numpy()]
    selected=np.zeros(len(r),bool);selected[sel]=True;asa=r.route.eq('asa').to_numpy()
    strict=read(DEST/'support_design.json');assert not strict['feasible']
    shutil.copyfile(DEST/'support_design.json',DEST/'support_design_strict_component.json')
    pools=collections.defaultdict(lambda:collections.defaultdict(set))
    for i in np.flatnonzero(selected&asa):
        if rowkey[i] is not None:pools[rowkey[i]][int(y[i])].add(int(r.component.iloc[i]))
    candidates=sorted([k for k,d in pools.items() if len(d[1])>=3 and len(d[2])>=3],key=keyhash)
    chosen=[];components=set();rejected=[]
    for k in candidates:
        if len(chosen)==3:break
        possible=[];union=set(chosen+[k])
        for cm in sorted(pools[k][1],key=lambda c:keyhash(('M',k,c))):
            for cs in sorted(pools[k][2],key=lambda c:keyhash(('S',k,c))):
                comps=components|{cm,cs}
                if all(len(pools[kk][c]-comps)>=2 for kk in union for c in [1,2]):possible.append(comps);break
            if possible:break
        if not possible:rejected.append({'key':k,'reason':'remaining_components_insufficient'});continue
        comps=possible[0];train=selected&~r.component.isin(comps).to_numpy();target=np.asarray([a in union for a in rowkey],bool)
        good=True
        for cls in [1,2]:
            needed=r.fold[train&asa&target&(y==cls)].value_counts();available=r.fold[train&asa&~target&(y==cls)].value_counts()
            if any(available.get(f,0)<n for f,n in needed.items()):good=False;break
        if good:chosen.append(k);components=comps
        else:rejected.append({'key':k,'reason':'same_class_route_and_source_fold_matched_mass_unavailable'})
    assert chosen,'No viable class/source-fold-matched controls; do not fake conditions'
    target=np.asarray([a in set(chosen) for a in rowkey],bool);train=selected&~r.component.isin(components).to_numpy()
    hold=fit&r.component.isin(components).to_numpy();evaluation=hold&target&asa;masks={'full':train.copy()};matching=[]
    rng=np.random.default_rng(8901)
    for cls in [1,2]:
        removal=train&target&asa&(y==cls);control=np.zeros(len(r),bool)
        for fold,n in r.fold[removal].value_counts().sort_index().items():
            for comp,m in r.component[removal&r.fold.eq(fold).to_numpy()].value_counts().sort_index().items():
                pool=np.flatnonzero(train&~target&asa&(y==cls)&r.component.eq(comp).to_numpy()&~control)
                if len(pool):control[rng.choice(pool,min(int(m),len(pool)),replace=False)]=True
            remaining=int(n)-int((control&r.fold.eq(fold).to_numpy()).sum())
            pool=np.flatnonzero(train&~target&asa&(y==cls)&r.fold.eq(fold).to_numpy()&~control)
            assert len(pool)>=remaining
            if remaining:control[rng.choice(pool,remaining,replace=False)]=True
        assert control.sum()==removal.sum()
        assert np.array_equal(np.bincount(y[control],minlength=3),np.bincount(y[removal],minlength=3))
        assert r.fold[control].value_counts().sort_index().equals(r.fold[removal].value_counts().sort_index())
        a=r.component[removal].value_counts();b=r.component[control].value_counts();allcomps=a.index.union(b.index)
        drift=int(np.abs(a.reindex(allcomps,fill_value=0)-b.reindex(allcomps,fill_value=0)).sum()//2)
        matching.append({'class':cls,'withdrawn_rows':int(removal.sum()),'class_route_source_fold_exact':True,
            'same_component_match_unavailable_rows':drift,'residual_component_composition_confound':bool(drift)})
        masks['withdraw_'+str(cls)]=train&~removal;masks['control_'+str(cls)]=train&~control
    for name,m in masks.items():np.save(DEST/('support_'+name+'_rows.npy'),np.flatnonzero(m).astype(np.int32))
    np.save(DEST/'support_evaluation_rows.npy',np.flatnonzero(hold).astype(np.int32));np.save(DEST/'support_target_rows.npy',np.flatnonzero(evaluation).astype(np.int32))
    design={'feasible':True,'chosen_keys':chosen,'holdout_components':sorted(components),'rejected':rejected,
      'selection_policy':'Training-only sha256 ordered keys, up to3; >=2 remaining components each class/key; exact class/ASA route/source-fold removal controls; same component preferred where available.',
      'training_counts':{k:np.bincount(y[m],minlength=3).tolist() for k,m in masks.items()},
      'evaluation_class_counts':np.bincount(y[hold],minlength=3).tolist(),'target_evaluation_class_counts':np.bincount(y[evaluation],minlength=3).tolist(),
      'control_matching':matching,'strict_component_design_failed':True,'not_selected_by_prediction_errors':True,
      'source_sha256':sha(__file__),'causal_scope':'Class mass, route and source folds matched; residual subject composition explicitly unresolved. Not an isolated causal estimate of support independent from every subject effect.'}
    save(DEST/'support_design.json',design)
    # Reconstruct actual production R0. The initial view-only audit omitted the corrective stable layer.
    examples=read(DEST/'input_stress.json');wanted={a['row_position'] for a in examples};samples={};offset=0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=16384,use_threads=False):
        end=offset+batch.num_rows;take=sorted(i for i in wanted if offset<=i<end)
        if take:
            df=batch.to_pandas()
            for i in take:samples[i]=df.iloc[i-offset].to_dict()
        offset=end
        if len(samples)==len(wanted):break
    parser=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');X=load_sparse(LAST/'X');result=[]
    def make(row):
        raw=row['message_sanitized'];rec=parser.prepare_record(row)
        fx=normalize(enc.transform([rec['facts']]).astype(np.float32),norm='l2',copy=False)
        mx,_,_=encode([row['src_port']],[rec['facts'].get('src_port_fixed',65536)])
        return sparse.hstack([byte_matrix([stable(view(raw)[0])]),fx,mx],format='csr')
    same=lambda a,b: (a.shape==b.shape and (a-b).nnz==0)
    for i,row in sorted(samples.items()):
        orig=make(row);rename=row.copy();rename['message_sanitized']=re.sub(r'((?:USER|HOST|CRED|ORG)-)(\d+)',lambda m:m[1]+str(int(m[2])+7654321),row['message_sanitized'] or '')
        blank=row.copy();blank['product_name']=None;blank['vendor_name']=None
        clock=row.copy();raw=row['message_sanitized'];_,ledger=view(raw);parts=[];cursor=0
        for a,b,kind in ledger['spans']:
            segment=raw[a:b]
            if kind=='absolute_clock':segment=re.sub(r'\b(\d{2}):(\d{2}):(\d{2})\b',lambda m:f'{(int(m[1])+1)%24:02}:{m[2]}:{m[3]}',segment)
            parts.append(segment)
        clock['message_sanitized']=''.join(parts) if raw is not None else None
        rowresult={'row_position':i,'route':r.route.iloc[i],'class':int(y[i]),'original_R0_runtime_matches_cached':same(orig,X[fid[i]]),
            'actual_R0_identity_invariant':same(orig,make(rename)),'actual_R0_product_empty_invariant':same(orig,make(blank)),
            'actual_R0_named_absolute_clock_invariant':same(orig,make(clock)),
            'clock_scope':'Editing recognized clock span clock fields; no chronological/context inference guarantee or unrecognized malformed-clock coverage'}
        result.append(rowresult)
    save(DEST/'actual_R0_stress.json',result)
    receipt={'status':'contract_prepared','source_sha256':sha(__file__),'support_design_sha256':sha(DEST/'support_design.json'),
       'stress_examples':len(result),'R0_runtime_cache_mismatches':sum(not a['original_R0_runtime_matches_cached'] for a in result),
       'actual_R0_identity_failures':sum(not a['actual_R0_identity_invariant'] for a in result),
       'actual_R0_product_failures':sum(not a['actual_R0_product_empty_invariant'] for a in result),
       'actual_R0_clock_failures':sum(not a['actual_R0_named_absolute_clock_invariant'] for a in result),
       'preliminary_view_only_failures_are_not_actual_R0_failures':True,'support_feasible':True,
       'support_matching_residual':matching,'new_fits':0}
    save(DEST/'contract_preparation.json',receipt);emit(**receipt)


if __name__=='__main__':main()
