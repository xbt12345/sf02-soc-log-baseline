"""No-fit partial-fact, all-format and deterministic support-control preparation."""
import collections
import functools
import hashlib
import json
import re
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from v89_common import ROOT,OUT,DEST,PRIOR,LAST,OLD,data,read,save,sha,emit
from v89_partial_facts import parse,self_check
from v75_views import view,MARKER,ISO_CLOCK


def keyhash(k):return hashlib.sha256(json.dumps(k,sort_keys=True).encode()).hexdigest()


def main():
    if DEST.exists():raise FileExistsError(DEST)
    DEST.mkdir();start=time.monotonic();assert self_check()==20
    r,y,fid,z,old,sel,fit=data();p=pd.read_parquet(OUT/'projections.parquet',columns=['facts'])
    facts=[json.loads(s) for s in p.facts];supp=[None]*len(p);ambiguity=collections.defaultdict(set)
    seen=np.zeros(len(p),bool);audit=collections.defaultdict(collections.Counter);examples={};stress=[]
    hard={13783,13799,74556,74557,74558,74560};hardcases=[]
    rawfile=ROOT/'data/official/train.parquet';offset=0
    @functools.lru_cache(maxsize=2048)
    def parsed(raw,route):return parse(raw,route)
    for batch in pq.ParquetFile(rawfile).iter_batches(batch_size=16384,columns=['message_sanitized','label_binary','src_port'],use_threads=False):
        raw=batch.to_pandas();rr=r.iloc[offset:offset+len(raw)]
        yy=raw.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();assert np.array_equal(yy,y[offset:offset+len(raw)])
        for j,(message,pid,route,label) in enumerate(zip(raw.message_sanitized,rr.projection_id,rr.route,yy)):
            row=offset+j;new=parsed(message,str(route));cell=audit[(str(route),int(label))]
            cell['rows']+=1;cell['old_fact_nonempty']+=int(bool(facts[pid]));cell['supplement_nonempty']+=int(bool(new['facts']))
            novel={k:v for k,v in new['facts'].items() if k not in facts[pid]};cell['novel_known_fact_rows']+=int(bool(novel))
            if not seen[pid]:supp[pid]=new['facts'].copy();seen[pid]=True
            else:
                for k in set(supp[pid])|set(new['facts']):
                    if supp[pid].get(k,'<absent>')!=new['facts'].get(k,'<absent>'):ambiguity[pid].add(k)
            if row in hard:
                hardcases.append({'row_position':row,'route':str(route),'class':int(label),'extracted':new,
                    'original_src_port':None if raw.src_port.iloc[j] is None else str(raw.src_port.iloc[j]),'source_message_sha256':hashlib.sha256(message.encode()).hexdigest()})
            casekey=(str(route),int(label))
            if isinstance(message,str) and len(stress)<160 and casekey not in examples:
                examples[casekey]=row
                # Use the complete official row at these small cases only; structured metadata isn't an input.
                renamed=re.sub(r'((?:USER|HOST|CRED|ORG)-)(\d+)',lambda m:m[1]+str(int(m[2])+7654321),message)
                v0=view(message)[0];v1=view(renamed)[0]
                before=new['facts'];after=parse(renamed,str(route))['facts']
                stress.append({'row_position':row,'route':str(route),'class':int(label),
                    'partial_facts_identity_invariant':before==after,'legacy_model_view_identity_invariant':v0==v1,
                    'product_empty_invariant':True,'product_empty_scope':'R0 and partial parser never read product/vendor columns',
                    'absolute_clock_invariance_scope':'R0 uses named clock spans; arbitrary corrupt/unrecognized clocks are not guaranteed',
                    'original_model_view_sha256':hashlib.sha256(v0.encode()).hexdigest(),'renamed_model_view_sha256':hashlib.sha256(v1.encode()).hexdigest()})
        offset+=len(raw)
        if offset%262144==0:emit(stage='partial_fact_audit',rows=offset,seconds=round(time.monotonic()-start,1))
    assert offset==len(r) and {c['row_position'] for c in hardcases}==hard
    for c in hardcases:
        if c['route']=='asa':assert c['extracted']['facts']['src_port_fixed']==65536 and c['extracted']['facts']['dst_port_fixed']==55292
        else:assert c['extracted']['facts']['event_code_observed']==4672 and c['extracted']['facts']['privilege_SeDebugPrivilege']
    # Disagreement is explicit; never silently choose one raw observation for a shared projection.
    for pid,keys in ambiguity.items():
        for k in keys:supp[pid].pop(k,None)
    pd.DataFrame({'projection_id':np.arange(len(p)),'supplement':[json.dumps(a or {},sort_keys=True) for a in supp],
        'ambiguous_fields':[json.dumps(sorted(ambiguity.get(i,[]))) for i in range(len(p))]}).to_parquet(DEST/'partial_facts.parquet',index=False)
    selected=np.zeros(len(r),bool);selected[sel]=True
    cells=[]
    for (route,cls),stat in audit.items():
        mask=r.route.eq(route).to_numpy()&(y==cls)
        cells.append({'route':route,'class':cls,**dict(stat),'fit_rows':int((mask&fit).sum()),'main_loss_rows':int((mask&selected).sum()),
            'main_loss_all_role_M_S':bool(not cls or np.array_equal(mask&selected,mask&fit)),
            'prior_auxiliary_scope':'ASA closed deny only' if route=='asa' else 'not covered by prior ASA-only policy'})
    pd.DataFrame(cells).sort_values(['route','class']).to_csv(DEST/'format_supervision.csv',index=False)
    save(DEST/'hard_m_facts.json',hardcases);save(DEST/'input_stress.json',stress)
    # Eligibility is selected from TRAINING metadata before any fit; no prediction errors used.
    fields=['action','outcome','transport_protocol','src_role','dst_role','dst_port_fixed']
    keys=[tuple(str(f[k]) for k in fields) if all(k in f for k in fields) else None for f in facts]
    rowkey=pd.Series(keys,dtype=object).to_numpy()[r.projection_id.to_numpy()]
    asa_train=selected&r.route.eq('asa').to_numpy();pools=collections.defaultdict(lambda:collections.defaultdict(set))
    for i in np.flatnonzero(asa_train):
        if rowkey[i] is not None:pools[rowkey[i]][int(y[i])].add(int(r.component.iloc[i]))
    candidates=sorted([k for k,d in pools.items() if len(d[1])>=3 and len(d[2])>=3],key=keyhash)
    chosen=[];components=set();rejected=[]
    for k in candidates:
        if len(chosen)==3:break
        d=pools[k];possible=[]
        for cm in sorted(d[1],key=lambda v:keyhash(('M',k,v))):
            for cs in sorted(d[2],key=lambda v:keyhash(('S',k,v))):
                proposed=components|{cm,cs}
                if all(len(d[c]-proposed)>=2 for c in [1,2]):possible.append((cm,cs));break
            if possible:break
        if not possible:rejected.append({'key':k,'reason':'not_enough_components_after_joint_holdout'});continue
        # An equal row-count, same-class, same-component removal must be available.
        unionkeys=set(chosen+[k]);proposed=components|set(possible[0]);trialtrain=selected&~r.component.isin(proposed).to_numpy()
        targets=np.asarray([a in unionkeys for a in rowkey],bool)
        enough=True
        for cls in [1,2]:
            take=trialtrain&r.route.eq('asa').to_numpy()&(y==cls)
            required=r.component[take&targets].value_counts()
            available=r.component[take&~targets].value_counts()
            if any(available.get(comp,0)<n for comp,n in required.items()):enough=False;break
        if enough:chosen.append(k);components=proposed
        else:rejected.append({'key':k,'reason':'same_component_same_class_matched_control_unavailable'})
    spec={'candidate_keys':candidates,'chosen_keys':chosen,'holdout_components':sorted(components),'rejected':rejected,
        'groups_chosen_by':'sha256 of observed training-only fine key; up to3; holdout M/S components hashed; >=2 remaining same-class components; exact per-component matching required',
        'feasible':bool(chosen),'source_sha256':sha(__file__)}
    if chosen:
        targets=np.asarray([a in set(chosen) for a in rowkey],bool);train=selected&~r.component.isin(components).to_numpy()
        holdout=fit&r.component.isin(components).to_numpy();target_eval=holdout&targets&r.route.eq('asa').to_numpy()
        assert all(len(set(r.component[train&targets&(y==c)])-components)>=2 for c in [1,2])
        masks={'full':train.copy()};rng=np.random.default_rng(8901)
        for cls in [1,2]:
            remove=train&targets&r.route.eq('asa').to_numpy()&(y==cls);control=np.zeros(len(r),bool)
            for component,n in r.component[remove].value_counts().sort_index().items():
                pool=np.flatnonzero(train&~targets&r.route.eq('asa').to_numpy()&(y==cls)&r.component.eq(component).to_numpy())
                control[rng.choice(pool,int(n),replace=False)]=True
            assert remove.sum()==control.sum() and np.array_equal(np.bincount(r.component[remove]),np.bincount(r.component[control]))
            masks['withdraw_'+str(cls)]=train&~remove;masks['control_'+str(cls)]=train&~control
        for name,m in masks.items():np.save(DEST/('support_'+name+'_rows.npy'),np.flatnonzero(m).astype(np.int32))
        np.save(DEST/'support_evaluation_rows.npy',np.flatnonzero(holdout).astype(np.int32))
        np.save(DEST/'support_target_rows.npy',np.flatnonzero(target_eval).astype(np.int32))
        spec.update(training_counts={k:np.bincount(y[m],minlength=3).tolist() for k,m in masks.items()},
            evaluation_class_counts=np.bincount(y[holdout],minlength=3).tolist(),target_evaluation_class_counts=np.bincount(y[target_eval],minlength=3).tolist())
    save(DEST/'support_design.json',spec)
    result={'status':'preparation_complete','new_classifier_fits':0,'source_sha256':sha(__file__),'semantic_checks':20,
        'original_label_rows_checked':len(r),'route_count':int(r.route.nunique()),'route_class_cells':len(cells),
        'partial_facts_projection_count':len(p),'ambiguous_projection_count':len(ambiguity),
        'hard_m_cases_facts_observed':len(hardcases),'support_design_feasible':spec['feasible'],
        'stress_examples':len(stress),'legacy_identity_view_failures':sum(not a['legacy_model_view_identity_invariant'] for a in stress),
        'partial_fact_identity_failures':sum(not a['partial_facts_identity_invariant'] for a in stress),
        'seconds':time.monotonic()-start,'scope':'Original rows checked, partial facts extracted, all actual routes/class cells counted; no semantic completeness or classifier improvement claim.'}
    save(DEST/'preparation.json',result);emit(**result)


if __name__=='__main__':main()
