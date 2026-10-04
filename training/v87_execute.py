"""Official-only, pre-registered 2x2 protection solver / M-S supervision trial."""
import argparse
import copy
import gc
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, read, save, sha, load_sparse
from v79_execute import rows
from v82_capacity import LAST, DEST as V82
from v85_protection import (Residual, csr_tensor, raw_counts, changes, cm_from_counts,
                            supervised_loss, protection_penalty, margin, inference,
                            sha_tensor_state, ALPHA)
from v81_training_contract import compare
from v87_solver import MarginJacobian, project_step
from v75_views import view

DEST = ROOT / 'artifacts/v87_solver_supervision_20260927'
OLD = ROOT / 'artifacts/v85_protection_20260927'
SEED = 8501
PAIR_SEED = 8701
EPOCHS = 60
CHECKPOINTS = [0,10,20,40,60]
ARMS = ['A0','A1','A2','A3']
FIELDS = ['action','outcome','transport_protocol','src_role','dst_role']
TOL = 1e-7
PAIR_WEIGHT = .05
TEMPERATURE = .1
MAX_ACTIVE = 256
MAX_CUTS = 16
MAX_TRUST = 16
QP_TOL = 1e-6
ARM_SECONDS = 1800
ASA = re.compile(r'\s*<ABSOLUTE_CLOCK>\s*<IDENTITY>\s+Deny\s+(tcp|udp)\s+src\s+([\w-]+):\s*<IDENTITY>\s*/(\d+)\s+dst\s+([\w-]+):\s*<IDENTITY>\s*/(\d+)\s+by\s+<IDENTITY>\s*-group\s+"[^"\r\n]*"\s+\[0x0,\s*0x0\]\s*',re.I)


def emit(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)


def data(h=1):
    r=rows(); y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy'); x=load_sparse(LAST/'X')
    selected=np.zeros(len(r),bool);selected[np.load(V82/'selected_rows.npy')]=True
    fit=~r.fold.isin([0,2,h]).to_numpy();inner=r.fold.eq(h).to_numpy();active=selected&fit
    assert np.array_equal(active&(y>0),fit&(y>0))
    cc=raw_counts(fid,y,active,x.shape[0]);full=raw_counts(fid,y,fit,x.shape[0]);vc=raw_counts(fid,y,inner,x.shape[0])
    return r,y,fid,x,cc,full,vc,fit,inner,np.flatnonzero(active)


def prepare_pairs(r,y,fid,cc,selected,folder):
    facts=[json.loads(t) for t in pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts]
    raw=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['message_sanitized'])
    asa=selected[(r.route.to_numpy()[selected]=='asa')&(y[selected]>0)]
    mixed=(cc>0).sum(1)>1
    records=[];audit=[]
    for row in asa:
        f=facts[int(r.projection_id.iloc[row])]
        body,_=view(raw.message_sanitized.iloc[row]);match=ASA.fullmatch(body)
        reason=None
        if mixed[fid[row]]:reason='encoded_input_has_multiple_train_classes'
        elif match is None:reason='raw_ASA_syntax_not_in_reviewed_deny_template'
        elif not all(k in f for k in FIELDS+['src_port_fixed','dst_port_fixed']):reason='required_facts_missing'
        elif (f['action']!='deny' or f['outcome']!='blocked'
              or f['transport_protocol']!=match[1].lower()
              or int(f['src_port_fixed'])!=int(match[3]) or int(f['dst_port_fixed'])!=int(match[5])):
            reason='raw_and_parsed_behavior_mismatch'
        if reason:
            audit.append({'row_position':int(row),'fid':int(fid[row]),'class':int(y[row]),'eligible':False,'reason':reason})
            continue
        # Canonical roles are used only for finding compatible real examples.
        # Raw interface spelling and both ports remain intact in model input.
        key=tuple(str(f[k]) for k in FIELDS)
        records.append({'row':int(row),'fid':int(fid[row]),'class':int(y[row]),'component':int(r.component.iloc[row]),
                        'coarse':key,'fine':key+(str(f['dst_port_fixed']),)})
    del raw
    pools={level:{} for level in ['fine','coarse']}
    for i,a in enumerate(records):
        for level in pools:pools[level].setdefault((a[level],a['class']),[]).append(i)
    rng=np.random.default_rng(PAIR_SEED)
    for pool in pools.values():
        for key,values in pool.items():rng.shuffle(values)
    pairs=[]
    for a in records:
        chosen=None
        for level in ['fine','coarse']:
            def candidate(cls):
                values=pools[level].get((a[level],cls),[])
                if not values:return None
                start=int(rng.integers(len(values)))
                for offset in range(len(values)):
                    b=records[values[(start+offset)%len(values)]]
                    if b['fid']!=a['fid'] and b['component']!=a['component']:return b
                return None
            positive=candidate(a['class']);negative=candidate(3-a['class'])
            if positive is not None and negative is not None:
                chosen=(level,positive,negative);break
        if chosen is None:
            audit.append({'row_position':a['row'],'fid':a['fid'],'class':a['class'],'eligible':False,'reason':'no_compatible_crosscomponent_positive_and_negative'})
            continue
        level,p,n=chosen
        pairs.append({'anchor_row':a['row'],'positive_row':p['row'],'negative_row':n['row'],
                      'anchor_fid':a['fid'],'positive_fid':p['fid'],'negative_fid':n['fid'],
                      'anchor_class':a['class'],'positive_class':p['class'],'negative_class':n['class'],
                      'anchor_component':a['component'],'positive_component':p['component'],'negative_component':n['component'],
                      'grade':level,'compatibility_key':json.dumps(a[level]),'raw_syntax_checked':True})
        audit.append({'row_position':a['row'],'fid':a['fid'],'class':a['class'],'eligible':True,'reason':level})
    p=pd.DataFrame(pairs);p.to_parquet(folder/'pairs.parquet',index=False)
    a=pd.DataFrame(audit);a.to_parquet(folder/'pair_coverage.parquet',index=False)
    assert len(p)>0 and set(p.anchor_class)=={1,2}
    selected_set=set(selected.tolist())
    for column in ['anchor_row','positive_row','negative_row']:assert set(p[column]).issubset(selected_set)
    assert (p.anchor_class==p.positive_class).all() and (p.anchor_class!=p.negative_class).all()
    assert (p.anchor_component!=p.positive_component).all() and (p.anchor_component!=p.negative_component).all()
    assert (p.anchor_fid!=p.positive_fid).all() and (p.anchor_fid!=p.negative_fid).all()
    save(folder/'pair_audit.json',{'asa_original_anchor_rows':len(asa),'reviewed_pairs':len(p),
        'by_class_grade':p.groupby(['anchor_class','grade']).size().reset_index(name='rows').to_dict('records'),
        'excluded_by_reason':a.loc[~a.eligible].groupby(['class','reason']).size().reset_index(name='rows').to_dict('records'),
        'raw_syntax_scope':'Every accepted row matched the closed ASA deny template and raw protocol/ports matched parsed facts. ACL policy and attack intent still unavailable; this is behavior compatibility, not a same-incident truth claim.',
        'primary_loss_rows_removed':0,'validation_rows_used':0,'pairs_sha256':sha(folder/'pairs.parquet')})
    # Original inner validation is stratified by support from training only.
    supports={}
    for row in selected:
        f=facts[int(r.projection_id.iloc[row])]
        if not all(k in f for k in FIELDS):continue
        key=tuple(str(f[k]) for k in FIELDS)+(str(f.get('dst_port_fixed','<missing>')),int(y[row]))
        supports.setdefault(key,set()).add(int(r.component.iloc[row]))
    bins=[]
    for row in np.flatnonzero(r.fold.eq(1).to_numpy()&(y>0)):
        f=facts[int(r.projection_id.iloc[row])]
        if not all(k in f for k in FIELDS):category='missing_behavior_facts';n=0
        else:
            key=tuple(str(f[k]) for k in FIELDS)+(str(f.get('dst_port_fixed','<missing>')),int(y[row]))
            n=len(supports.get(key,set()));category='same_class_multi_component' if n>=2 else 'same_class_single_component' if n==1 else 'same_class_zero_support'
        bins.append({'row_position':int(row),'class':int(y[row]),'route':str(r.route.iloc[row]),'support_components':n,'support_bin':category})
    pd.DataFrame(bins).to_parquet(folder/'inner_support_bins.parquet',index=False)


def prepare():
    if DEST.exists():raise FileExistsError(DEST)
    DEST.mkdir();folder=DEST/'fold1';folder.mkdir()
    r,y,fid,x,cc,full,vc,fit,inner,selected=data()
    old=np.load(OLD/'fold1/teacher_prediction.npy');z=np.load(OLD/'fold1/teacher_scores.npy',mmap_mode='r')
    assert read(OLD/'fold1/teacher_fit.json')['converged']
    protected=full[np.arange(len(old)),old];ids=np.flatnonzero(protected)
    margins=z[ids,old[ids]]-np.max(np.where(np.arange(3)[None,:]==old[ids,None],-np.inf,z[ids]),axis=1)
    assert (margins>0).all()
    assert int(protected.sum())==922009
    np.save(folder/'selected_rows.npy',selected.astype(np.int32));np.save(folder/'protection_fids.npy',ids.astype(np.int32))
    np.save(folder/'protection_original_mass.npy',protected[ids].astype(np.int32))
    np.save(folder/'protection_epsilon.npy',np.minimum(.001,margins/2))
    prepare_pairs(r,y,fid,full,selected,folder)
    save(folder/'exposure.json',{'train_population_rows':int(fit.sum()),'selected_original_rows':len(selected),
        'selected_per_class':cc.sum(0).tolist(),'full_per_class':full.sum(0).tolist(),
        'protected_original_rows':int(protected.sum()),'protected_unique_inputs':len(ids),
        'all_role_M_S_selected':bool(np.array_equal((y[selected]>0).sum(),(fit&(y>0)).sum())),
        'official_native_flow_M_preserved':int(((r.route=='native_flow')&(y==1)).sum()),
        'fit_native_flow_M_in_loss':int(((r.route=='native_flow')&(y==1)&fit).sum()),
        'teacher_reused_fold_local':True,'teacher_new_fit_count':0})
    role=np.where(r.fold.eq(0),'H',np.where(r.fold.eq(2),'C',np.where(inner,'inner','train')))
    for name in ['component','body_group','source_symbol']:
        take=r[name]>=0
        assert pd.DataFrame({'id':r.loc[take,name],'role':role[take]}).groupby('id').role.nunique().max()==1
    pd.DataFrame({'row_position':r.row_position,'fold':r.fold,'component':r.component,'role':role,
                  'selected':np.isin(r.row_position,selected)}).to_parquet(DEST/'manifest.parquet',index=False)
    emit(stage='prepared',exposure=read(folder/'exposure.json'),pairs=read(folder/'pair_audit.json'))


def register():
    assert not (DEST/'registration.json').exists()
    assert read(DEST/'mechanism_tests.json')['passed']
    folder=DEST/'fold1'
    files=[ROOT/'training/v87_execute.py',ROOT/'training/v87_solver.py',ROOT/'training/v87_mechanism_checks.py',ROOT/'training/test_v87_solver.py',
           ROOT/'docs/V86_MALICIOUS_REGRESSION_ROOT_CAUSE_AND_PLAN.md',ROOT/'data/official/train.parquet',
           OUT/'rows.parquet',OUT/'projections.parquet',ROOT/'training/v75_views.py',
           LAST/'X.data',LAST/'X.indices',LAST/'X.indptr',LAST/'row_feature_id.npy',V82/'selected_rows.npy',
           OLD/'fold1/teacher.joblib',OLD/'fold1/teacher_scores.npy',OLD/'fold1/teacher_prediction.npy',
           OLD/'fold1/teacher_fit.json',DEST/'manifest.parquet']+[p for p in folder.iterdir() if p.is_file()]
    prior={}
    for path in ['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json',
                 'evidence/2026-09-27/v81_diagnosis/delivery.json','evidence/2026-09-27/v82_capacity/delivery.json',
                 'evidence/2026-09-27/v83_root_review/delivery.json','evidence/2026-09-27/v84_preservation/delivery.json',
                 'evidence/2026-09-27/v85_protection/delivery.json','evidence/2026-09-27/v86_boundary_review/delivery.json']:
        prior.update(read(ROOT/path)['artifact_sha256'])
    changed=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not changed,changed
    save(DEST/'registration.json',{'version':'v87-actual-step-crosscomponent-1','source_sha256':sha(__file__),
        'source_bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
        'arms':{'A0':{'solver':'old_backtrack','contrastive':False},'A1':{'solver':'active_QP','contrastive':False},
                'A2':{'solver':'old_backtrack','contrastive':True},'A3':{'solver':'active_QP','contrastive':True}},
        'main_loss':'Unchanged exact original-frequency OVR BCE plus residual L2; historical selected-P hinge included in every arm.',
        'teacher':'Verified frozen v85 fold1 teacher reused; no new fit or held-out label exposure.',
        'protection':'All teacher-correct original train population, compressed by identical input; both competing classes checked.',
        'epsilon':'min(1e-3, original teacher positive margin/2)',
        'lr':.003,'alpha':ALPHA,'residual_dimensions':[66287,256,64,3],'residual_seed':SEED,
        'pair_seed':PAIR_SEED,'pair_weight':PAIR_WEIGHT,'pair_temperature':TEMPERATURE,
        'pair_loss':'Within-class mean 2-way positive-vs-negative cosine InfoNCE on hidden64; equal average of actual M/S anchor classes; same immutable pairs in A2/A3.',
        'max_attempts':EPOCHS,'checkpoints':CHECKPOINTS,'early_stop_snapshot':True,
        'max_active_constraints':MAX_ACTIVE,'max_cuts':MAX_CUTS,'max_trust_halvings':MAX_TRUST,
        'QP_tolerance':QP_TOL,'nonlinear_margin_tolerance':TOL,'QP_max_iterations':20000,
        'max_seconds_per_arm':ARM_SECONDS,'no_progress_limit':3,
        'optimizer_state':'Retain Adam proposal moments after accepted projection/interpolation; restore full state on rejection. Persist model, optimizer, CPU/CUDA RNG at snapshots.',
        'primary_fold':1,'rotation_folds':[3,4],'locked_folds':[2,0],
        'selection':'Saved fixed/early-stop states: full-train protection+zero negative flips and real repairs on selected-fit/inner, exact per-class guard; choose inner errors then earlier epoch then arm. No C/H selection.',
        'continuation':'All four arms execute regardless of other-arm quality; eligible winner only then rotates at fixed selected epoch with a fold-local teacher. Full task final fitting remains gated on rotation/C/H real regression.',
        'support_diagnostics':'Fixed training-only support bins scored after selection; no support-withdrawal refits are authorized before primary candidate acceptance.',
        'development_not_blind':True,'external_data':False,'pseudo_labels':False,
        'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,
        'runtime':{'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'osqp':__import__('osqp').__version__}})
    emit(stage='registered',arms=ARMS,old_files_verified=len(prior))


def check():
    reg=read(DEST/'registration.json')
    for name in ['training/v87_execute.py','training/v87_solver.py']:
        assert sha(ROOT/name)==reg['source_bindings'][name],name
    return reg


def hidden(model,xt):
    h=torch.sparse.mm(xt,model.first.weight.T)
    return torch.nn.functional.gelu(model.second(torch.nn.functional.gelu(h,approximate='tanh')),approximate='tanh')


def contrastive_loss(h,pair_indices,pair_labels):
    h=torch.nn.functional.normalize(h,dim=1,eps=1e-8)
    a,p,n=[h[ix] for ix in pair_indices]
    logits=torch.stack([(a*p).sum(1),(a*n).sum(1)],dim=1)/TEMPERATURE
    values=torch.nn.functional.cross_entropy(logits,torch.zeros(len(a),dtype=torch.long,device=h.device),reduction='none')
    return (values[pair_labels==1].mean()+values[pair_labels==2].mean())/2


class Protection:
    def __init__(self,x,z0,old,folder):
        self.ids=np.load(folder/'protection_fids.npy');self.x=x[self.ids];self.xt=csr_tensor(self.x)
        self.labels=torch.as_tensor(old[self.ids].astype(np.int64),device='cuda')
        self.z0=torch.as_tensor(np.asarray(z0[self.ids]),dtype=torch.float64,device='cuda')
        self.eps=torch.as_tensor(np.load(folder/'protection_epsilon.npy'),dtype=torch.float64,device='cuda')
        self.mass=torch.as_tensor(np.load(folder/'protection_original_mass.npy'),device='cuda')
        self.old_margin=margin(self.z0,self.labels)

    def evaluate(self,model):
        with torch.no_grad():
            z=self.z0+model(self.xt).double();true=z.gather(1,self.labels[:,None])
            gaps=true-z;gaps.scatter_(1,self.labels[:,None],torch.inf)
            deficit=self.eps[:,None]-gaps
            wrong=z.argmax(1)!=self.labels
            bad=(deficit>TOL)
            # Decision arithmetic is checked even where teacher margins are tiny.
            bad|=wrong[:,None]&(gaps<=0)
            ids,other=torch.nonzero(bad,as_tuple=True)
            minimum=gaps.min(1).values
            rec={'violating_input_groups':int(bad.any(1).sum()),'violating_original_rows':int(self.mass[bad.any(1)].sum()),
                 'protected_negative_flips':int(self.mass[wrong].sum()),
                 'max_violation':float(deficit.clamp_min(0).max()),'min_protected_margin':float(minimum.min())}
            return rec,(ids.cpu().numpy(),other.cpu().numpy()),gaps.cpu().numpy()

    def margins_report(self,model):
        rec,_,gaps=self.evaluate(model);old=self.old_margin.cpu().numpy();new=gaps.min(1)
        labels=self.labels.cpu().numpy();mass=self.mass.cpu().numpy()
        rec['by_class']=[]
        for cls in range(3):
            take=labels==cls
            rec['by_class'].append({'class':cls,'protected_original_rows':int(mass[take].sum()),
                 'unique_inputs':int(take.sum()),'margin_quantiles':np.quantile(new[take],[0,.01,.1,.5,1]).tolist(),
                 'old_margin_retention_quantiles':np.quantile(new[take]/old[take],[0,.01,.1,.5,1]).tolist(),
                 'retention_below_1_percent_original_rows':int(mass[take&(new/old<.01)].sum())})
        return rec


def assign(model,base,direction,fraction=1.):
    with torch.no_grad():
        for p,a,d in zip(model.parameters(),base,direction):p.copy_(a+fraction*d)


def projected_update(model,base,proposal,protection):
    direction=[b-a for a,b in zip(base,proposal)];active=[];seen=set();logs=[]
    assign(model,base,[torch.zeros_like(t) for t in base])
    _,_,basegaps=protection.evaluate(model)
    for trust in range(MAX_TRUST+1):
        current=[d*(2.**-trust) for d in direction]
        for cut in range(MAX_CUTS):
            if active:
                assign(model,base,[torch.zeros_like(t) for t in base])
                ids=np.array([i for i,o in active]);rivals=np.array([o for i,o in active])
                classes=protection.labels[ids].cpu().numpy()
                jac=MarginJacobian(model,protection.x[ids],classes,rivals)
                bound=protection.eps[ids].cpu().numpy()-basegaps[ids,rivals]
                proposed=[d*(2.**-trust) for d in direction]
                current,info=project_step(jac,proposed,bound,QP_TOL)
                info.update(trust_half_count=trust,cut=cut,active_boundaries=[{'fid':int(protection.ids[i]),'true':int(c),'other':int(o)} for (i,o),c in zip(active,classes)])
                logs.append(info);del jac
                if current is None:
                    assign(model,base,[torch.zeros_like(t) for t in base])
                    return False,0.,logs,{'reason':'QP_or_KKT_failed','last_QP':info}
            assign(model,base,current);feas,bad,_=protection.evaluate(model)
            if feas['violating_input_groups']==0:
                return True,2.**-trust,logs,None
            additions=[(int(i),int(o)) for i,o in zip(*bad) if (int(i),int(o)) not in seen]
            if not additions:break  # Existing constraints: reduce trust and re-solve.
            # Never silently truncate constraints: exceeding the registered cap
            # is an explicit resource stop, with the last feasible model restored.
            if len(active)+len(additions)>MAX_ACTIVE:
                assign(model,base,[torch.zeros_like(t) for t in base])
                return False,0.,logs,{'reason':'active_constraint_cap','required':len(active)+len(additions),'cap':MAX_ACTIVE}
            active.extend(additions);seen.update(additions)
    assign(model,base,[torch.zeros_like(t) for t in base])
    return False,0.,logs,{'reason':'nonlinear_trust_region_exhausted'}


def train(arm):
    reg=check();folder=DEST/'fold1'
    if (folder/(arm+'_fit.json')).exists():raise FileExistsError('Completed arm')
    r,y,fid,x,cc,full,vc,fit,inner,selected=data()
    used=np.flatnonzero(cc.sum(1));vused=np.flatnonzero(vc.sum(1))
    z0=np.load(OLD/'fold1/teacher_scores.npy',mmap_mode='r');old=np.load(OLD/'fold1/teacher_prediction.npy')
    xt=csr_tensor(x[used]);ct=torch.as_tensor(cc[used],dtype=torch.float32,device='cuda')
    zt=torch.as_tensor(np.asarray(z0[used]),dtype=torch.float32,device='cuda')
    labels=torch.as_tensor(old[used].astype(np.int64),device='cuda');pt=ct.gather(1,labels[:,None]).squeeze(1)
    eps=torch.minimum(torch.full_like(pt,.001),margin(zt,labels)/2)
    protection=Protection(x,z0,old,folder)
    pairs=pd.read_parquet(folder/'pairs.parquet')
    pi=[torch.as_tensor(np.searchsorted(used,pairs[name]),dtype=torch.long,device='cuda') for name in ['anchor_fid','positive_fid','negative_fid']]
    pl=torch.as_tensor(pairs.anchor_class.to_numpy(),device='cuda')
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=Residual().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=reg['lr'])
    initialhash=sha_tensor_state({k:t.detach().cpu() for k,t in model.state_dict().items()})
    assert initialhash==read(OLD/'fold1/C_fit.json')['initial_state_hash']
    trace=[];progress=[];saved=[];start=time.monotonic();stagnation=0;fatal=None
    def snapshot(epoch,early_stop=False):
        with torch.no_grad():delta=model(xt).cpu().numpy()
        tp=(np.asarray(z0[used])+delta).argmax(1).astype(np.int8)
        vp=(np.asarray(z0[vused])+inference(model,x,vused)).argmax(1).astype(np.int8)
        fm=changes(cc[used],old[used],tp);vm=changes(vc[vused],old[vused],vp)
        pr=protection.margins_report(model);guard=compare(cm_from_counts(vc,old),np.asarray(vm['cm']),True)
        eligible=pr['violating_input_groups']==0 and pr['protected_negative_flips']==0 and fm['negative_flips']==0 and fm['positive_flips']>0 and vm['negative_flips']==0 and vm['positive_flips']>0 and guard['eligible']
        name=f'{arm}_epoch{epoch:03}'
        state={'model':{k:t.detach().cpu().clone() for k,t in model.state_dict().items()},
               'optimizer':optimizer.state_dict(),'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
               'epoch':epoch,'arm':arm,'source_sha256':sha(__file__)}
        torch.save(state,folder/(name+'.pt'))
        np.savez_compressed(folder/(name+'_predictions.npz'),fit_feature_ids=used,fit_prediction=tp,inner_feature_ids=vused,inner_prediction=vp)
        item={'name':name,'epoch':epoch,'early_stop':early_stop,'selected_fit':fm,'inner':vm,'protection':pr,'eligible':bool(eligible),'inner_metric_guard':guard}
        trace.append(item);saved.append(name);save(folder/(arm+'_selection_trace.json'),trace)
        emit(stage='snapshot',arm=arm,epoch=epoch,fit_errors=fm['errors'],fit_positive=fm['positive_flips'],fit_negative=fm['negative_flips'],
             inner_errors=vm['errors'],inner_positive=vm['positive_flips'],inner_negative=vm['negative_flips'],eligible=bool(eligible))
    snapshot(0)
    emit(stage='arm_start',arm=arm,protected_inputs=len(protection.ids),pairs=len(pairs))
    for epoch in range(1,EPOCHS+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        h=hidden(model,xt);z=zt+model.last(h)
        main=supervised_loss(z,ct);l2=ALPHA/2*sum(p.square().sum() for p in model.parameters())
        contrast=contrastive_loss(h,pi,pl) if reg['arms'][arm]['contrastive'] else z.new_zeros(())
        multiplier=10. if epoch<=20 else 100. if epoch<=40 else 1000.
        hinge=protection_penalty(z,labels,eps,pt,multiplier)
        loss=main+l2+PAIR_WEIGHT*contrast+hinge;loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        base=[p.detach().clone() for p in model.parameters()];before=copy.deepcopy(optimizer.state_dict())
        rngbefore={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
        optimizer.step();proposal=[p.detach().clone() for p in model.parameters()]
        records=[];fraction=1.;stop=None;checks=0
        if reg['arms'][arm]['solver']=='active_QP':
            accepted,fraction,records,stop=projected_update(model,base,proposal,protection)
            checks=len(records)
        else:
            direction=[b-a for a,b in zip(base,proposal)];accepted=False
            for k in range(MAX_TRUST+1):
                fraction=2.**-k;assign(model,base,direction,fraction)
                pr,_,_=protection.evaluate(model);checks+=1
                if pr['violating_input_groups']==0:accepted=True;break
        if not accepted:
            assign(model,base,[torch.zeros_like(a) for a in base]);optimizer.load_state_dict(before)
            torch.set_rng_state(rngbefore['cpu']);torch.cuda.set_rng_state_all(rngbefore['cuda']);fraction=0.
        displacement=float(torch.sqrt(sum((p-a).square().sum() for p,a in zip(model.parameters(),base))))
        stagnation=stagnation+1 if not accepted or displacement<1e-9 else 0
        pr,_,_=protection.evaluate(model);assert pr['protected_negative_flips']==pr['violating_input_groups']==0
        with torch.no_grad():pred=(np.asarray(z0[used])+model(xt).cpu().numpy()).argmax(1)
        fm=changes(cc[used],old[used],pred)
        item={'epoch':epoch,'main_before':float(main),'contrastive_before':float(contrast),'hinge_before':float(hinge),
              'objective_before':float(loss),'accepted':accepted,'proposal_fraction':fraction,'parameter_displacement':displacement,
              'consecutive_no_progress':stagnation,'selected_fit_errors':fm['errors'],'selected_fit_positive':fm['positive_flips'],
              'selected_fit_negative':fm['negative_flips'],'protection':pr,'QP_records':records,'checks':checks,
              'seconds':time.monotonic()-start,'stop':stop}
        progress.append(item);save(folder/(arm+'_progress.json'),progress)
        emit(stage='epoch',arm=arm,epoch=epoch,fit_errors=fm['errors'],repairs=fm['positive_flips'],
             accepted=accepted,projection_solves=len(records),seconds=round(time.monotonic()-start,1),stop=stop and stop['reason'])
        if epoch in CHECKPOINTS:snapshot(epoch)
        if stop or stagnation>=3 or time.monotonic()-start>ARM_SECONDS:
            fatal=stop or {'reason':'three_consecutive_no_progress' if stagnation>=3 else 'registered_runtime_limit'}
            if epoch not in CHECKPOINTS:snapshot(epoch,True)
            break
        del base,before,proposal,h,z,loss,main,l2,contrast,hinge
    eligible=[s for s in trace if s['eligible']]
    chosen=min(eligible,key=lambda t:(t['inner']['errors'],t['epoch'])) if eligible else None
    report={'arm':arm,'actual_classifier_fits':1,'actual_calibration_fits':0,'attempts':len(progress),
            'accepted_updates':sum(i['accepted'] for i in progress),'stopped_reason':fatal,'saved_states':saved,
            'selected':None if chosen is None else chosen['name'],'initial_state_hash':initialhash,
            'seconds':time.monotonic()-start,'source_sha256':sha(__file__),
            'teacher_sha256':sha(OLD/'fold1/teacher.joblib'),'selection_frozen_before_C_H':True}
    save(folder/(arm+'_fit.json'),report);emit(stage='arm_complete',**report)
    del model,optimizer,protection,xt;gc.collect();torch.cuda.empty_cache()


def finish():
    check();folder=DEST/'fold1';candidates=[]
    for arm in ARMS:
        report=read(folder/(arm+'_fit.json'))
        if report['selected']:
            state=next(s for s in read(folder/(arm+'_selection_trace.json')) if s['name']==report['selected'])
            candidates.append({'arm':arm,'epoch':state['epoch'],'name':state['name'],'inner_errors':state['inner']['errors']})
    chosen=min(candidates,key=lambda d:(d['inner_errors'],d['epoch'],d['arm'])) if candidates else None
    save(DEST/'continuation.json',{'primary_selected':chosen,'rotation_authorized':chosen is not None,
         'reason':'eligible_primary' if chosen else 'no_saved_state_passed_full_train_P_and_inner_real_gain_with_zero_negative_flip',
         'selection_frozen_before_C_H':True,'final_full_fit_authorized':False,'C_H_used_for_selection':False,
         'support_withdrawal_refits_authorized':chosen is not None})
    emit(stage='primary_selection',selected=chosen)


def diagnose():
    check();assert (DEST/'continuation.json').exists()
    r,y,fid,x,cc,full,vc,fit,inner,selected=data();folder=DEST/'fold1'
    old=np.load(OLD/'fold1/teacher_prediction.npy');z0=np.load(OLD/'fold1/teacher_scores.npy',mmap_mode='r')
    roles={'fit_full':fit,'inner':inner,'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    bins=pd.read_parquet(folder/'inner_support_bins.parquet');allresults=[];casebooks=[]
    for arm in ARMS:
        report=read(folder/(arm+'_fit.json'));names=list(dict.fromkeys([report['saved_states'][-1]]+([] if report['selected'] is None else [report['selected']])))
        for name in names:
            state=torch.load(folder/(name+'.pt'),map_location='cuda',weights_only=True);model=Residual().cuda();model.load_state_dict(state['model'])
            p=np.empty(x.shape[0],np.int8)
            for start in range(0,len(p),8192):
                ids=np.arange(start,min(start+8192,len(p)));p[ids]=(np.asarray(z0[ids])+inference(model,x,ids)).argmax(1)
            np.save(folder/(name+'_all_prediction.npy'),p);new=p[fid];base=old[fid];rr=[];classwise=[]
            for role,mask in roles.items():
                counts=raw_counts(fid,y,mask,len(p));m=changes(counts,old,p);m['role']=role;rr.append(m)
                for route in sorted(r.route.unique()):
                    for cls in range(3):
                        take=mask&r.route.eq(route).to_numpy()&(y==cls)
                        classwise.append({'role':role,'route':route,'class':cls,'support':int(take.sum()),
                          'old_correct':int((take&(base==y)).sum()),'new_correct':int((take&(new==y)).sum()),
                          'repairs':int((take&(base!=y)&(new==y)).sum()),'regressions':int((take&(base==y)&(new!=y)).sum())})
                for kind,mark in [('repair',(base!=y)&(new==y)),('regression',(base==y)&(new!=y)),('remaining_old_error',(base!=y)&(new!=y))]:
                    take=np.flatnonzero(mask&mark)
                    z=r.iloc[take][['row_position','component','route','label_index','fold']].copy()
                    z['name']=name;z['role']=role;z['kind']=kind;z['fid']=fid[take];z['old']=base[take];z['new']=new[take];casebooks.append(z)
            pd.DataFrame(classwise).to_csv(folder/(name+'_classwise.csv'),index=False)
            b=bins.copy();pos=b.row_position.to_numpy();b['old_correct']=base[pos]==y[pos];b['new_correct']=new[pos]==y[pos]
            b['repair']=~b.old_correct&b.new_correct;b['regression']=b.old_correct&~b.new_correct
            b.groupby(['route','class','support_bin']).agg(rows=('row_position','size'),old_correct=('old_correct','sum'),new_correct=('new_correct','sum'),repairs=('repair','sum'),regressions=('regression','sum')).reset_index().to_csv(folder/(name+'_support_bins.csv'),index=False)
            allresults.append({'name':name,'selected':name==report['selected'],'roles':rr})
            emit(stage='locked_diagnostic',name=name,errors={q['role']:q['errors'] for q in rr},repairs={q['role']:q['positive_flips'] for q in rr},regressions={q['role']:q['negative_flips'] for q in rr})
            del state,model;torch.cuda.empty_cache()
    pd.concat(casebooks,ignore_index=True).to_parquet(DEST/'problem_casebook.parquet',index=False)
    save(folder/'diagnosis.json',allresults)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','register','train','finish','diagnose']);parser.add_argument('--arm',choices=ARMS)
    a=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with threadpool_limits(limits=4):
        if a.stage=='prepare':prepare()
        elif a.stage=='register':register()
        elif a.stage=='train':train(a.arm)
        elif a.stage=='finish':finish()
        else:diagnose()
