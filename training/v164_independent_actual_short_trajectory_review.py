"""Independent original-gold audit of actual V164 saved trajectories, CPU only."""
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz

from v160_independent_fixed_diagnostic_review import read, sha, rows_review
from v160_margin_normal import input_identity
from v160_active_margin_direction_v5 import solve_direction
from v161_independent_all_finite_results_review import close, target_risk, parameter_hash, gradient_repeat
from v163_one_sided_joint_restoration import propose
from v163_independent_readout_override_bound_review import review as readout_review

ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
COHORT=ROOT/'artifacts/v161_independent_frozen_error_cohort_review_20261002'
OUT=ROOT/'artifacts/v164_independent_actual_short_trajectory_review_20261002'
EPS=float(np.finfo(np.float64).eps)


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def tables(folder):
    return {s:pd.read_parquet(folder/f'{s}_original_rows.parquet') for s in ['OOF','deployment']}


def proposal_review(path,baseline,targets,gradients,state,gold):
    report=read(path/'probe.json'); direction=np.load(path/'direction.npy'); step=report['step']
    assert step in [1.,1/16] and report['actual_parameter_change']
    assert parameter_hash(state,direction,step)==report['probe_parameter_sha256']
    slopes=np.array([math.fsum(g*direction) for g in gradients]);close(slopes,report['class_slopes'])
    frames={};metrics={}
    for scope in baseline:
        frame,fullrisk,_,met=rows_review(path,scope,baseline[scope],gold)
        frames[scope]=frame;metrics[scope]=met
        close(np.exp(frame[['logp0','logp1','logp2']].to_numpy()),frame[['p0','p1','p2']].to_numpy())
        for key in ['original_rows','M_errors','S_errors','pure_errors','total_errors','protected_regressions','new_errors_vs_initial','repairs_vs_initial']:
            assert met[key]==report[scope+'_stats'][key]
        if scope=='OOF':
            after=target_risk(frame,targets);close(after,np.load(path/'fixed_error_risk.npy'))
            close(fullrisk,np.load(path/'full_original_class_risk.npy'))
    before=target_risk(baseline['OOF'],targets);old=baseline['OOF']
    counts=all(metrics['OOF'][key]<=int((old.pred.ne(old.truth)&old.truth.eq(c)).sum())
               for key,c in [('M_errors',1),('S_errors',2)])
    guard=(counts and metrics['OOF']['protected_regressions']==0 and metrics['deployment']['new_errors_vs_initial']==0
           and report['deployment_stats']['mastered'] and report['joint_TRAIN_retention']['passed'])
    assert counts==report['full_original_M_S_error_count_guard'] and guard==report['classification_guard']
    resolution=16*EPS*np.maximum(1.,np.maximum(abs(before),abs(after)))
    drop=before-after;slack=before+1e-4*step*slopes-after
    accepted=bool(guard and np.all(slopes<0) and np.all(drop>resolution) and np.all(slack>resolution))
    assert accepted==report['accepted']==report['finite_error_target_review']['accepted']
    actual=pd.read_parquet(path/'actual_blocking_original_rows.parquet');expected=[]
    for scope,frame in frames.items():
        protected=frame.protected_correct if scope=='OOF' else frame.initial_correct
        expected += [(scope,int(row.row_position),int(row.local),int(row.truth),int(row.pred))
                     for row in frame[protected&frame.pred.ne(frame.truth)].itertuples()]
    assert sorted(expected)==sorted((row.scope,int(row.row_position),int(row.local),int(row.truth),int(row.rival)) for row in actual.itertuples())
    return dict(path=path.relative_to(ROOT).as_posix(),accepted=accepted,metrics=metrics,fixed_target_drop=drop.tolist()),frames


def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/fit.json').exists() for r in range(3))
    OUT.mkdir();planpath=ROOT/'training/review_policy/v164_short_supervised_trajectory_contract.json';plan=read(planpath)
    goldpath=ROOT/'data/official/train.parquet';gold=pd.read_parquet(goldpath,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert np.bincount(gold,minlength=3).tolist()==[1899723,111728,45420]
    paths={p for p in TRIAL.rglob('*') if p.is_file()}|{Path(__file__).resolve(),planpath,goldpath}
    paths|={ROOT/'training'/name for name in ['v160_independent_fixed_diagnostic_review.py','v160_margin_normal.py',
        'v160_active_margin_direction_v5.py','v161_independent_all_finite_results_review.py',
        'v163_one_sided_joint_restoration.py','v163_independent_readout_override_bound_review.py']}
    for role in range(3):
        paths|={ROOT/f'artifacts/v159_class_boundary_numeric_trial_20261002/fold{role}_B/endpoint.pt',
                COHORT/f'role{role}/fixed_pure_error_targets.parquet',
                ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy',
                ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/deployment_probabilities.npy'}
        for c in [1,2]:paths.add(ROOT/f'artifacts/v161_fixed_error_endpoint_diagnostic_20261002/role{role}/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy')
    paths.add(ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz')
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    save(OUT/'pre_review_bindings.json',dict(source_sha256=bindings,official_calls=0))
    x=load_npz(ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz').tocsr();x.sort_indices();roles=[]
    for role,spec in enumerate(plan['roles']):
        folder=TRIAL/f'role{role}';fit=read(folder/'fit.json');assert fit['exception'] is None
        initial=torch.load(ROOT/f'artifacts/v159_class_boundary_numeric_trial_20261002/fold{role}_B/endpoint.pt',weights_only=True,map_location='cpu')['state']
        assert parameter_hash(initial)==fit['initial_parameter_sha256'];state=initial
        baseline=tables(folder/'baseline');current=baseline;targets=pd.read_parquet(COHORT/f'role{role}/fixed_pure_error_targets.parquet').row_position
        for scope,frame in baseline.items():rows_review(folder/'baseline',scope,frame,gold)
        assert np.array_equal(targets,baseline['OOF'].loc[baseline['OOF'].pure_current_input&baseline['OOF'].pred.ne(baseline['OOF'].truth),'row_position'])
        gs=[np.load(ROOT/f'artifacts/v161_fixed_error_endpoint_diagnostic_20261002/role{role}/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1,2]]
        opinions={s:np.asarray(np.load(ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{s}_probabilities.npy',mmap_mode='r')[:,:16],np.float64) for s in baseline}
        prior=np.log(np.maximum(opinions['OOF'].mean(1),1e-12));points=[];all_proposals=[]
        accepted=int(fit['permanent_updates']);assert 0<=accepted<=10
        candidate_paths=[folder/'initial_cached_candidate_replay'] if accepted else []
        for index in range(1,accepted+1):
            if index>1:
                previous_point=folder/f'parameter_point{index-1}'
                paths_here=sorted(previous_point.rglob('probe.json'))
                selected=[p.parent for p in paths_here if read(p)['accepted']];assert len(selected)==1
                candidate_paths.append(selected[0])
            path=candidate_paths[-1];review,frames=proposal_review(path,current,targets,gs,state,gold);assert review['accepted']
            target=folder/f'accepted{index}';receipt=read(target/'commit.json');checkpoint=torch.load(target/'checkpoint.pt',weights_only=True,map_location='cpu')['state']
            assert receipt['state_index']==index and receipt['previous_parameter_sha256']==parameter_hash(state)
            assert parameter_hash(checkpoint)==receipt['parameter_sha256']==read(path/'probe.json')['probe_parameter_sha256']
            displacement=np.load(path/'direction.npy')*read(path/'probe.json')['step'];offset=0
            for key,value in state.items():
                expected=value.numpy()+displacement[offset:offset+value.numel()].reshape(tuple(value.shape));offset+=value.numel()
                assert np.array_equal(expected,checkpoint[key].numpy())
            assert offset==1060832
            committed=tables(target);old=current['OOF'];after=frames['OOF']
            repairs=old.pred.ne(old.truth)&after.pred.eq(after.truth);mask=old.protected_correct.to_numpy()|repairs.to_numpy()
            repair_table=pd.read_parquet(target/'newly_repaired_original_rows.parquet')
            assert np.array_equal(repair_table.row_position,old.loc[repairs,'row_position'])
            assert receipt['newly_repaired_original_rows']==int(repairs.sum())
            assert receipt['newly_repaired_mixed_original_rows']==int((repairs&~old.pure_current_input).sum())
            assert receipt['cumulative_protected_original_rows']==int(mask.sum())
            for scope,frame in committed.items():
                reference=frames[scope].copy()
                if scope=='OOF':reference['protected_correct']=mask
                rows_review(target,scope,reference,gold)
                assert np.array_equal(frame.pred,reference.pred)
                assert np.array_equal(frame[['p0','p1','p2']].to_numpy(),reference[['p0','p1','p2']].to_numpy())
            protection=pd.read_parquet(target/'cumulative_correct_protection.parquet')
            assert np.array_equal(protection.row_position,old.loc[mask,'row_position'])
            current=committed;state=checkpoint;point=folder/f'parameter_point{index}';gradient_pairs=[]
            for c in [1,2]:
                pair=[np.load(point/f'class{c}_repeat{k}/complete_fixed_error_target_gradient.npy') for k in [0,1]]
                gradient_repeat(*pair);gradient_pairs.append(pair[0])
                for k in [0,1]:
                    identity=read(point/f'class{c}_repeat{k}/parameter_point.json')
                    assert identity['parameter_sha256']==parameter_hash(state)
                    observed,_,_,_=rows_review(point/f'class{c}_repeat{k}','OOF',current['OOF'],gold)
                    close(observed[['p0','p1','p2']].to_numpy(),current['OOF'][['p0','p1','p2']].to_numpy())
                    close(observed[['logp0','logp1','logp2']].to_numpy(),current['OOF'][['logp0','logp1','logp2']].to_numpy())
                    close(target_risk(current['OOF'],targets),np.load(point/f'class{c}_repeat{k}/fixed_pure_error_contribution.npy'))
            gs=gradient_pairs
            points.append(dict(index=index,parameter_sha256=parameter_hash(state),proposal=review,
                readout=readout_review(current['OOF'],prior,state['output_weight'].numpy())))
        # Every new-origin proposal, including rejected attempts and final stop.
        for point in sorted(folder.glob('parameter_point*'),key=lambda p:int(p.name.replace('parameter_point',''))):
            index=int(point.name.replace('parameter_point',''));start=folder/f'accepted{index}';point_state=torch.load(start/'checkpoint.pt',weights_only=True,map_location='cpu')['state'];point_rows=tables(start)
            point_g=[np.load(point/f'class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1,2]]
            if (point/'base_direction_QP_review.json').exists():
                result=solve_direction(*point_g,np.empty((0,1060832)))
                assert {k:v for k,v in result.items() if k!='direction'}==read(point/'base_direction_QP_review.json')
                assert np.array_equal(result['direction'],np.load(point/'base_direction.npy'))
            basepath=point/'base_finite_proposal'
            if not (basepath/'probe.json').exists():continue
            item,baseframes=proposal_review(basepath,point_rows,targets,point_g,point_state,gold);all_proposals.append(item)
            logs={s:np.load(basepath/f'{s}_logq.npy') for s in point_rows};u=np.load(basepath/'direction.npy')/16
            for restoration in sorted(point.glob('restoration*'),key=lambda p:int(p.name.replace('restoration',''))):
                refs=read(restoration/'active_normal_references.json');normals=[]
                b=np.load(restoration/'base_margins.npy');c=np.load(restoration/'actual_current_margins.npy')
                assert np.array_equal(u,np.load(restoration/'current_displacement.npy'))
                for j,(identity,ref) in enumerate(refs.items()):
                    meta=ref['metadata'];assert meta['base_parameter_sha256']==parameter_hash(point_state)
                    scope=meta['scope'];local=meta['local'];truth=meta['truth'];rival=meta['rival'];ids=np.sort(point_rows[scope].local.unique()) if scope=='OOF' else np.arange(22546)
                    pos=int(np.searchsorted(ids,local));chunk=ids[pos//2048*2048:pos//2048*2048+2048];query=pos%2048
                    assert input_identity(role,scope,chunk,x[chunk],opinions[scope][chunk],query,truth,rival)==identity
                    nf=ROOT/ref['gradient'];pair=[np.load(nf.parent/f'repeat{k}_gradient.npy') for k in [0,1]];gradient_repeat(*pair);normals.append(pair[0])
                    assert read(nf.parent/'input_binding.json')==meta
                    origin_lp=np.load(start/f'{scope}_logq.npy')
                    assert b[j]==origin_lp[local,truth]-origin_lp[local,rival] and c[j]==logs[scope][local,truth]-logs[scope][local,rival]
                result=propose(u,np.stack(normals),b,c,*point_g)
                assert {k:v for k,v in result.items() if k not in ['displacement','correction']}==read(restoration/'original_unit_restoration_review.json')
                for key in ['displacement','correction']:
                    if key in result:assert np.array_equal(result[key],np.load(restoration/f'{key}.npy'))
                fp=restoration/'finite_probe'
                if (fp/'probe.json').exists():
                    item,_=proposal_review(fp,point_rows,targets,point_g,point_state,gold);all_proposals.append(item)
                    u=result['displacement'];logs={s:np.load(fp/f'{s}_logq.npy') for s in point_rows}
        endpoint_state=torch.load(folder/'endpoint.pt',weights_only=True,map_location='cpu')['state']
        assert parameter_hash(state)==parameter_hash(endpoint_state)==fit['endpoint_parameter_sha256']
        for scope in current:
            observed,_,_,_=rows_review(folder/'endpoint',scope,current[scope],gold)
            close(observed[['p0','p1','p2']].to_numpy(),current[scope][['p0','p1','p2']].to_numpy())
            close(observed[['logp0','logp1','logp2']].to_numpy(),current[scope][['logp0','logp1','logp2']].to_numpy())
        events=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()];counts={}
        for kind,prefix in [('head','head'),('feature','feature'),('fixed_error_target_gradient','gradient'),('full_parameter_margin_gradient','margin')]:
            for event,suffix in [('attempt','attempts'),('completed','completed')]:
                found=[e for e in events if e['kind']==kind and e['event']==event]
                assert [e['ordinal'] for e in found]==list(range(1,len(found)+1));counts[prefix+'_'+suffix]=len(found)
        assert counts==fit['counts'] and counts['gradient_attempts']==counts['gradient_completed']==4*accepted
        assert counts['head_attempts']==counts['head_completed']==counts['feature_attempts']==counts['feature_completed']
        assert counts['head_attempts']==(2+fit['finite_proposals'])*(spec['OOF_chunks']+12)+counts['gradient_attempts']*spec['OOF_chunks']+counts['margin_attempts']<=spec['head_cap']
        assert counts['margin_attempts']==counts['margin_completed']<=spec['margin_gradient_cap']
        assert len(all_proposals)+int((folder/'initial_cached_candidate_replay/probe.json').exists())==fit['finite_proposals']
        direction_reports=list(folder.glob('parameter_point*/base_direction_QP_review.json'))
        restoration_reports=list(folder.glob('parameter_point*/restoration*/original_unit_restoration_review.json'))
        qp_reports=[read(p) for p in restoration_reports if 'optimizer_iterations' in read(p)]
        assert len(direction_reports)==fit['direction_QP_solves']<=9
        assert len(restoration_reports)==fit['restoration_solves']<=54
        assert len(qp_reports)==fit['restoration_QP_solves']<=54
        assert math.fsum(p['optimizer_iterations'] for p in qp_reports)==fit['restoration_optimizer_iterations']
        qp_calls=[e for e in events if e['kind']=='restoration_QP' and e['event']=='attempt']
        qp_returns=[e for e in events if e['kind']=='restoration_QP' and e['event']=='returned']
        assert [e['ordinal'] for e in qp_calls]==[e['ordinal'] for e in qp_returns]==list(range(1,len(qp_reports)+1))
        assert math.fsum(e['optimizer_iterations'] for e in qp_returns)==fit['restoration_optimizer_iterations']
        updates=[e for e in events if e['kind']=='permanent_update' and e['event']=='committed'];assert len(updates)==accepted
        assert [e['parameter_sha256'] for e in updates]==[p['parameter_sha256'] for p in points]
        assert len({p['parameter_sha256'] for p in points})==accepted
        initial_repair=current['OOF'].pred.eq(current['OOF'].truth)&baseline['OOF'].pred.ne(baseline['OOF'].truth)
        initial_regression=current['OOF'].pred.ne(current['OOF'].truth)&baseline['OOF'].pred.eq(baseline['OOF'].truth)
        last5=fit['actual_accepted_states'][-5:]
        mastery=accepted>=5 and len({s['parameter_sha256'] for s in last5})==5 and all(s['OOF']['mastered'] for s in last5)
        assert mastery==fit['last_five_actual_distinct_states_mastered']
        roles.append(dict(role=role,status=fit['status'],accepted_updates=accepted,counts=counts,points=points,
            proposals=all_proposals,repairs_vs_fixed_endpoint=int(initial_repair.sum()),new_errors_vs_fixed_endpoint=int(initial_regression.sum()),
            pure_endpoint_errors=int((current['OOF'].pred.ne(current['OOF'].truth)&current['OOF'].pure_current_input).sum()),
            last_five_actual_distinct_states_mastered=mastery))
    assert all(sha(ROOT/p)==value for p,value in bindings.items())
    report=dict(status='actual_short_trajectories_original_gold_gradients_functions_commits_and_costs_independently_verified',
                roles=roles,new_heads=sum(r['counts']['head_attempts'] for r in roles),new_target_gradients=sum(r['counts']['gradient_attempts'] for r in roles),
                new_margin_gradients=sum(r['counts']['margin_attempts'] for r in roles),new_fits=3,permanent_updates=sum(r['accepted_updates'] for r in roles),
                official_calls_by_this_review=0,training_issue_mastered=all(r['last_five_actual_distinct_states_mastered'] for r in roles),quality_acceptance=False,
                scope='Supervised training/development rows. Not full task scoring, independent transfer validation, or model promotion.')
    save(OUT/'review.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='roles'},ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
