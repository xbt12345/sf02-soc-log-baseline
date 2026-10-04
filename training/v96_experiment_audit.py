"""Audit v95 paired populations, pseudo-behavior buckets, and actual meta gradient."""
import copy
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from v89_common import ROOT, OUT, read, save, sha, data, raw_counts
from v85_protection import changes
from v92_train import Branch, csr
from v95_four_arm import make_context
from v96_protocol_audit import DEST, V95


def gradient_vector(net):
    return torch.cat([p.grad.detach().flatten() for p in net.parameters()])


def main():
    assert not (DEST/'experiment_audit.json').exists()
    r,y,fid,_,old,_,fit=data()
    roles=pd.read_parquet(V95/'full_format_roles.parquet',columns=['role']).role.to_numpy()
    base=np.load(V95/'teacher_prediction.npy');new=np.load(V95/'E10_step200_prediction.npy')
    bp=base[fid];op=old[fid];npred=new[fid]
    flows=[]
    for role,mask in [('inner',r.fold.eq(1).to_numpy()),('C',r.fold.eq(2).to_numpy()),('H',r.fold.eq(0).to_numpy()),('fit',fit)]:
        for route in sorted(r.route[mask].unique()):
            take=mask & r.route.eq(route).to_numpy();o=(op==y);b=(bp==y);n=(npred==y)
            flows.append({'role':role,'route':route,'rows':int(take.sum()),
                'old_errors':int((take&~o).sum()),'A_teacher_errors':int((take&~b).sum()),'E10_errors':int((take&~n).sum()),
                'old_correct_lost_in_teacher_and_still_wrong':int((take&o&~b&~n).sum()),
                'old_correct_lost_by_branch':int((take&o&b&~n).sum()),
                'old_correct_teacher_lost_branch_restored':int((take&o&~b&n).sum()),
                'branch_corrected_teacher_error':int((take&~b&n).sum()),'branch_lost_teacher_correct':int((take&b&~n).sum())})
    pd.DataFrame(flows).to_csv(DEST/'teacher_branch_error_attribution.csv',index=False)
    aux=pd.read_parquet(V95/'auxiliary_rows.parquet')
    texts=pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    empty=aux[aux.behavior.eq('{}')].copy()
    empty['protocol_observed']=texts.reindex(r.new_text_id.iloc[empty.row_position].to_numpy()).str.extract(r'(?i)\bDeny\s+(\w+)',expand=False).fillna('unrecognized').to_numpy()
    buckets=empty.groupby(['role','label_index','protocol_observed']).agg(rows=('row_position','size'),components=('component','nunique')).reset_index()
    buckets.to_csv(DEST/'empty_bucket_protocol_composition.csv',index=False)
    weights=[];N=int(np.isin(roles,['A','B']).sum())
    for (behavior,role,cl),sl in aux[aux.role.isin(['A','B'])].groupby(['behavior','role','label_index']):
        weights.append({'behavior':behavior,'role':role,'class':int(cl),'rows':len(sl),'components':int(sl.component.nunique()),
                        'auxiliary_loss_weight_per_row':.01/(6*len(sl)),
                        'main_loss_weight_per_row':1/N,'auxiliary_to_main_per_row_ratio':.01*N/(6*len(sl))})
    save(DEST/'auxiliary_objective_weights.json',weights)
    # Measure local contribution of the virtual update; all real model weights remain frozen.
    ctx=make_context();device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    xt=csr(ctx['x'][ctx['used']],device)
    teacher=torch.as_tensor(np.asarray(ctx['zz'][ctx['train_ids']]),device=device,dtype=torch.float64)
    truth=torch.as_tensor(ctx['cc'],device=device,dtype=torch.float64)
    aw={s:torch.as_tensor(w,device=device,dtype=torch.float64) for s,w in ctx['aux_weights'].items()}
    results=[]
    def loss(net,weight):return (-torch.log_softmax(teacher+net(xt).double(),1)*weight).sum()
    for step in [0,50,200]:
        state=torch.load(V95/f'E01_step{step:03}_model.pt',map_location='cpu',weights_only=True)
        net=Branch(66287).to(device);net.load_state_dict(state['state_dict']);before={k:v.detach().clone() for k,v in net.state_dict().items()}
        net.zero_grad(set_to_none=True);main=loss(net,truth)/N;main.backward();gm=gradient_vector(net)
        for a,b in [('A','B'),('B','A')]:
            net.zero_grad(set_to_none=True);outer0=loss(net,aw[b]);outer0.backward();g0=gradient_vector(net)
            clone=copy.deepcopy(net);clone.zero_grad(set_to_none=True);inner=loss(clone,aw[a]);inner.backward()
            with torch.no_grad():
                for p in clone.parameters():p.add_(p.grad,alpha=-.01)
            clone.zero_grad(set_to_none=True);outer1=loss(clone,aw[b]);outer1.backward();g1=gradient_vector(clone)
            results.append({'checkpoint':step,'inner_role':a,'feedback_role':b,'main_gradient_norm':float(gm.norm().item()),
                'weighted_aux_to_main_gradient_norm_ratio':float((.01*g1.norm()/gm.norm()).item()),
                'virtual_update_relative_feedback_gradient_change':float(((g1-g0).norm()/g0.norm()).item()),
                'gradient_cosine_with_no_inner_step':float(torch.nn.functional.cosine_similarity(g1[None],g0[None]).item()),
                'feedback_loss_before_inner_step':float(outer0.item()),'feedback_loss_after_inner_step':float(outer1.item()),
                'scope':'Frozen local gradient comparison; not a trained eta=0 control or full trajectory equivalence'})
            assert all(torch.equal(v,before[k]) for k,v in net.state_dict().items())
            del clone
        del net
    save(DEST/'meta_gradient_diagnosis.json',results)
    prefreeze=[]
    for arm in ['E00','E10','E01','E11']:
        for rec in read(V95/f'{arm}_snapshots.json'):
            prefreeze.append({'arm':arm,'step':rec['step'],'locked_roles_already_in_snapshot':sorted(set(rec['roles'])&{'inner','C','H'})})
    result={'status':'implementation_and_design_audited_no_fit','teacher_branch_attribution':pd.DataFrame(flows).groupby('role').sum(numeric_only=True).reset_index().to_dict('records'),
        'empty_group_protocol_composition':buckets.to_dict('records'),'maximum_aux_to_main_per_row_ratio':max(w['auxiliary_to_main_per_row_ratio'] for w in weights),
        'gradient_relative_changes':[x['virtual_update_relative_feedback_gradient_change'] for x in results],
        'gradient_cosines':[x['gradient_cosine_with_no_inner_step'] for x in results],
        'snapshots_computing_locked_metrics_before_selection':len(prefreeze),
        'locked_labels_used_by_selection_formula':False,
        'loss_reporting_issue':'v95 main_loss is ASA contribution divided by full A+B rows; unchanged non-ASA constant is omitted, gradients unchanged. All-format branch learning is not established.',
        'new_classifier_fits':0,'new_calibration_fits':0,'source_sha256':sha(__file__)}
    save(DEST/'experiment_audit.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
