"""Two matched ASA-only branch fits; V and historic labels are not loaded."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, read, save, sha
from v92_train import Branch, csr, margin
from v97_prepare import DEST, SEED

V92=ROOT/'artifacts/v92_evidence_training_20260928'


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fit(arm):
    assert arm in ['ERM','MAG'] and not (DEST/f'{arm}_fit.json').exists()
    reg=read(DEST/'registration.json');teacher_fit=read(DEST/'teacher_fit.json')
    assert reg['source_sha256']==sha(ROOT/'training/v97_prepare.py')
    assert teacher_fit['converged'] and teacher_fit['source_sha256']==sha(ROOT/'training/v97_teacher.py')
    assert sha(DEST/'teacher.joblib')==teacher_fit['model_sha256']
    assert sha(DEST/'teacher_scores.npy')==teacher_fit['scores_sha256']
    with np.load(DEST/'AB_train_counts.npz') as z:raw=z['A'].astype(np.int32)+z['B'].astype(np.int32)
    assert int(raw.sum())==753709
    ids=np.load(V92/'ASA_input_ids.npy');xx=sparse.load_npz(V92/'ASA_R0.npz')
    assert xx.shape==(len(ids),66287)
    cc=raw[ids];used=np.flatnonzero(cc.sum(1));train_ids=ids[used];truth_np=cc[used]
    assert truth_np.sum(0).tolist()==[0,25940,5349]
    scores=np.load(DEST/'teacher_scores.npy',mmap_mode='r');base=np.load(DEST/'teacher_prediction.npy')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    if device=='cuda':torch.cuda.manual_seed_all(SEED)
    model=Branch(xx.shape[1]).to(device);initial=state_hash(model)
    xt=csr(xx[used],device)
    teacher=torch.as_tensor(np.asarray(scores[train_ids]),device=device,dtype=torch.float64)
    truth=torch.as_tensor(truth_np,device=device,dtype=torch.float64)
    mass=truth.sum(1)
    teacher_label=torch.as_tensor(base[train_ids].astype(np.int64),device=device)
    protected_mass=truth.gather(1,teacher_label[:,None]).ravel()
    eps=torch.minimum(torch.full_like(protected_mass,.001),margin(teacher,teacher_label).clamp_min(0.)/2)
    assert bool((eps[protected_mass>0]>0).all().item())
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    started=time.monotonic();history=[]
    save(DEST/f'{arm}_started.json',{'stage':'started','arm':arm,'seed':SEED,'initial_state_sha256':initial,
        'source_sha256':sha(__file__),'ASA_A_B_rows':int(truth.sum().item()),'V_labels_loaded':False,'locked_labels_loaded':False})
    for step in range(1,201):
        model.train();opt.zero_grad(set_to_none=True)
        residual=model(xt).double();z=teacher+residual
        main=(-torch.log_softmax(z,1)*truth).sum()/int(raw.sum())
        v=(eps-margin(z,teacher_label)).clamp_min(0.)
        compatibility=z.new_zeros(())
        for cl in range(3):
            w=protected_mass*(teacher_label==cl)
            if float(w.sum().item())>0:compatibility=compatibility+(w*v.square()).sum()/w.sum()
        compatibility=compatibility+v[protected_mass>0].max().square()
        magnitude=(mass*residual.square().sum(1)).sum()/int(raw.sum())
        objective=main+.01*compatibility+(.01*magnitude if arm=='MAG' else 0.)
        objective.backward()
        grad_inf=max(float(p.grad.detach().abs().max().item()) for p in model.parameters())
        assert np.isfinite(grad_inf)
        opt.step()
        if step==1 or step%25==0:
            model.eval()
            with torch.no_grad():
                end_logits=teacher+model(xt).double();pred=end_logits.argmax(1)
                err=int((mass-truth.gather(1,pred[:,None]).ravel()).sum().item())
                neg=int(protected_mass[(pred!=teacher_label)&(protected_mass>0)].sum().item())
                metrics={'step':step,'main_loss_before_update':float(main.item()),
                    'compatibility_loss_before_update':float(compatibility.item()),
                    'magnitude_loss_before_update':float(magnitude.item()),
                    'objective_before_update':float(objective.item()),'gradient_inf_before_update':grad_inf,
                    'ASA_A_B_errors_after_update':err,'teacher_correct_training_negative_flips':neg,
                    'seconds':time.monotonic()-started}
            history.append(metrics);save(DEST/f'{arm}_progress.json',history)
            print(json.dumps({'stage':'branch_progress','arm':arm,**metrics},ensure_ascii=False),flush=True)
    model.eval();all_scores=np.empty((len(ids),3),np.float64)
    with torch.no_grad():
        for start in range(0,len(ids),4096):
            end=min(start+4096,len(ids))
            all_scores[start:end]=np.asarray(scores[ids[start:end]])+model(csr(xx[start:end],device)).double().cpu().numpy()
    result_pred=base.copy();result_pred[ids]=all_scores.argmax(1).astype(np.int8)
    np.save(DEST/f'{arm}_prediction.npy',result_pred)
    torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'width':xx.shape[1],'seed':SEED,'step':200,'arm':arm},DEST/f'{arm}_model.pt')
    final={'status':'fit_executed','arm':arm,'actual_classifier_fits':1,'actual_calibration_fits':0,
        'source_sha256':sha(__file__),'teacher_model_sha256':teacher_fit['model_sha256'],
        'initial_state_sha256':initial,'steps_executed':200,'A_B_original_rows_supervised':int(raw.sum()),
        'A_B_ASA_original_rows_supervised':int(truth.sum().item()),
        'A_B_ASA_M_S_rows_supervised':truth.sum(0).cpu().numpy().astype(int).tolist(),
        'V_labels_loaded':False,'inner_C_H_labels_loaded':False,'intermediate_V_metrics_computed':False,
        'fixed_final_selection_step':200,'auxiliary_behavior_groups_used':0,
        'new_ICMP_features':False,'nonASA_branch_updates':0,
        'output_magnitude_coefficient':.01 if arm=='MAG' else 0.,
        'shared_soft_protection_coefficient':.01,
        'final_train_metrics':history[-1],'seconds':time.monotonic()-started,
        'model_sha256':sha(DEST/f'{arm}_model.pt'),'prediction_sha256':sha(DEST/f'{arm}_prediction.npy')}
    save(DEST/f'{arm}_fit.json',final)
    print(json.dumps({'stage':'branch_complete','arm':arm,'ASA_train_errors':history[-1]['ASA_A_B_errors_after_update'],
        'ASA_train_negative_flips':history[-1]['teacher_correct_training_negative_flips'],
        'seconds':final['seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['ERM','MAG']);arg=p.parse_args()
    with threadpool_limits(limits=4):fit(arg.arm)
