"""Frozen endpoint gradient accounting; no optimizer or parameter update."""
import json
import numpy as np
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from v89_common import ROOT, read, save, sha
from v92_train import Branch, csr, margin
from v99_normalization_feasibility import DEST, V92


def main():
    assert not (DEST/'objective_scale_audit.json').exists()
    v97=ROOT/'artifacts/v97_matched_teacher_20260928'
    with np.load(v97/'AB_train_counts.npz') as a:counts=a['A'].astype(np.int64)+a['B'].astype(np.int64)
    ids=np.load(V92/'ASA_input_ids.npy');x=sparse.load_npz(V92/'ASA_R0.npz')
    c=counts[ids];take=np.flatnonzero(c.sum(1));n=int(c.sum());d=int(counts.sum());ratio=d/n
    target=torch.tensor(c[take],dtype=torch.float64);mass=target.sum(1)
    teacher=torch.tensor(np.load(v97/'teacher_scores.npy')[ids[take]],dtype=torch.float64)
    label=teacher.argmax(1);protected=target.gather(1,label[:,None]).ravel()
    eps=torch.minimum(torch.full_like(protected,.001),margin(teacher,label).clamp_min(0)/2)
    model=Branch(x.shape[1]);state=torch.load(v97/'MAG_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['state_dict']);model.eval();residual=model(csr(x[take],'cpu')).double();z=teacher+residual
    ce=(-torch.log_softmax(z,1)*target).sum()/d
    mag=(mass*residual.square().sum(1)).sum()/d
    violation=(eps-margin(z,label)).clamp_min(0)
    p=z.new_zeros(())
    for cl in range(3):
        weight=protected*(label==cl)
        if float(weight.sum())>0:p=p+(weight*violation.square()).sum()/weight.sum()
    p=p+violation[protected>0].max().square()
    params=tuple(model.parameters())
    def vector(loss):return torch.cat([g.ravel() for g in torch.autograd.grad(loss,params,retain_graph=True)])
    g=vector(ce);h=vector(.01*mag);j=vector(.01*p);whole=g+h+j
    result={'status':'frozen_objective_units_audited_no_fit','source_sha256':sha(__file__),
        'full_training_rows_denominator':d,'active_ASA_rows':n,'full_over_ASA':ratio,
        'full_normalized_CE':float(ce.detach()),'ASA_normalized_CE':float(ce.detach()*ratio),
        'weighted_magnitude':float((.01*mag).detach()),'weighted_protection':float((.01*p).detach()),
        'CE_gradient_l2':float(g.norm()),'weighted_magnitude_gradient_l2':float(h.norm()),
        'weighted_protection_gradient_l2':float(j.norm()),
        'CE_magnitude_gradient_cosine':float(torch.dot(g,h)/(g.norm()*h.norm())),
        'total_gradient_inf_full_normalized':float(whole.abs().max()),
        'total_gradient_inf_equivalent_ASA_units':float(whole.abs().max()*ratio),
        'equivalent_protection_coefficient_in_ASA_units':.01*ratio,
        'new_classifier_fits':0,'parameter_updates':0,
        'scope':'One frozen MAG endpoint. Loss scaling and gradient interaction only; not proof of converged optimum or cause of all training errors.'}
    assert all(torch.equal(model.state_dict()[k],v) for k,v in state['state_dict'].items())
    save(DEST/'objective_scale_audit.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):main()
