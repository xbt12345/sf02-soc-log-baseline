"""Independent contrast formula, derivatives and real full-frequency CE identities."""
import unittest
import numpy as np
import torch
from v144_pair_objective import contrast_from_features,TEMPERATURE
from v142_runtime import ROOT,read,sha


class Tests(unittest.TestCase):
    def test_root_prototype_loss_against_explicit_original_weighted_formula(self):
        x=torch.tensor([[[1.,.2,.3]],[[.7,.1,.8]],[[.8,.6,.2]],[[.1,.8,.4]],[[.3,.7,.1]]],dtype=torch.float64,requires_grad=True)
        p=dict(source_index=np.array([0,1,2,3,4]),cell_index=np.array([0,0,1,2,3]),edge_mass=np.array([2.,1.,1.,1.,1.]),
               cell_mass=np.array([3.,1.,1.,1.]),labels=np.array([1,1,2,2]),groups=[np.arange(4)])
        actual=contrast_from_features(x,p)
        u=x/torch.linalg.vector_norm(x,dim=-1,keepdim=True)
        centers=torch.stack([(2*u[0]+u[1])/3,u[2],u[3],u[4]])[:,0]
        centers=centers/torch.linalg.vector_norm(centers,dim=-1,keepdim=True)
        labels=[1,1,2,2];values=[]
        for i in range(4):
            logits=torch.stack([(centers[i]*centers[j]).sum()/TEMPERATURE for j in range(4) if j!=i])
            positive=next(j for j in range(4) if i!=j and labels[i]==labels[j])
            values.append(torch.logsumexp(logits,0)-(centers[i]*centers[positive]).sum()/TEMPERATURE)
        reference=sum(values)/4
        self.assertLess(abs(float(actual-reference)),1e-12)
        a=torch.autograd.grad(actual,x,retain_graph=True)[0];b=torch.autograd.grad(reference,x)[0]
        self.assertLess(float((a-b).abs().max()),1e-12)

    def test_auxiliary_feature_gradient_matches_finite_difference(self):
        gen=torch.Generator().manual_seed(17)
        x=torch.randn(4,2,5,generator=gen,dtype=torch.float64,requires_grad=True)
        p=dict(source_index=np.arange(4),cell_index=np.arange(4),edge_mass=np.ones(4),cell_mass=np.ones(4),labels=np.array([1,1,2,2]),groups=[np.arange(4)])
        direction=torch.randn(4,2,5,generator=gen,dtype=torch.float64);direction/=direction.norm()
        loss=contrast_from_features(x,p);g=torch.autograd.grad(loss,x)[0]
        step=1e-6;fd=(contrast_from_features(x.detach()+step*direction,p)-contrast_from_features(x.detach()-step*direction,p))/(2*step)
        self.assertLess(abs(float(fd-(g*direction).sum())),1e-7)

    def test_real_CE_derivatives_and_fixed_ratio_are_consistent(self):
        a=read(ROOT/'artifacts/v144_aux_gradient_evidence_20261001/probe.json')
        self.assertEqual((a['new_fits'],a['new_updates'],a['full_CE_gradient_evaluations'],a['full_auxiliary_gradient_evaluations']),(0,0,3,6))
        self.assertTrue(a['baseline_mastery_guard']['passed']);self.assertFalse(a['second_issue_solved'])
        for p,h in a['evidence_bindings'].items():self.assertEqual(sha(ROOT/p),h)
        for r in a['folds']:
            self.assertEqual(r['capacity']['minimum_sources_per_class'],2)
            cn=r['full_CE_gradient_norm'];an=r['auxiliary_gradient_norm'];lam=r['fixed_initial_auxiliary_coefficient']
            self.assertAlmostEqual(lam*an/cn,0.1,places=12)
            mass=np.array(r['original_class_mass_seen'])
            s={v['direction']:v['classes'] for v in r['margin_differentials']}
            weighted=lambda key:sum(v['full_original_member_CE_derivative']*mass[v['truth']]/mass.sum() for v in s[key])
            self.assertAlmostEqual(weighted('aux'),-cn*r['CE_auxiliary_gradient_cosine'],places=11)
            self.assertAlmostEqual(weighted('combined'),r['combined_full_CE_descent_derivative'],places=11)
            self.assertLess(weighted('combined'),0)
            # Actual risk remains visible: standalone auxiliary raises M CE in every role.
            self.assertGreater(s['aux'][0]['full_original_member_CE_derivative'],0)


if __name__=='__main__':unittest.main()
