import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from v135_runtime import learning_rate,stats,window_pass,guard_sources,confirmation_allowed,endpoint,ReviewError
from v131_model import objective


class V135Tests(unittest.TestCase):
    def test_late_schedule_keeps_learning_until80(self):
        for arm in ['R_decay','O_decay']:
            self.assertTrue(all(learning_rate(arm,e)==.002 for e in range(1,81)))
            self.assertAlmostEqual(learning_rate(arm,100),.00002,12)
            self.assertTrue(all(learning_rate(arm,e)>learning_rate(arm,e+1) for e in range(80,100)))
        for e in range(1,101):self.assertEqual(learning_rate('R_const',e),.002)

    def test_mixed_mass_and_original_batch_denominators(self):
        z=torch.randn(4,16,3,dtype=torch.float64,requires_grad=True)
        c=torch.tensor([[0,72,6],[0,4,0],[0,0,3],[0,2,0]],dtype=torch.float64)
        pure=torch.tensor([0,1,1,1],dtype=torch.float64);totals=(c*pure[:,None]).sum(0);n=float(c.sum())
        full=objective(z,c,pure,n,totals,1,True)[0]
        split=sum(objective(z[ids],c[ids],pure[ids],n,totals,2,True)[0] for ids in [[0],[1,2,3]])/2
        self.assertAlmostEqual(float(full.detach()),float(split.detach()),12)
        mixed=objective(z[:1],c[:1],pure[:1],n,totals,1,True)[0]
        original=objective(z[:1],c[:1],pure[:1],n,totals,1,False)[0]
        self.assertAlmostEqual(float(mixed.detach()),.5*float(original.detach()),12)

    def test_stability_cannot_be_endpoint_only(self):
        h=[{'epoch':e,'stats':{'mastered':True}} for e in range(1,101)]
        self.assertTrue(window_pass(h));bad=deepcopy(h);bad[96]['stats']['mastered']=False
        self.assertFalse(window_pass(bad))
        with self.assertRaises(ReviewError):window_pass(h[:-1])

    def test_source_overwrite_duplicate_and_missing_population_rejected(self):
        expected=pd.Series([72,6],index=pd.MultiIndex.from_tuples([(1,1),(1,2)],names=['root','truth']),name='support')
        source=pd.DataFrame({'root':[1,1],'truth':[1,2],'support':[72,6],'epoch':[97,97]})
        guard_sources(source,expected,97)
        for bad in [source.assign(epoch=98),source.iloc[:1],pd.concat([source,source.iloc[:1]])]:
            with self.assertRaises(ReviewError):guard_sources(bad,expected,97)

    def test_all_benign_and_majority_reversal_do_not_pass(self):
        frame=pd.DataFrame({'local':[0]*78,'canonical_key':['same']*78,'truth':[1]*72+[2]*6})
        old=np.array([1]*78);pure=np.array([0.])
        self.assertTrue(stats(frame,np.array([[0.,1.,0.]]),pure,old)['mastered'])
        self.assertFalse(stats(frame,np.array([[1.,0.,0.]]),pure,old)['mastered'])
        self.assertFalse(stats(frame,np.array([[0.,0.,1.]]),pure,old)['mastered'])

    def test_quality_fail_blocks_seed_confirmation(self):
        self.assertTrue(confirmation_allowed({'R_decay_all_roles_mastered':True},{'candidate_quality_passed':True}))
        self.assertFalse(confirmation_allowed({'R_decay_all_roles_mastered':True},{'candidate_quality_passed':False}))
        good={'status':'fit_executed','completed_epochs':100,'prediction_epoch':100,'optimizer_steps':3200,'expected_steps':3200}
        endpoint(good)
        with self.assertRaises(ReviewError):endpoint({**good,'prediction_epoch':80})


if __name__=='__main__':unittest.main()
