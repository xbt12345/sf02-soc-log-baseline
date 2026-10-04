import unittest
import numpy as np
import pandas as pd
import torch
from train_v63_factorial import port_codes, nibbles, PortEmbedding, auxiliary_batches, select_checkpoint, budget


class FactorialTests(unittest.TestCase):
    def test_all_ports_roundtrip_without_collision(self):
        codes = torch.from_numpy(port_codes([str(x) for x in range(65536)]))
        digits = nibbles(codes)
        restored = sum(digits[:, j] << shift for j, shift in enumerate([12, 8, 4, 0]))
        self.assertTrue(torch.equal(restored, torch.arange(65536)))
        self.assertEqual(int(torch.unique(codes).numel()), 65536)
        self.assertEqual(port_codes(['__MISSING__','0','65535']).tolist(), [0, 1, 65536])

    def test_bad_ports_fail(self):
        for value in ['-1','65536','NaN','','12.0',None,12]:
            with self.assertRaises(ValueError): port_codes([value])
        for value in [-1, 65537]:
            with self.assertRaises(ValueError): nibbles(torch.tensor([value]))

    def test_compositional_and_missing_weights_get_gradient(self):
        torch.manual_seed(63); module = PortEmbedding()
        value = module(torch.tensor([0,1,65536,437]))
        self.assertEqual(tuple(value.shape), (4,16))
        self.assertFalse(torch.equal(value[0], value[1]))
        value.square().sum().backward()
        self.assertGreater(float(module.missing.grad.norm()),0)
        self.assertTrue(all(float(e.weight.grad.norm()) > 0 for e in module.digits))

    def test_auxiliary_schedule_caps_and_balance(self):
        pool = pd.DataFrame([{'behavior': 'b', 'group': g, 'label': c, 'index': g*10+v}
                             for c, gs in [(1,range(20)),(2,range(30,41))] for g in gs for v in [0,1]])
        schedule = auxiliary_batches(pool,1,100)
        self.assertEqual(schedule, auxiliary_batches(pool.sample(frac=1,random_state=3).sort_index(),1,100))
        table = pool.set_index('index'); seen = []
        for batch in schedule.values():
            x = table.loc[batch['indices']]
            self.assertTrue(x.groupby('label').size().eq(len(x)//2).all())
            self.assertTrue(x.group.is_unique)
            seen.extend(batch['indices'])
        counts = table.loc[seen].groupby(['label','group']).size()
        self.assertLessEqual(int(counts.max()),8)
        self.assertTrue(counts.loc[2].eq(8).all())
        self.assertEqual(min(schedule),0); self.assertEqual(max(schedule),99)

    def test_auxiliary_random_stream_restores_main_rng(self):
        torch.manual_seed(123); expected=torch.rand(10)
        torch.manual_seed(123)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(987); torch.rand(100)
        self.assertTrue(torch.equal(expected,torch.rand(10)))

    def test_improved_S_cannot_bypass_M_budget(self):
        base={'M_source_recall':.99,'S_source_recall':.3,'source_balanced':.645,
              'ASA':{'recall_B_M_S':[None,.99,.3],'macro_f1_M_S':.9},
              'hard':{p:{'1':.99,'2':.1} for p in ['tcp','udp']}}
        import copy
        candidate=copy.deepcopy(base);candidate.update(M_source_recall=.8,S_source_recall=.9,source_balanced=.85)
        result=select_checkpoint([{'step':1,'metrics':candidate}],base,base)
        self.assertFalse(result['budget_eligible']);self.assertFalse(result['screen_passed'])

    def test_stronger_reference_detects_hidden_row_regression(self):
        reference={'M_source_recall':.999,'S_source_recall':.28,
                   'ASA':{'recall_B_M_S':[None,.917,.96],'macro_f1_M_S':.916},
                   'hard':{p:{'1':.99,'2':.1} for p in ['tcp','udp']}}
        import copy
        candidate=copy.deepcopy(reference);candidate['S_source_recall']=.4
        peak=copy.deepcopy(reference)
        peak['ASA'].update(recall_B_M_S=[None,.993,.96],macro_f1_M_S=.98)
        self.assertTrue(budget(candidate,reference))
        self.assertFalse(budget(candidate,peak))

    def test_whole_source_bootstrap_preserves_joint_class_tradeoff(self):
        from analyze_v63_factorial import paired_source
        frame=pd.DataFrame({'group':np.repeat(np.arange(50),2),'label':np.tile([1,2],50)})
        base=np.zeros((100,3));base[:,1]=1
        candidate=base.copy();mask=frame.group.lt(25).to_numpy()
        candidate[mask,1]=0;candidate[mask,2]=1
        result=paired_source(frame,base,candidate,repeats=200)
        self.assertEqual(result['source_M_delta'],-.5)
        self.assertEqual(result['source_S_delta'],.5)
        self.assertEqual(result['balanced_95_interval'],[0.,0.])


if __name__=='__main__': unittest.main()
