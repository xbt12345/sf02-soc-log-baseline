import unittest
from collections import Counter
import numpy as np
import pandas as pd
import torch
from train_v65_rank_heads import Head,rank_loss,pair_schedule
from v61_common import FIELDS


def pool():
    return pd.DataFrame([{'role':'fit','behavior':'b','group':g,'label':c,
           'index':100*g+v,'view':str((g,v))}
           for c,gs in [(1,range(20)),(2,range(30,35))] for g in gs for v in range(2)])


class RankTests(unittest.TestCase):
    def test_rank_loss_improves_correct_order_and_has_expected_gradients(self):
        m=torch.zeros(3,3,requires_grad=True);s=torch.zeros(3,3,requires_grad=True)
        loss=rank_loss(m,s);loss.backward()
        self.assertTrue((m.grad[:,1]<0).all());self.assertTrue((s.grad[:,2]<0).all())
        better_m=torch.tensor([[0.,2.,0.]]);better_s=torch.tensor([[0.,0.,2.]])
        self.assertLess(float(rank_loss(better_m,better_s)),float(loss))
        torch.testing.assert_close(rank_loss(better_m+10,better_s-5),rank_loss(better_m,better_s))

    def test_pair_caps_membership_class_and_source(self):
        p=pool();lookup=p.set_index('index');schedule,info=pair_schedule(p,1,30,0)
        counts=Counter();pairs=[]
        for b in schedule.values():
            self.assertLessEqual(len(b),16)
            for m,s in b:
                a=lookup.loc[m];z=lookup.loc[s]
                self.assertEqual((a.label,z.label),(1,2));self.assertNotEqual(a.group,z.group)
                self.assertNotEqual(a['view'],z['view']);counts[a.group]+=1;counts[z.group]+=1;pairs.append((m,s))
        self.assertEqual(len(pairs),40);self.assertLessEqual(max(counts.values()),8)
        self.assertEqual(schedule,pair_schedule(p,1,30,0)[0])

    def test_mixed_view_excluded_only_from_pairs(self):
        p=pool();p.loc[p.index.isin([0,40]),'view']='contradiction'
        before=p.copy(deep=True);blocked=set(p.loc[p['view'].eq('contradiction'),'index'])
        schedule,_=pair_schedule(p,1,30,0)
        self.assertFalse(blocked & {i for b in schedule.values() for pair in b for i in pair})
        pd.testing.assert_frame_equal(before,p)

    def test_total_source_cap_shared_between_class_roles(self):
        p=pool();p.loc[p.group.eq(30),'group']=0
        schedule,_=pair_schedule(p,1,30,0);lookup=p.set_index('index');counts=Counter()
        for b in schedule.values():
            for m,s in b:counts[lookup.loc[m,'group']]+=1;counts[lookup.loc[s,'group']]+=1
        self.assertLessEqual(max(counts.values()),8)

    def test_validation_pool_rejected(self):
        p=pool();p.loc[0,'role']='validation'
        with self.assertRaises(ValueError):pair_schedule(p,1,30,0)

    def test_neighbor_order_and_padding_invariance(self):
        torch.manual_seed(65);model=Head([list(range(5)) for _ in FIELDS],8).eval()
        x={'text':torch.randn(2,8),'facts':torch.zeros(2,len(FIELDS),dtype=torch.long),
           'stats':torch.randn(2,16),'neighbor_facts':torch.randint(0,5,(2,4,len(FIELDS))),
           'mask':torch.tensor([[True,True,False,False],[False,False,False,False]]),
           'relations':torch.randn(2,4,2)}
        y=model(**x);permutation=torch.tensor([3,1,0,2]);other={k:v.clone() for k,v in x.items()}
        for k in ['neighbor_facts','mask','relations']:other[k]=other[k][:,permutation]
        other['neighbor_facts'][~other['mask']]=4;other['relations'][~other['mask']]=999
        torch.testing.assert_close(model(**other),y,rtol=1e-5,atol=1e-6)
        self.assertTrue(torch.isfinite(y).all())


if __name__=='__main__':unittest.main()
