"""Counterexamples for support leakage, exact fallback and bounded subtype fit."""
import copy
import unittest
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
import v42_core as c

F={'action':'deny','outcome':'blocked','transport_protocol':'udp','src_role':'outside','dst_role':'dmz','src_port_fixed':1234,'dst_port_fixed':53}


class ContextTests(unittest.TestCase):
    def test_missing_values_cannot_be_context_support(self):
        for key in ('src_port_fixed','dst_port_fixed','src_role','dst_role'):
            f=F.copy();f.pop(key);self.assertIsNone(c.complete_context('',f)[0])
        for value in (True,65536,-1,2.5,None,'53'):
            f=dict(F,dst_port_fixed=value);self.assertIsNone(c.complete_context('',f)[0])

    def test_body_multiplicity_does_not_fake_independent_support(self):
        support=c.ContextSupport().fit([''],[F],[0]*100,[7]*100)
        self.assertEqual(len(support.context_keys),0)
        support=c.ContextSupport().fit([''],[F],[0]*3,[7,8,9]);self.assertEqual(len(support.context_keys),1)

    def test_components_seen_but_combination_unknown(self):
        f2=dict(F,src_port_fixed=2345,dst_port_fixed=443)
        support=c.ContextSupport().fit(['',''],[F,f2],[0,0,0,1,1,1],[1,2,3,4,5,6])
        mixed=dict(F,dst_port_fixed=443);ids,why=support.encode([''],[mixed])
        self.assertEqual(ids[0],-1);self.assertEqual(why[0],'unseen_full_context')

    def test_full_text_and_additional_facts_not_ignored(self):
        a=c.complete_context('',F)[0]
        self.assertNotEqual(a,c.complete_context('different observed body',F)[0])
        self.assertNotEqual(a,c.complete_context('',dict(F,extra_observation='x'))[0])

    def test_icmp_does_not_require_inapplicable_ports(self):
        f=dict(F,transport_protocol='icmp',src_port_fixed=65536,dst_port_fixed=65536,icmp_type=3,icmp_code=13)
        self.assertIsNotNone(c.complete_context('',f)[0]);f.pop('icmp_code');self.assertIsNone(c.complete_context('',f)[0])

    def test_no_evaluation_support_added_at_prediction(self):
        s=c.ContextSupport().fit([''],[F],[0,0],[1,2]);before=copy.deepcopy(s.__dict__)
        for _ in range(10):s.encode(['']*10,[F]*10)
        self.assertEqual(s.__dict__,before);self.assertEqual(s.encode([''],[F])[0][0],-1)


class OffsetTests(unittest.TestCase):
    def test_inactive_and_zero_offsets_are_exact_identity(self):
        p=np.array([[.2,.5,.3],[.8,.1,.1],[.01,.5,.49]])
        np.testing.assert_array_equal(c.apply_offsets(p,np.zeros(3),np.ones(3,bool)),p)
        np.testing.assert_array_equal(c.apply_offsets(p,np.full(3,c.MAX_OFFSET),np.zeros(3,bool)),p)

    def test_never_changes_benign_probability_or_binary_decision(self):
        p=np.array([[.40,.59,.01],[.4,.15,.45],[.45,.3,.25],[.01,.1,.89]])
        for value in (-c.MAX_OFFSET,c.MAX_OFFSET):
            q=c.apply_offsets(p,np.full(4,value),np.ones(4,bool))
            np.testing.assert_array_equal(q[:,0],p[:,0]);np.testing.assert_array_equal(q.argmax(1)==0,p.argmax(1)==0)

    def test_optimizer_matches_independent_scalar_solver(self):
        s=c.ContextSupport().fit([''],[F],[0]*5,[1,2,3,4,5]);p=np.array([[.02,.6,.38]])
        model=c.FrozenSubtypeOffset().fit(s,[''],[F],p,[0]*5,[1,2,2,2,1]);m=c.conditional_margin(p)[0]
        loss=lambda d:5*np.logaddexp(0,m+d)-2*(m+d)+5*d*d
        ref=minimize_scalar(loss,bounds=(-c.MAX_OFFSET,c.MAX_OFFSET),method='bounded',options={'xatol':1e-12})
        self.assertAlmostEqual(model.delta[0],ref.x,places=7)
        self.assertEqual(model.optimization['exact_aggregated_row_weight'],5)

    def test_empty_support_and_normal_only_targets_do_not_learn(self):
        for bodies in ([1,1,1],[1,2,3]):
            s=c.ContextSupport().fit([''],[F],[0]*3,bodies);p=np.array([[.8,.1,.1]])
            model=c.FrozenSubtypeOffset().fit(s,[''],[F],p,[0]*3,[0]*3)
            np.testing.assert_array_equal(model.predict([''],[F],p)[0],p)

    def test_no_cross_context_parameter_interference(self):
        f2=dict(F,dst_port_fixed=443);s=c.ContextSupport().fit(['',''],[F,f2],[0]*3+[1]*3,[1,2,3,4,5,6]);p=np.array([[.01,.5,.49],[.01,.3,.69]])
        a=c.FrozenSubtypeOffset().fit(s,['',''],[F,f2],p,[0]*3+[1]*3,[1]*3+[1]*3)
        b=c.FrozenSubtypeOffset().fit(s,['',''],[F,f2],p,[0]*3+[1]*3,[1]*3+[2]*3)
        idx=s.encode([''],[F])[0][0];self.assertEqual(a.delta[idx],b.delta[idx])


if __name__=='__main__':unittest.main()
