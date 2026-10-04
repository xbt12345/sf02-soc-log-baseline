import unittest
import numpy as np
import pandas as pd
import torch
from v148_observed_relation import make_graph, observed, slots
from v148_relation_loss import prepare, relation


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor([[1.1, .2],[.1,.8]],dtype=torch.float64))
    def features(self,x):
        return (x@self.weight.T)[:,None,:]


class ObservedRelations(unittest.TestCase):
    def fixture(self):
        frame=pd.DataFrame(dict(local=range(5),root=range(10,15),canonical_key=list('abcde'),truth=[1,2,1,2,2]))
        pairs=[(1111,53),(2222,53),(1111,54),(2222,54),(65536,65536)]
        facts=pd.Series([dict(action='deny',outcome='blocked',transport_protocol='udp',src_role='inside',dst_role='outside',
            src_port_fixed=a,dst_port_fixed=b) for a,b in pairs])
        return frame,facts
    def test_unknown_is_not_a_concrete_positive_and_label_is_not_target(self):
        frame,facts=self.fixture(); nodes,edges,info=make_graph(frame,facts)
        self.assertEqual(nodes.mass.sum(),5)
        self.assertNotIn(4,set(edges.left_local))
        self.assertNotIn(4,set(edges.right_local))
        self.assertEqual(slots(facts.iloc[4]),[])
        frame.truth=2-frame.truth
        n,e,i=make_graph(frame,facts)
        self.assertTrue(n.equals(nodes) and e.equals(edges) and i==info)
        edge=edges[edges.left_local.eq(0)&edges.right_local.eq(1)].iloc[0]
        self.assertEqual(edge.target_distance,.5) # same destination; different observed source
        self.assertEqual(edge.observed_fields,2)
    def test_protocol_applicability_and_port_value_difference_are_retained(self):
        frame,facts=self.fixture(); _,edges,_=make_graph(frame,facts)
        edge=edges[edges.left_local.eq(0)&edges.right_local.eq(3)].iloc[0]
        self.assertEqual(edge.target_distance,1.) # both known exact port values differ
        icmp=dict(transport_protocol='icmp',src_port_fixed=53,dst_port_fixed=53,icmp_type=3,icmp_code=13)
        self.assertFalse(observed(icmp,'src_port_fixed'))
        self.assertEqual(slots(icmp),[('icmp_type_code','3/13')])
    def test_chunked_gradient_matches_finite_difference_and_adds_to_CE(self):
        model=Toy(); x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float64)
        edges=pd.DataFrame(dict(left_local=[0,1,2],right_local=[1,2,0],target_distance=[.25,.5,0.],weight=[3.,2.,1.]))
        graph=prepare(edges); model.zero_grad(); value=relation(model,x,graph,backward_scale=1.,chunk=1); g=model.weight.grad.clone()
        with torch.no_grad():
            h=1e-6; plus=dict(weight=model.weight.detach().clone()); minus=dict(weight=model.weight.detach().clone())
            plus['weight'][0,0]+=h;minus['weight'][0,0]-=h
            fd=(relation(model,x,graph,parameters=plus,chunk=2)-relation(model,x,graph,parameters=minus,chunk=2))/(2*h)
        self.assertAlmostEqual(float(g[0,0]),fd,places=8)
        original=model.weight.detach().clone(); model.zero_grad(); (model.weight.square().sum()).backward(); ce=model.weight.grad.clone()
        repeated=relation(model,x,graph,backward_scale=.3,chunk=2)
        self.assertAlmostEqual(value,repeated,places=14)
        self.assertTrue(torch.allclose(model.weight.grad,ce+.3*g,atol=1e-14,rtol=1e-14))
        self.assertTrue(torch.equal(original,model.weight))


if __name__=='__main__':unittest.main()
