"""Semantic and leakage counterexamples, not SOC model quality tests."""
import unittest
import numpy as np
from scipy import sparse
import v40_core as core


class ContractTests(unittest.TestCase):
    def test_all_ports_lossless_and_distinct_from_missing(self):
        v=list(range(65536))+[None]
        bits=core.observed_bits(v,'dst_port_fixed')
        reconstructed=bits.dot(2.**np.arange(17))
        np.testing.assert_array_equal(reconstructed[:-1],np.arange(65536)+1)
        self.assertEqual(reconstructed[-1],0)
        self.assertEqual(len(np.unique(reconstructed)),65537)

    def test_other_finite_boundaries(self):
        for key,(width,sentinel) in core.FINITE.items():
            x=core.observed_bits([0,sentinel-1,None,sentinel],key)
            np.testing.assert_array_equal(x.dot(2.**np.arange(width)),[1,sentinel,0,0])

    def test_invalid_codes_fail(self):
        for n in [-1,65537,.5,True,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):core.canonicalize({'dst_port_fixed':n})

    def test_empty_has_no_schema_votes(self):
        e=core.NeutralFacts().fit([{}, {'dst_port_fixed':0}])
        self.assertEqual(e.transform([{}]).nnz,0)
        self.assertGreater(e.transform([{'dst_port_fixed':0}]).nnz,0)

    def test_observed_numeric_zero_is_not_absence(self):
        e=core.NeutralFacts().fit([{}, {'bytes':0}])
        self.assertGreater(e.transform([{'bytes':0}]).nnz,0)
        self.assertEqual(e.transform([{}]).nnz,0)

    def test_known_alias_is_deduplicated_not_unknown_failure(self):
        f=core.canonicalize({'auth_result':'failure','outcome':'failure'})[0]
        self.assertEqual(f,{'auth_result':'failure'})
        self.assertEqual(core.canonicalize({'outcome':'failure'})[0],{'outcome':'failure'})
        self.assertEqual(core.canonicalize({'auth_result':'success','outcome':'failure'})[0],{'auth_result':'success','outcome':'failure'})

    def test_no_response_observation_survives(self):
        f=core.canonicalize({'authentication_interaction':'no_response','response':'missing'})[0]
        self.assertEqual(f,{'authentication_interaction':'no_response'})
        e=core.NeutralFacts().fit([f,{}]);self.assertGreater(e.transform([f]).nnz,0)

    def test_repeated_body_does_not_meet_support(self):
        f=[{'transport_protocol':'tcp','dst_port_fixed':443}]
        e=core.ParameterEffects().fit(f,[0]*50,[7]*50)
        self.assertEqual(e.pair_count,0)
        e=core.ParameterEffects().fit(f,[0,0,0],[7,8,9])
        self.assertEqual(e.pair_count,1)

    def test_holdout_cannot_change_vocabulary(self):
        f=[{'transport_protocol':'tcp','dst_port_fixed':443},{'transport_protocol':'udp','dst_port_fixed':53}]
        e=core.ParameterEffects().fit(f,[0,0,0],[1,2,3])
        names=list(e.term_names);self.assertEqual(e.transform([f[1]]).nnz,0)
        self.assertEqual(names,e.term_names)

    def test_unseen_exact_value_preserves_fixed_fallback(self):
        facts=[{'transport_protocol':'tcp','dst_port_fixed':443}]
        e=core.ParameterEffects().fit(facts,[0,0,0],[1,2,3])
        n=core.NeutralFacts().fit(facts)
        unseen=[{'transport_protocol':'tcp','dst_port_fixed':8443},{'transport_protocol':'tcp','dst_port_fixed':8444}]
        self.assertEqual(e.transform(unseen).nnz,0)
        x=n.transform(unseen);self.assertGreater((x[0]-x[1]).nnz,0)

    def test_no_missing_or_cross_port_interactions(self):
        m,p=core.terms({'src_port_fixed':65536,'dst_port_fixed':65536})
        self.assertEqual((m,p),([],[]))
        m,p=core.terms({'src_port_fixed':1,'dst_port_fixed':2,'transport_protocol':'tcp'})
        self.assertFalse(any(t[1:3]==('src_port_fixed','dst_port_fixed') for t in p))

    def test_protocol_separates_icmp_joint_values(self):
        a=core.terms({'transport_protocol':'icmp','icmp_type':3,'icmp_code':1})[1]
        b=core.terms({'transport_protocol':'icmp6','icmp_type':3,'icmp_code':1})[1]
        self.assertNotEqual(a,b)

    def test_outer_metadata_cannot_form_terms(self):
        self.assertEqual(core.terms({'product_name':'X','timestamp':'2099','src_ip':'1.2.3.4'}),([],[]))

    def test_duplicate_multiplicity_preserves_learning_objective(self):
        # Actual optimizer on small synthetic data: representation grouping
        # retains row multiplicity rather than downweighting duplicates.
        x=sparse.csr_matrix([[1,0],[0,1],[1,1]],dtype=float)
        m,r=core.prior.learning.fit_aggregated(x,[0,0,1,1,2,2],[0,0,1,1,2,2],.1)
        self.assertEqual(r['sum_weights'],6)
        self.assertEqual(r['aggregated_rows'],3)


if __name__=='__main__':unittest.main()
