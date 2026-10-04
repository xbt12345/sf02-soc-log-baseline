"""Counterexamples for allocation, final quarantine, tokens and threshold ties."""
import unittest
import numpy as np
from v32_finalize import finalize_text
from v32_features import word_tokens
from v32_split import allocate,repair,role_summary
from run_v32_train import risk_threshold,alerts,new_vectorizer


class ExecutionContracts(unittest.TestCase):
    def test_dotted_assembly_component_is_explicitly_unknown(self):
        a,spans=finalize_text("file_v4.0_100.64.85.111_hash.cdf-ms")
        b,_=finalize_text("file_v4.0_100.64.85.112_hash.cdf-ms")
        self.assertEqual(a,b)
        self.assertIn("file_v4.0_",a)
        self.assertIn(".cdf-ms",a)
        self.assertEqual(spans[0]["old_value"],"100.64.85.111")

    def test_compound_address_tail_does_not_survive_as_identity(self):
        text,spans=finalize_text("address 100.64.137.117f9b6:4a31:6b16:f5fa port 443")
        self.assertEqual(text,"address unknown_dotted_value port 443")
        self.assertTrue(spans)

    def test_explicit_versions_and_non_ipv4_numeric_values_retained(self):
        for text in ["version 1.2.3.4","C:\\app\\23.3.3.264\\runner.exe"]:
            self.assertEqual(finalize_text(text),(text,()))

    def test_zero_one_negation_and_command_operators_are_distinct(self):
        for a,b in [("value 0","value 1"),("code 4624","code 4625"),("failure","no failure"),
                    ("echo x > file","echo x >> file"),("cmd -flag","cmd --flag")]:
            self.assertNotEqual(word_tokens(a),word_tokens(b))
            vectorizer=new_vectorizer()
            x=vectorizer.fit_transform([a,b])
            self.assertNotEqual((x[0]-x[1]).nnz,0)

    def test_float32_ties_do_not_break_zero_budget(self):
        risk=np.array([0.5]*100,dtype=np.float32)
        threshold=risk_threshold(risk,0.001)
        result=alerts(np.zeros(100,dtype=np.uint8),risk,threshold)
        self.assertEqual(result["alert_rows"],0)

    def test_threshold_allows_only_complete_admissible_score_ties(self):
        risk=np.array([0.9,0.8,0.8,0.1],dtype=np.float32)
        threshold=risk_threshold(risk,0.5)
        self.assertEqual(alerts(np.zeros(4,dtype=np.uint8),risk,threshold)["alert_rows"],1)

    def test_large_group_can_be_allocated_without_splitting_or_singleton_padding(self):
        # A group-size/support counterexample, not synthetic SOC training.
        counts=np.zeros((601,3),dtype=np.int64)
        for c in range(3):counts[c*200:(c+1)*200,c]=50
        counts[-1,2]=4000
        roles=allocate(counts,counts,np.arange(len(counts)),[0.6,0.2,0.2])
        roles,moves,checked,ok=repair(roles,counts,counts)
        self.assertTrue(ok)
        self.assertTrue(np.isin(roles,[0,1,2]).all())
        self.assertEqual(sum(counts[roles==r].sum() for r in range(3)),counts.sum())
        for r in (1,2):
            groups,total,largest=role_summary(roles,counts,r)
            self.assertTrue((groups>=30).all())
            self.assertTrue((largest<=total*0.5).all())


if __name__=="__main__":unittest.main(verbosity=2)
