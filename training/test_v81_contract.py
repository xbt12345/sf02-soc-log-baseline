import unittest
from v81_training_contract import accept_candidate
from v81_flow_observations import derive_vpc_observations


class ContractTests(unittest.TestCase):
    def test_local_gain_with_unchanged_other_source_is_allowed(self):
        a=[[100,0,0],[0,8,2],[0,2,8]];b=[[100,0,0],[0,9,1],[0,1,9]]
        unchanged=[[50,0,0],[0,4,1],[0,1,4]]
        self.assertTrue(accept_candidate(a,b,{'other_source':(unchanged,unchanged)})['eligible'])

    def test_fewer_errors_cannot_hide_malicious_regression(self):
        a=[[100,0,0],[0,10,0],[0,8,2]];b=[[100,0,0],[0,6,4],[0,0,10]]
        self.assertFalse(accept_candidate(a,b,{'same':(a,b)})['eligible'])

    def test_overall_gain_cannot_hide_source_regression(self):
        a=[[100,0,0],[0,8,2],[0,2,8]];b=[[100,0,0],[0,9,1],[0,1,9]]
        g0=[[10,0,0],[0,2,0],[0,0,2]];g1=[[10,0,0],[0,1,1],[0,0,2]]
        self.assertFalse(accept_candidate(a,b,{'bad':(g0,g1)})['eligible'])

    def test_empty_required_slice_is_not_success(self):
        a=[[100,0,0],[0,8,2],[0,2,8]];b=[[100,0,0],[0,9,1],[0,1,9]];empty=[[0]*3]*3
        self.assertFalse(accept_candidate(a,b,{'missing':(empty,empty)})['eligible'])

    def test_different_population_is_rejected(self):
        with self.assertRaises(ValueError):accept_candidate([[1,0,0],[0,1,0],[0,0,1]],[[2,0,0],[0,1,0],[0,0,1]],{})

    def example(self,start='100',end='130',packets='10',bytecount='1000'):
        return f'2 123456789012 eni-a 10.0.0.1 10.0.0.2 1234 443 6 {packets} {bytecount} {start} {end} REJECT OK'

    def test_absolute_clock_shift_cannot_change_features(self):
        a=derive_vpc_observations(self.example(),'vpc_v2');b=derive_vpc_observations(self.example('1000000','1000030'),'vpc_v2')
        self.assertEqual(a['values'],b['values']);self.assertEqual(a['values']['observed_interval_seconds'],30)
        self.assertEqual(a['values']['observed_bytes_per_packet'],100)
        self.assertFalse({'start','end','timestamp','label'}&set(a['values']))

    def test_redacted_and_negative_time_not_guessed(self):
        for start,end in [('CRED-1','130'),('140','130')]:
            a=derive_vpc_observations(self.example(start,end),'vpc_v2')
            self.assertNotIn('observed_interval_seconds',a['values']);self.assertNotIn('observed_packets_per_second',a['values'])

    def test_observed_zero_kept_without_fabricated_denominator(self):
        a=derive_vpc_observations(self.example('100','100'),'vpc_v2')
        self.assertEqual(a['values']['observed_interval_seconds'],0);self.assertNotIn('observed_packets_per_second',a['values'])

    def test_wrong_grammar_or_zero_packets_cannot_make_rate(self):
        self.assertEqual(derive_vpc_observations(self.example(),'unsupported')['values'],{})
        self.assertEqual(derive_vpc_observations('2 missing','vpc_v2')['values'],{})
        self.assertNotIn('observed_bytes_per_packet',derive_vpc_observations(self.example(packets='0'),'vpc_v2')['values'])


if __name__=='__main__':unittest.main()
