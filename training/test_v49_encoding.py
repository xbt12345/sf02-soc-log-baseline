import unittest
import numpy as np
import v49_encoding as v

class ObservedEncoding(unittest.TestCase):
    def encoder(self, facts): return v.EvidenceFacts().fit(facts)

    def test_absent_fields_have_no_value_contribution(self):
        f = {'outcome':'failure','auth_result':'failure','src_port_fixed':65536,'dst_port_fixed':65536}
        e=self.encoder([f]);x=e.transform([f]).toarray()[0]
        for k in v.FINITE:
            self.assertTrue(np.all(x[e.blocks[k]]==0))
        self.assertEqual(x[-len(v.FINITE):].sum(),0)
        self.assertNotEqual(x.sum(),0)

    def test_every_finite_value_bit_is_preserved_and_zero_is_observed(self):
        for k,(width,missing) in v.FINITE.items():
            values=sorted(set([0,1,missing-1]+[1<<j for j in range(width) if 1<<j < missing]))
            records=[{k:n} for n in values]+[{}, {k:missing}]
            e=self.encoder(records);x=e.transform(records).toarray();j=list(v.FINITE).index(k)
            for i,n in enumerate(values):
                self.assertEqual(int(sum(x[i,c]*(1<<bit) for bit,c in enumerate(e.blocks[k]))),n)
                self.assertEqual(x[i,len(e.original_names)+j],1)
            np.testing.assert_array_equal(x[-1],x[-2])
            self.assertFalse(np.array_equal(x[0],x[-1]))

    def test_observed_rows_and_other_facts_preserved(self):
        f={'src_port_fixed':443,'dst_port_fixed':0,'icmp_type':3,'icmp_code':13,'transport_protocol':'tcp',
           'outcome':'failure','attempt_count':0,'auth_result':'failure','status':'c000006d'}
        e=self.encoder([f]);before=e.base.transform([f]).toarray()[0];after=e.transform([f]).toarray()[0]
        observed=set()
        for k in v.FINITE:
            if v.finite_observation(f,k) is not None:observed.update(e.blocks[k])
        allbits=set(np.concatenate(list(e.blocks.values())))
        keep=[i for i in range(len(before)) if i not in allbits or i in observed]
        np.testing.assert_array_equal(before[keep],after[keep])

    def test_invalid_and_unknown_fields_fail_visibly(self):
        e=self.encoder([{}])
        for f in [{'src_port_fixed':-1},{'icmp_type':1.5},{'status':True},{'mystery':'foo'},{'action':'invented'}]:
            with self.assertRaises((ValueError,TypeError)):e.transform([f])

    def test_no_label_or_source_used(self):
        from test_v48_input import msg
        raw=msg();a=v.input_adapter.prepare_record({'message_sanitized':raw})
        b=v.input_adapter.prepare_record({'message_sanitized':raw,'label_binary':'benign','product_name':'a','timestamp':'2099'})
        e=self.encoder([a['facts']]);np.testing.assert_array_equal(e.transform([a['facts']]).toarray(),e.transform([b['facts']]).toarray())

if __name__=='__main__':unittest.main()
