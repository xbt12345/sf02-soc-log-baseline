import json,unittest
import numpy as np
import v38_representation as r
import v38_learning as l

ASA='Deny tcp src outside:10.1.2.3/12345 dst dmz-1:10.2.3.4/443 by access-group "acl" [0x0, 0x0]'
CEF='CEF:0|vendor|WAF|1|119|detector|1|act=DENY app=TLSv1.3 spt=54941 msg=a\\=1'
class V38(unittest.TestCase):
    def test_real_series_product_column_not_method(self):
        import pandas as pd
        from run_v38_prepare import full_source_audit
        for source in ('Barracuda WAF','Duo'):
            self.assertTrue(full_source_audit(pd.Series({'product':source})))
        self.assertFalse(full_source_audit(pd.Series({'product':'other'})))
    def test_all_ports_roundtrip(self):
        a=np.arange(65537);bits=l.port_bits(a)
        np.testing.assert_array_equal(bits.dot(2.**np.arange(17)),a)
        for bad in [np.array([-1]),np.array([1.5]),np.array([65537])]:
            with self.assertRaises(ValueError):l.port_bits(bad)
    def test_views_do_not_bypass_ports(self):
        a=r.prepare_message(ASA);b=r.prepare_message(ASA.replace('/12345','/12346'))
        for view in r.VIEWS[:2]:self.assertEqual(r.view_record(a,view),r.view_record(b,view))
        self.assertNotEqual(r.view_record(a,r.VIEWS[2]),r.view_record(b,r.VIEWS[2]))
        for view in r.VIEWS:
            v=r.view_record(a,view);self.assertEqual(v['text'],'');self.assertNotIn('category',v['facts'])
    def test_zero_missing_unseen_are_distinct(self):
        e=l.FixedFacts('C_BOTH').fit([{'dst_port_fixed':80}])
        x=e.transform([{'dst_port_fixed':0},{'dst_port_fixed':65536},{'dst_port_fixed':65000},{'dst_port_fixed':65001}]).toarray()
        self.assertEqual(len(np.unique(x,axis=0)),4)
        self.assertEqual(e.transform([{'credential_check':'valid'}]).shape[1],len(e.names()))
        self.assertIn('credential_check=valid',e.names())
    def test_credential_is_not_whole_event_success(self):
        p=r.project('reason valid_passcode',{'category':'authentication','credential_check':'valid','outcome':'failure'},'authentication')
        self.assertEqual(p['facts']['outcome'],'failure');self.assertNotIn('category',p['facts'])
    def test_protocol_layers_and_outer_invariance(self):
        p=r.prepare_message(CEF);self.assertEqual(p['facts']['application_protocol'],'tlsv1.3')
        self.assertNotIn('transport_protocol',p['facts'])
        for pre in ['','<190>Jul 26 00:00:00 host ','<190>Jul 26 - other ']:self.assertEqual(p,r.prepare_message(pre+CEF))
        class Row(dict):
            def get(self,key,*args):
                if key!='message_sanitized':raise AssertionError(key)
                return CEF
        self.assertEqual(p,r.prepare_record(Row()))
    def test_real_action_change_retained(self):
        self.assertNotEqual(r.prepare_message(CEF),r.prepare_message(CEF.replace('DENY','ALLOW')))
    def test_port_audit_redacted_invalid_zero(self):
        self.assertEqual(r.token_state('USER-123')['state'],'redacted')
        self.assertEqual(r.token_state('1.5')['state'],'invalid')
        self.assertEqual(r.token_state('0'),{'state':'observed','value':0})
    def test_forbidden_fields_fail(self):
        for f in [{'category':'authentication'},{'src_port_category':'tcp|1'}]:
            with self.assertRaises(ValueError):l.FixedFacts('A_SEMANTIC').fit([f])
    def test_binary_values_are_not_learned_vocabulary(self):
        e=l.FixedFacts('C_BOTH').fit([{'dst_port_fixed':1}]);before=e.names().tolist()
        e.transform([{'dst_port_fixed':65535}]);self.assertEqual(before,e.names().tolist())
    def test_raw_inference_and_joint_decision_roundtrip(self):
        import tempfile,joblib
        from pathlib import Path
        from scipy import sparse
        records=[r.prepare_message(ASA),r.prepare_message(CEF),r.prepare_message('')]
        rows=[r.view_record(p,'C_BOTH') for p in records]
        texts=[p['text'] for p in rows];facts=[p['facts'] for p in rows]
        te=l.FrequencyTfidf().fit(texts,[3,2,4]);fe=l.FixedFacts('C_BOTH').fit(facts)
        x=sparse.hstack([te.transform(texts),fe.transform(facts)],format='csr')
        labels=np.array([1,1,1,2,2,0,0,0,0]);ids=np.array([0,0,0,1,1,2,2,2,2])
        model,receipt=l.fit_aggregated(x,ids,labels,1.)
        self.assertEqual(receipt['sum_weights'],9.)
        bundle={'model':model,'text_encoder':te,'fact_encoder':fe}
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'model.joblib';joblib.dump(bundle,file);loaded=joblib.load(file)
            probabilities,pred=l.classify(loaded,texts,facts)
        np.testing.assert_allclose(probabilities,model.predict_proba(x),atol=1e-12)
        np.testing.assert_array_equal(pred,probabilities.argmax(1))

if __name__=='__main__':unittest.main()
