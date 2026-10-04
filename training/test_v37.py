import json
import unittest
import numpy as np
import v37_representation as r
import v37_learning as learn
BODY='Deny tcp src outside:10.1.2.3/12345 dst dmz-1:10.2.3.4/443 by access-group "acl" [0x0, 0x0]'
CEF='CEF:0|vendor|WAF|1|119|detector|1|act=DENY app=TLSv1.3 spt=54941 msg=a\\=1'
class Invariants(unittest.TestCase):
    def test_cef_clock_formats_have_identical_model_input(self):
        keys=[r.prepare_message(p+CEF)['b2_key'] for p in ['', '<190>Jul 26 06:10:11 host ','<190>Jul 26 USER-CRED-25046 host ','<190>Jul 26 - newhost ']]
        self.assertEqual(len(set(keys)),1)
    def test_cef_scalar_decode_shared(self):
        self.assertEqual(r.prepare_message(CEF)['b2_key'],r.prepare_message(CEF.replace('act=DENY','act="DENY"').replace('app=TLSv1.3','app="TLSv1.3"'))['b2_key'])
    def test_cef_equivalent_escape(self):
        self.assertEqual(r.prepare_message(CEF)['b2_key'],r.prepare_message(CEF.replace('msg=a\\=1','msg="a=1"'))['b2_key'])
    def test_cef_header_and_order(self):
        a=r.prepare_message(CEF);b=r.prepare_message('CEF:0|new|NEW|999|2|normal|9|msg=a\\=1 spt=54941 app=TLSv1.3 act=DENY')
        self.assertEqual(a['b2_key'],b['b2_key'])
    def test_cef_field_presence_no_text_duplication(self):
        v=r.prepare_message(CEF)
        self.assertNotIn('src_port',v['b1']);self.assertNotIn('54941',v['b1'])
        self.assertEqual(v['facts']['src_port_category'],'tlsv1.3|54941')
    def test_actual_action_change_is_not_forced_invariant(self):
        self.assertNotEqual(r.prepare_message(CEF)['b2_key'],r.prepare_message(CEF.replace('DENY','ALLOW'))['b2_key'])
    def test_quoted_attack_is_not_outer_cef(self):
        v=r.prepare_message('request body: '+CEF)
        self.assertFalse(v['supported']);self.assertEqual(v['b1'],'')
    def test_malformed_cef_never_raw_fallback(self):
        for raw in [CEF+' act=ALLOW','CEF:0|v|w|1|2|name|1|msg="unterminated act=DENY']:
            v=r.prepare_message(raw);self.assertEqual(v['route'],'unsupported');self.assertEqual(v['facts'],{})
    def test_untrusted_nested_content_not_promoted(self):
        v=r.prepare_message(json.dumps({'request':{'message':CEF}}))
        self.assertFalse(v['supported'])
    def test_invalid_priority_rejected(self):
        self.assertEqual(r.prepare_message('<999>Jul 26 12:00:00 host '+CEF)['route'],'unsupported')
    def test_asa_canonical_semantics_and_port_evidence(self):
        v=r.prepare_message(BODY);w=r.prepare_message(BODY.replace('dmz-1','dmz-99').replace('"acl"','"bad"'))
        self.assertEqual(v['b2_key'],w['b2_key'])
        self.assertNotIn('12345',v['b1']);self.assertEqual(v['facts']['src_port_category'],'tcp|12345')
        self.assertNotEqual(v['b2_key'],r.prepare_message(BODY.replace('/443','/445'))['b2_key'])
    def test_redacted_port_not_reconstructed(self):
        v=r.prepare_message(BODY.replace('/12345','/4CRED-123'))
        self.assertNotIn('src_port_category',v['facts']);self.assertNotIn('123',v['b1'])
    def test_empty_and_unknown_not_security_label(self):
        for raw in ['', 'opaque corrupt record 2024-01-01 malicious']:
            v=r.prepare_message(raw);self.assertFalse(v['supported']);self.assertEqual(v['facts'],{})
    def test_outer_columns_cannot_be_read(self):
        class Row(dict):
            def get(self,key,*args):
                assert key=='message_sanitized'
                return CEF
        self.assertEqual(r.prepare_record(Row())['b2_key'],r.prepare_message(CEF)['b2_key'])
    def test_json_payload_not_metadata(self):
        x={'message':'A user quoted malicious in a request','timestamp':'2024','sensitivity':'normal'}
        a=r.prepare_message(json.dumps(x))
        x.update(timestamp='2099',sensitivity='suspicious')
        self.assertEqual(a['b2_key'],r.prepare_message(json.dumps(x))['b2_key'])
        self.assertIn('malicious',a['b1'])
    def test_windows_metadata_and_outcome(self):
        x={'message':'An account failed to log on.\nStatus: 0xc000006d','winlog':{}}
        a=r.prepare_message(json.dumps(x));x['timestamp']=0;x['product_name']='malicious'
        self.assertEqual(a['b2_key'],r.prepare_message(json.dumps(x))['b2_key'])
        self.assertEqual(a['facts']['outcome'],'failure')
    def test_pam_missing_clock_invariant(self):
        body='sshd[99]: pam_unix(sshd:session): session opened for user USER-1 by (uid=0)'
        self.assertEqual(r.prepare_message('Dec 24 03:28:10 host '+body)['b2_key'],r.prepare_message('Dec 24 USER-99 host '+body)['b2_key'])
    def test_vpc_timestamp_account_interface_not_predictors(self):
        a='2 100000000006 eni-aaa 10.0.0.1 10.0.0.2 50000 443 6 1 40 1234567890 1234567891 REJECT OK'
        b=a.replace('100000000006','999999999999').replace('eni-aaa','eni-new').replace('1234567890','9999999999').replace('1234567891','CRED-999')
        self.assertEqual(r.prepare_message(a)['b2_key'],r.prepare_message(b)['b2_key'])
        self.assertEqual(r.prepare_message(a)['facts']['action'],'deny')
    def test_vpc_custom_shape_not_guessed(self):
        self.assertEqual(r.prepare_message('2 123 eni-aaa 1 2 3 REJECT OK')['route'],'unsupported')
    def test_native_flow_pattern_not_truth(self):
        a='<134>Original Address=10.0.0.1 1 1234.567 host flows src=10.0.0.1 dst=10.0.0.2 protocol=tcp sport=40000 dport=443 pattern: 1 all'
        self.assertEqual(r.prepare_message(a)['b2_key'],r.prepare_message(a.replace('pattern: 1 all','pattern: normal'))['b2_key'])

class Learning(unittest.TestCase):
    def test_no_generic_operator_feature(self):
        a=learn.word_tokens('normal a=1 && b=2')
        self.assertNotIn('op_equals',a);self.assertIn('syntax:a=1',a)
    def test_fact_ports_are_nominal_no_presence_or_ordinal(self):
        f=learn.fact_dictionary({'src_port_category':'tcp|50000','bytes':0,'http_status':200})
        self.assertFalse(any(k.startswith('observed:') for k in f))
        self.assertEqual(f['category:src_port_category'],'tcp|50000')
        with self.assertRaises(ValueError):learn.fact_dictionary({'src_port':50000})
    def test_no_holdout_vocabulary_or_scaling(self):
        e=learn.FrequencyTfidf().fit(['alpha beta','beta'],[2,1])
        names=e.names().tolist();e.transform(['targetsecret'])
        self.assertEqual(names,e.names().tolist());self.assertNotIn('targetsecret',names)
        f=learn.FactEncoder().fit([{'src_port_category':'tcp|443'}])
        self.assertEqual(f.transform([{'src_port_category':'tcp|9999'}]).nnz,0)
    def test_empty_vocab_is_explicit_zero_information(self):
        e=learn.FrequencyTfidf().fit([''],[9]);self.assertEqual(e.transform(['unexpected']).shape,(1,0))
    def test_np_support_and_ties(self):
        self.assertFalse(learn.np_threshold(np.zeros(2994))['computable'])
        r=learn.np_threshold(np.zeros(2995))
        self.assertTrue(r['computable']);self.assertLessEqual(r['binomial_tail'],.05)
        self.assertGreater(r['threshold'],0);self.assertFalse(r['formal_guarantee_accepted'])
    def test_gating_separates_detection_from_subtype(self):
        p=np.array([[.6,.1,.3],[.99,.009,.001]])
        self.assertEqual(p.argmax(1).tolist(),[0,0])
        self.assertEqual(learn.gated_predictions(p,.2).tolist(),[2,0])
    def test_count_aggregation_preserves_original_loss_weights(self):
        from scipy import sparse
        x=sparse.csr_matrix([[1.,0.],[0.,1.],[1.,1.]])
        model,receipt=learn.fit_aggregated(x,np.array([0,0,1,1,2,2]),np.array([0,1,1,2,2,0]),.1)
        self.assertEqual(receipt['sum_weights'],6.)
import v37_representation as rep

class LiteralNativeFacts(unittest.TestCase):
    def test_normal_acl_denial_preserves_observed_fact(self):
        raw='<164>Jul 26 USER-9546 06:08:51: host TCP ORG-1738 denied by ACL from 100.64.56.162/USER-1234 to outside:100.64.56.163/22'
        v=rep.prepare_message(raw)
        self.assertEqual(v['route'],'asa_acl');self.assertEqual(v['facts']['action'],'deny')
        self.assertEqual(v['facts']['dst_port_category'],'tcp|22')
        self.assertEqual(v['facts']['src_role'],'unknown')
        self.assertNotIn('src_port_category',v['facts'])
    def test_firewall_decision_is_outcome_not_attack_truth(self):
        raw='<134>Original Address=192.0.2.1 1 USER-1234 host l7_firewall src=192.0.2.2 dst=192.0.2.3 protocol=tcp sport=55555 dport=443 decision=blocked'
        v=rep.prepare_message(raw)
        self.assertEqual(v['facts']['outcome'],'blocked');self.assertNotIn('malicious',str(v))
        self.assertNotEqual(v['b2_key'],rep.prepare_message(raw.replace('decision=blocked','decision=allowed'))['b2_key'])
    def test_audit_time_and_identity_do_not_replace_operations(self):
        a='<134>Jul 26 00:00:00 host audispd[22]: node=node1 type=SYSCALL msg=audit(123:456): arch=c000003e syscall=2 success=no exit=-13 exe="/usr/bin/cat" a0=1234'
        b=a.replace('00:00:00','USER-9999').replace('node1','node2').replace('123:456','999:888').replace('a0=1234','a0=9999')
        v=rep.prepare_message(a)
        self.assertEqual(v['b2_key'],rep.prepare_message(b)['b2_key'])
        self.assertEqual(v['facts']['outcome'],'failure');self.assertIn('/usr/bin/cat',v['b1'])
    def test_report_profile_does_not_enter_format_payload(self):
        a='<110>Jul 26 00:00:00 host ADAuditPlus: [ Category = alert ] [ REPORT_PROFILE = malware ] [ FORMAT_MESSAGE = Member was added to group. ]'
        b=a.replace('malware','normal').replace('alert','other')
        self.assertEqual(rep.prepare_message(a)['route'],'format_payload')
        self.assertEqual(rep.prepare_message(a)['b2_key'],rep.prepare_message(b)['b2_key'])
    def test_partially_redacted_collector_clock_not_an_admission_gate(self):
        a='<134>2024-07-26T06:10:17.632106-05:00 host audispd[22]: type=ORG-1234 msg=audit(123:456): exe="/usr/bin/chmod" exit=0'
        b=a.replace('2024-07-26T06:10:17.632106-05:00','USER-9999-07-26T06:USER-8888-05:00')
        v=rep.prepare_message(a)
        self.assertEqual(v['b2_key'],rep.prepare_message(b)['b2_key'])
        self.assertIn('chmod',v['b1']);self.assertNotIn('operation_record',v['facts']);self.assertNotIn('outcome',v['facts'])
    def test_icmp_type_is_not_forced_into_tcp_port_layout(self):
        a='<134>Original Address=192.0.2.1 1 USER-1234 host flows src=192.0.2.2 dst=192.0.2.3 protocol=icmp type=8 pattern: 1 all'
        v=rep.prepare_message(a);self.assertEqual(v['facts']['icmp_type'],8)
        self.assertFalse(any('port' in k for k in v['facts']))
    def test_numeric_protocol_and_explicit_flow_denial(self):
        prefix='<134>Original Address=192.0.2.1 1 USER-1234 host '
        a=prefix+'flows deny src=192.0.2.2 dst=192.0.2.3 mac=AA:BB:CC:DD:EE:FF protocol=tcp sport=60000 dport=443'
        v=rep.prepare_message(a);self.assertEqual(v['facts']['action'],'deny')
        self.assertEqual(v['b2_key'],rep.prepare_message(a.replace('AA:BB:CC:DD:EE:FF','11:22:33:44:55:66'))['b2_key'])
        b=prefix+'flows src=192.0.2.2 dst=192.0.2.3 protocol=47 pattern: 1 all'
        self.assertEqual(rep.prepare_message(b)['facts']['protocol'],'ipproto_47')
    def test_numeric_asa_protocol_no_port_invention(self):
        a='<164>Aug 26 2022 19:33:08: host Deny protocol 47 src outside:192.0.2.1 dst dmz-1:192.0.2.2 by ORG-1738-group "acl" [0x0, 0x0]'
        v=rep.prepare_message(a);self.assertEqual(v['facts']['protocol'],'ipproto_47');self.assertEqual(v['facts']['dst_role'],'dmz')
if __name__=='__main__':unittest.main()
