import json
import unittest

import v36_representation as revised


class NewlyObservedRegressions(unittest.TestCase):
    def test_broken_late_native_object_does_not_hide_complete_body(self):
        raw='ORG-1 ::: {"event":{"provider":"Microsoft-Windows-Security-Auditing","code":"5156"},"CRED-1":"The Windows filtering platform has permitted a connection.\\nDestination Port: 443","sensitivity":"normal","winlog":{"keywords":["broken}}'
        r=revised.prepare_message(raw)
        self.assertEqual(r['route'],'windows_message')
        self.assertNotIn('sensitivity',r['b1'])
        self.assertIn('permitted a connection',r['b1'])
        self.assertIn('443',r['b1'])

    def test_unknown_json_root_metadata_is_quarantined_not_payload_words(self):
        a='{"sensitivity":"normal","message":"A user typed malicious in a search request"}'
        b=a.replace('"normal"','"suspicious"')
        self.assertEqual(revised.prepare_message(a)['b2_key'],revised.prepare_message(b)['b2_key'])
        self.assertIn('malicious',revised.prepare_message(a)['b1'])
    def test_native_event_meaning_requires_provider_and_does_not_create_target(self):
        raw='{"event":{"provider":"Microsoft-Windows-Security-Auditing","code":"4624"},"winlog":{}}'
        self.assertEqual(revised.native_auth_observation(raw)['outcome'],'success')
        self.assertIsNone(revised.native_auth_observation(raw.replace('Security-Auditing','Other-Application')))
        self.assertIsNone(revised.native_auth_observation(raw.replace('"4624"','"46CRED-1"')))
        self.assertIsNone(revised.native_auth_observation('{"payload":"4624"}'))

    def test_native_xml_layout_is_checked_and_result_conflict_stays_unknown(self):
        raw="<Provider Name='Microsoft-Windows-Security-Auditing' Guid='{54849625-5478-4994-a5ba-3e3b0328c30d}'/><USER-1ID>4624</USER-1ID><Version>2</Version>"
        obs=revised.native_auth_observation(raw,xml=True)
        self.assertEqual(obs['outcome'],'success')
        facts={'category':'authentication','outcome':'failure'};evidence=[]
        revised.add_native_auth(facts,evidence,obs)
        self.assertNotIn('outcome',facts)
        self.assertTrue(any(e['kind']=='conflicting_visible_outcomes' for e in evidence))
    def test_windows_identifier_changes_do_not_change_text_but_status_does(self):
        a='Handle ID: 0x640\nProcess ID: 0xb8c\nStatus: 0xC000006D\nSecurity ID: S-1-5-21-1-2-3-44'
        b=a.replace('0x640','0x99').replace('0xb8c','0x12').replace('S-1-5-21-1-2-3-44','S-1-5-21-9-8-7-66')
        self.assertEqual(revised.windows_body(a),revised.windows_body(b))
        self.assertNotEqual(revised.windows_body(a),revised.windows_body(a.replace('0xC000006D','0xC000006A')))

    def test_syslog_date_fraction_and_pid_are_collector_fields(self):
        body='pam_unix(sshd:session): session opened for user USER-0016 by (uid=0)'
        a='<86>USER-9546-07-26T07:10:02.71CRED-CRED-33212-04:00 USER-0010-0013 sshd[173663]: '+body
        b='<86>2099-08-01T10:20:30.123Z newhost sshd[99999]: '+body
        self.assertEqual(revised.prepare_message(a)['b2_key'],revised.prepare_message(b)['b2_key'])
        self.assertEqual(revised.prepare_message(a)['facts'],{'category':'session','action':'open'})

    def test_pam_failure_not_hard_coded_threat_label(self):
        x=revised.prepare_message('<86>2024-07-26T07:10:02Z host sshd[1]: pam_unix(sshd:auth): authentication failure; user=ORG-1')
        self.assertEqual(x['facts'],{'category':'authentication','outcome':'failure'})
        self.assertNotIn('label',x)

    def test_rendered_xml_keeps_body_not_timestamp_wrapper(self):
        raw="<USER-1 xmlns='something'><Time date='2024-01-01'/><RenderingORG-1 attr>An account was successfully logged on.\nStatus: 0x0\nProcess ID: 0x123</CRED-1></RenderingORG-1></USER-1>"
        a=revised.prepare_message(raw)
        b=revised.prepare_message(raw.replace('2024-01-01','2099-12-31').replace('0x123','0x999'))
        self.assertEqual(a['b2_key'],b['b2_key'])
        self.assertEqual(a['facts']['outcome'],'success')
        self.assertIn('status: 0x0',a['b1'])

    def test_rendered_xml_cannot_promote_nested_payload_or_multiple_blocks(self):
        self.assertIsNone(revised.rendered_windows("<x xmlns='x'><RenderingX><request>attack</request></RenderingX></x>"))
        self.assertIsNone(revised.rendered_windows("<x xmlns='x'><RenderingX>"+'text '*20+"</x><RenderingX>"+'text '*20+"</x></x>"))

    def test_integer_observations_have_one_numeric_spelling(self):
        self.assertEqual(str(revised.number('443')), '443')
import v36_representation as p

BODY = 'Deny tcp src outside:10.1.2.3/12345 dst dmz-1:10.2.3.4/443 by access-group "acl_name" [0x0, 0x0]'


class Representation(unittest.TestCase):
    def test_outer_metadata_is_never_read(self):
        class Row(dict):
            def get(self, key, *args):
                assert key == 'message_sanitized'
                return BODY
        self.assertEqual(p.prepare_record(Row())['b2_key'], p.prepare_record({'message_sanitized':BODY})['b2_key'])

    def test_asa_ids_dates_and_acl_names_cannot_change_model_views(self):
        a = p.prepare_message('<164>Jul 23 2022 09:00:00: USER-1 '+BODY)
        b = p.prepare_message('<164>Jul 26 USER-9546 06:42:12: USER-99 '+BODY.replace('10.1.2.3','100.64.3.4').replace('acl_name','evil_or_good'))
        self.assertEqual(a['b2_key'], b['b2_key'])

    def test_asa_port_roles_are_retained(self):
        a=p.prepare_message(BODY)
        b=p.prepare_message(BODY.replace('/443','/445'))
        self.assertNotEqual(a['b2_key'],b['b2_key'])
        self.assertEqual(a['facts']['src_port'],12345)

    def test_redacted_port_digits_are_not_recovered(self):
        a=p.prepare_message(BODY.replace('/12345','/5CRED-23898'))
        self.assertNotIn('src_port',a['facts'])
        self.assertNotIn('23898',a['b1'])

    def test_auth_failure_aligned_but_not_a_threat_label(self):
        raw=json.dumps({'application':{},'user':{},'txid':'a','reason':'invalid_passcode','result':'denied'})
        a=p.prepare_message(raw)
        self.assertEqual(a['facts']['category'],'authentication')
        self.assertEqual(a['facts']['outcome'],'failure')
        self.assertNotIn('suspicious',a['b2_key'])

    def test_unknown_auth_result_remains_unknown(self):
        raw=json.dumps({'application':{},'user':{},'txid':'a','reason':'valid_passcode','result':'ORG-89'})
        a=p.prepare_message(raw)
        self.assertNotIn('outcome',a['facts'])
        self.assertEqual(a['facts']['credential_check'],'valid')

    def test_auth_no_observation_is_explicit_not_benign(self):
        raw=json.dumps({'application':{},'user':{},'txid':'a','reason':'ORG-99','result':'ORG-89'})
        a=p.prepare_message(raw)
        self.assertTrue(a['no_observed_fact'])
        self.assertEqual(a['facts'],{})
        self.assertEqual(a['b1'],'')

    def test_upstream_auth_fields_do_not_change_inputs(self):
        x={'application':{},'user':{},'txid':'a','reason':'no_response','result':'denied'}
        a=p.prepare_message(json.dumps(x))
        x['adaptive_trust_assessments']={'reason':'confirmed benign','trust_level':'CRITICAL'}
        x['timestamp']=99999999
        b=p.prepare_message(json.dumps(x))
        self.assertEqual(a['b2_key'],b['b2_key'])

    def test_windows_failed_login_shared_fact(self):
        raw=json.dumps({'message':'An account failed to log on.\n\nStatus: 0xc000006d', 'winlog':{'event_data':{'Status':'0xc000006d'}}})
        a=p.prepare_message(raw)
        self.assertEqual(a['facts']['outcome'],'failure')
        self.assertIn('0xc000006d',a['b1'])

    def test_windows_metadata_and_field_order_invariance(self):
        x={'message':'An account failed to log on.\n\nStatus: 0xc000006d', 'winlog':{'event_data':{'Status':'0xc000006d'}}}
        a=p.prepare_message(json.dumps(x))
        x.update(product_name='suspicious',timestamp=0,agent={'version':'99.1'},sensitivity='normal')
        b=p.prepare_message(json.dumps(dict(reversed(list(x.items())))))
        self.assertEqual(a['b2_key'],b['b2_key'])

    def test_nested_payload_does_not_create_auth_fact(self):
        x={'winlog':{},'request':{'message':'An account failed to log on.'}}
        self.assertNotEqual(p.prepare_message(json.dumps(x))['route'],'windows_message')

    def test_quoted_auth_phrase_is_not_auth_outcome(self):
        x={'winlog':{},'message':'Command executed: echo "An account failed to log on."'}
        self.assertNotIn('outcome',p.prepare_message(json.dumps(x))['facts'])

    def test_cef_rules_and_header_severity_do_not_change_input(self):
        a='CEF:0|vendor|WAF|1|119|detector_A|1|act=DENY msg=[PRI * HTTP/2.0] cs4=Protocol Violations'
        b='CEF:0|other|OTHER|99|987|detector_B|9|act=DENY msg=[PRI * HTTP/2.0] cs4=benign'
        self.assertEqual(p.prepare_message(a)['b2_key'],p.prepare_message(b)['b2_key'])

    def test_cef_quoted_payload_equals_preserved_container_removed(self):
        raw='CEF:0|vendor|WAF|1|119|event|1|act=DENY msg="echo a=1 && x=2 | tool" src=10.1.1.1'
        a=p.prepare_message(raw)
        self.assertIn('a=1 && x=2 | tool',a['b1'])
        self.assertNotIn('act=',a['b1'])
        self.assertNotIn('src=',a['b1'])

    def test_bad_cef_quoted_value_does_not_get_guessed(self):
        raw='CEF:0|vendor|WAF|1|119|event|1|msg="unterminated act=DENY'
        self.assertIsNone(p.cef(raw))

    def test_cef_redacted_key_not_restored_but_literal_path_retained(self):
        a=p.prepare_message('CEF:0|vendor|WAF|1|119|event|1|CRED-12=/path?a=1 act=DENY')
        self.assertIn('/path?a=1',a['b1'])
        self.assertNotIn('CRED-12',a['b1'])

    def test_syslog_auth_anchor_and_duration(self):
        raw='<142>Jul 26 11:10:21 USER-1 USER-2: USER-3 Duration: 12:34:56 Count: 20; INFO: Authentication failure.'
        a=p.prepare_message(raw)
        self.assertEqual(a['facts']['duration_seconds'],45296)
        self.assertEqual(a['facts']['outcome'],'failure')
        self.assertIsNone(p.syslog_auth('request body: '+raw))

    def test_invalid_escape_is_not_repaired_to_unknown_semantics(self):
        self.assertEqual(p.literal_string('"text\\nC:\\Zbroken"'),'text\nC:\\Zbroken')


if __name__=='__main__':unittest.main(verbosity=2)
