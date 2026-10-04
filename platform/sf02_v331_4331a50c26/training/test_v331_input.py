"""Adversarial boundaries for v3.3.1, not synthetic model quality tests."""
import json
import unittest

import v331_prepare as p

BODY = 'Deny tcp src outside:100.64.1.2/12345 dst dmz-1:10.2.3.4/443 by access-group "outside_acl" [0x0, 0x0]'
AUTH = {"application":{"key":"APP-SECRET","name":"test"},"user":{"name":"PERSON"},
        "txid":"SECRET","reason":"invalid_passcode","result":"denied",
        "adaptive_trust_assessments":{"trust_level":"NORMAL","reason":"not an attack","model_version":"2022.1"},
        "timestamp":1720000000}


def wrap(obj):
    return "ORG-1780 ::: payload="+json.dumps(obj)+" ::: tags=[irrelevant]"


class InputContracts(unittest.TestCase):
    def test_headers_do_not_change_body(self):
        prefixes = ["<164>Jul 23 2022 09:02:26: USER-0010-0324 ",
                    "<180>Jul 23 USER-9564 09:02:26: USER-0010-0324 ",
                    "<180>2026-09-12T12:00:00Z collector.example ",
                    "<180>Sep 12 12:00:00 collector ",
                    ""]
        texts=[p.prepare_record({"message_sanitized":q+BODY})["text"] for q in prefixes]
        self.assertEqual(len(set(texts)),1)

    def test_event_code_is_not_collection_metadata(self):
        a=p.prepare_record({"message_sanitized":"%ASA-4-106023: "+BODY})["text"]
        b=p.prepare_record({"message_sanitized":"%ASA-4-106015: "+BODY})["text"]
        self.assertIn("106023",a);self.assertIn("native_severity 4",a)
        self.assertNotEqual(a,b)

    def test_quoted_or_unverified_header_does_not_get_cut(self):
        for head in ['message="', 'request body: ', 'Sep 12 12:00:00 malicious payload pretending collector: ']:
            self.assertIsNone(p.asa_parts(head+BODY))

    def test_unknown_body_is_not_guessed(self):
        self.assertIsNone(p.asa_parts("<164>Sep 12 12:00:00 collector Deny tcp no connection"))
        self.assertIsNone(p.asa_parts(BODY.replace("dst","destination")))

    def test_body_acl_text_with_deny_or_time_is_preserved(self):
        raw=BODY.replace("outside_acl","deny time test")
        self.assertIn('deny time test',p.prepare_record({"message_sanitized":raw})["text"])

    def test_complete_ports_and_direction_remain_distinct(self):
        a=p.prepare_record({"message_sanitized":BODY})["text"]
        for b in [BODY.replace("/443","/445"),BODY.replace("src outside:","src inside:")]:
            self.assertNotEqual(a,p.prepare_record({"message_sanitized":b})["text"])

    def test_redacted_port_is_not_completed_from_digits(self):
        a=p.prepare_record({"message_sanitized":BODY.replace("/12345","/CRED-250250")})["text"]
        self.assertIn("/unknown_port",a);self.assertNotIn("250250",a)

    def test_template_masks_values_but_preserves_direction_protocol(self):
        a=p.template_of(p.asa_parts(BODY))
        self.assertEqual(a,p.template_of(p.asa_parts(BODY.replace("/443","/445"))))
        self.assertNotEqual(a,p.template_of(p.asa_parts(BODY.replace("tcp","udp"))))
        self.assertNotEqual(a,p.template_of(p.asa_parts(BODY.replace("src outside:","src inside:"))))

    def test_auth_upstream_and_identity_invariance(self):
        obj=dict(AUTH);obj["user"]={"name":"DIFFERENT"};obj["application"]={"key":"OTHER"}
        obj["txid"]="OTHER";obj["adaptive_trust_assessments"]={"trust_level":"CRITICAL","model_version":"2099"}
        a=p.prepare_record({"message_sanitized":wrap(AUTH)})
        b=p.prepare_record({"message_sanitized":wrap(obj)})
        self.assertEqual(a["text"],b["text"])
        self.assertIn("denied",a["text"]);self.assertNotIn("critical",b["text"])
        self.assertNotIn("normal",a["text"])

    def test_auth_field_order_is_irrelevant(self):
        self.assertEqual(p.prepare_record({"message_sanitized":wrap(AUTH)})["text"],
                         p.prepare_record({"message_sanitized":wrap(dict(reversed(list(AUTH.items()))))})["text"])

    def test_unknown_result_does_not_become_success(self):
        obj=dict(AUTH);obj["result"]="ORG-0893";obj["reason"]="user_USER-28756"
        value=p.prepare_record({"message_sanitized":wrap(obj)})
        self.assertTrue(value["authentication_result_unknown"])
        self.assertTrue(value["no_observable_auth_facts"])
        self.assertNotIn("success",value["text"]);self.assertNotIn("0893",value["text"])

    def test_known_auth_result_is_a_fact_not_forced_class(self):
        obj=dict(AUTH);obj["result"]="success"
        self.assertNotEqual(p.prepare_record({"message_sanitized":wrap(AUTH)})["text"],
                            p.prepare_record({"message_sanitized":wrap(obj)})["text"])

    def test_invalid_bare_time_does_not_require_json_repair(self):
        raw=wrap(AUTH).replace("1720000000","172CRED-1000")
        self.assertEqual(p.prepare_record({"message_sanitized":raw})["text"],
                         p.prepare_record({"message_sanitized":wrap(AUTH)})["text"])

    def test_nested_application_payload_not_promoted_to_auth_root(self):
        self.assertIsNone(p.auth_object(json.dumps({"request":AUTH})))

    def test_damaged_final_identity_does_not_destroy_complete_facts(self):
        raw=wrap(AUTH)
        obj={k:v for k,v in AUTH.items() if k!="user"}
        damaged="payload="+json.dumps(obj)[:-1]+',"user":{"groups":["incomplete"}}'
        value=p.prepare_record({"message_sanitized":damaged})
        self.assertEqual(value["text"],p.prepare_record({"message_sanitized":raw})["text"])
        self.assertTrue(any(s.get("path")=="unparsed_identity_tail" for s in value["repair_spans"]))

    def test_damaged_behavior_field_is_not_guessed_as_identity(self):
        obj={k:v for k,v in AUTH.items() if k not in {"user","reason"}}
        raw="payload="+json.dumps(obj)[:-1]+',"reason":{"value":["incomplete"}}'
        self.assertIsNone(p.auth_object(raw))

    def test_duplicate_root_keys_fail_closed(self):
        raw=wrap(AUTH).replace('"result": "denied"','"result":"denied","result":"success"')
        self.assertIsNone(p.auth_object(raw))

    def test_label_and_source_columns_cannot_change_preparation(self):
        for raw in [BODY,wrap(AUTH),'{"event":{"code":4625},"message":"failure"}']:
            rows=[{"message_sanitized":raw,"label_binary":label,"product_name":label,"timestamp":label}
                  for label in ["benign","malicious","suspicious"]]
            self.assertEqual(len({p.prepare_record(r)["text"] for r in rows}),1)


if __name__=="__main__":
    unittest.main(verbosity=2)
