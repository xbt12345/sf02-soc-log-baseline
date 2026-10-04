"""Independent semantic and information-boundary counterexamples, no fit labels."""
import unittest
import numpy as np
import v48_input as v

def msg(tail='pattern: 1 all', action='', protocol='udp', ports='sport=1234 dport=53'):
    return '<134>Original Address=192.0.2.1 1 12345.67 MX flows '+action+'src=192.0.2.2 dst=192.0.2.3 protocol='+protocol+' '+ports+' '+tail

class InformationContract(unittest.TestCase):
    def test_equivalent_documented_actions_have_identical_vectors(self):
        groups = [['1', 'deny'], ['0', 'allow']]
        for group in groups:
            records = [v.prepare_message(msg('pattern: '+s+' all')) for s in group]
            records.append(v.prepare_message(msg('', action=('deny' if group[0]=='1' else 'allow')+' ')))
            records.append(v.prepare_message(msg('pattern: '+group[0]+' dst 192.0.2.3')))
            self.assertTrue(all(r['facts'] == records[0]['facts'] for r in records))
            e = v.old.SemanticFacts().fit([r['facts'] for r in records])
            x = e.transform([r['facts'] for r in records]).toarray()
            np.testing.assert_array_equal(x[0], x[1]); np.testing.assert_array_equal(x[0], x[2])
            np.testing.assert_array_equal(x[0], x[3])
        f = [v.prepare_message(msg('pattern: '+s+' all'))['facts'] for s in ['0','1','unknown']]
        x = v.old.SemanticFacts().fit(f).transform(f).toarray()
        self.assertFalse(np.array_equal(x[0], x[1])); self.assertFalse(np.array_equal(x[1], x[2]))

    def test_no_unbounded_numeric_or_quoted_action(self):
        for raw in ['message="pattern: 1 all"', msg('pattern: 10 all'), msg('pattern: 1 dst foo'),
                    msg('pattern: 1 all deny'), msg('pattern: CRED-1 all'), msg('pattern: not 1 all')]:
            self.assertNotEqual(v.prepare_message(raw)['facts'].get('action'), 'deny')

    def test_conflict_does_not_silently_choose(self):
        r = v.prepare_message(msg('pattern: 1 all', action='allow '))
        self.assertNotIn('action', r['facts']); self.assertNotIn('outcome', r['facts'])
        self.assertEqual(r['information_audit']['disposition'], 'conflicting_actions_withheld')
        self.assertTrue(any(f['status']=='unresolved' for f in r['information_audit']['fields']))

    def test_unknown_pattern_retains_independent_observation(self):
        r = v.prepare_message(msg('pattern: unknown', action='allow '))
        self.assertEqual(r['facts']['action'], 'allow')
        self.assertEqual(r['information_audit']['disposition'], 'unresolved_pattern_preserved_in_audit')

    def test_wrapper_and_external_fields_do_not_change_facts(self):
        raw = msg(); a = v.prepare_message(raw)
        changed = raw.replace('12345.67','2099-invalid-clock').replace('192.0.2.', '198.51.100.').replace(' MX ', ' OtherDevice ')
        b = v.prepare_message(changed)
        self.assertEqual((a['text'],a['facts']), (b['text'],b['facts']))
        c = v.prepare_record({'message_sanitized':raw, 'label_binary':'benign', 'product_name':'anything', 'timestamp':'2099', 'dst_ip':'x'})
        self.assertEqual(a['facts'], c['facts'])

    def test_original_spans_survive_outer_whitespace(self):
        raw = '  '+msg()+'  \n'; r=v.prepare_message(raw)
        for f in r['information_audit']['fields']:
            a,b=f['span']; self.assertEqual(raw[a:b],f['raw'])

    def test_actual_ports_and_unknown_are_distinct(self):
        a=v.prepare_message(msg())['facts']; b=v.prepare_message(msg(ports='sport=1234 dport=CRED-53'))['facts']
        self.assertEqual(a['dst_port_fixed'],53); self.assertEqual(b['dst_port_fixed'],65536)
        self.assertEqual(a['action'],b['action'])

    def test_encoder_unknown_value_is_visible(self):
        self.assertEqual(v.fact_dispositions({'action':'invented'})['action'], 'unencoded_enum')
        self.assertEqual(v.fact_dispositions({'new_fact':2})['new_fact'], 'unencoded_key')

if __name__ == '__main__': unittest.main()
