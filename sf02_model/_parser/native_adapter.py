"""Fixed raw-log feature decoder; input-only dependency closure."""
import hashlib
import ipaddress
import re
from . import semantics as old
SOURCE = 'https://documentation.meraki.com/Platform_Management/Dashboard_Administration/Operate_and_Maintain/Monitoring_and_Reporting/Syslog_Server_Overview_and_Configuration'
NATIVE = re.compile('<(?P<priority>\\d{1,3})>Original Address=(?P<collector>\\S+)\\s+1\\s+(?P<clock>\\S+)\\s+(?P<device>\\S+)\\s+(?P<kind>flows|l7_firewall)\\s+(?:(?P<action>deny|allow)\\s+)?src=(?P<src_address>\\S+)\\s+dst=(?P<dst_address>\\S+)(?:\\s+mac=(?P<mac>\\S+))?\\s+protocol=(?P<protocol>tcp|udp|icmp|icmpv6|[0-9]{1,3})(?:\\s+sport=(?P<src_port>\\S+)\\s+dport=(?P<dst_port>\\S+))?(?:\\s+type=(?P<icmp_type>\\S+)(?:\\s+code=(?P<icmp_code>\\S+))?)?(?:\\s+pattern:\\s+(?P<pattern>.+)|\\s+decision=(?P<decision>blocked|allowed))?\\s*', re.I)
PATTERN = re.compile('(?P<action>0|1|allow|deny)\\s+(?:all|dst\\s+(?P<target>\\S+))', re.I)
IDENTITIES = {'collector', 'clock', 'device', 'src_address', 'dst_address', 'mac', 'priority'}

def prepare_record(row):
    return prepare_message(old.previous.prior.base.string(row.get('message_sanitized')))

def prepare_message(raw):
    prior = old.previous.prior.prepare_message(raw)
    result = old.prepare_message(raw)
    result['facts'] = dict(result['facts'])
    result['route'] = prior['route']
    audit = {'raw_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(), 'raw_characters': len(raw), 'parser_route': prior['route'], 'disposition': 'unchanged_frozen_parser', 'fields': [], 'raw_retained_by_original_row_reference': True, 'complete_raw_semantics_claimed': False}
    result['information_audit'] = audit
    if prior['route'] not in ('native_flow', 'native_firewall'):
        return result
    m = NATIVE.fullmatch(raw.strip())
    if m is None:
        raise ValueError('Frozen native parser and adapter grammar disagree')
    offset = len(raw) - len(raw.lstrip())
    explicit = m['action'].casefold() if m['action'] else None
    decision = {'blocked': 'deny', 'allowed': 'allow'}.get((m['decision'] or '').casefold())
    p = PATTERN.fullmatch(m['pattern'].strip()) if m['pattern'] else None
    if p and p['target']:
        try:
            ipaddress.ip_address(p['target'])
        except ValueError:
            p = None
    decoded = {'0': 'allow', '1': 'deny', 'allow': 'allow', 'deny': 'deny'}[p['action'].casefold()] if p else None
    observations = [v for v in (explicit, decision, decoded) if v is not None]
    conflict = len(set(observations)) > 1
    before = dict(result['facts'])
    if conflict:
        result['facts'].pop('action', None)
        result['facts'].pop('outcome', None)
        audit['disposition'] = 'conflicting_actions_withheld'
    elif decoded is not None:
        result['facts'].update(action=decoded, outcome='blocked' if decoded == 'deny' else 'allowed')
        audit['disposition'] = 'restored_documented_action' if 'action' not in before else 'corroborated_action'
    elif m['pattern']:
        audit['disposition'] = 'unresolved_pattern_preserved_in_audit'
    else:
        audit['disposition'] = 'explicit_native_action'
    for name, value in m.groupdict().items():
        if value is None:
            continue
        a, b = m.span(name)
        record = {'field': name, 'span': [a + offset, b + offset], 'raw': raw[a + offset:b + offset]}
        assert record['raw'] == value
        if name in IDENTITIES:
            record.update(status='audit_only', reason='identity_or_absolute_wrapper_not_risk_evidence')
        elif name == 'kind':
            record.update(status='audit_only', reason='format_selects_decoder_not_security_label')
        elif name in ('pattern', 'action', 'decision'):
            record.update(status='unresolved' if conflict or (name == 'pattern' and (not p)) else 'modeled', target_fields=['action', 'outcome'], reason='conflicting_native_decisions' if conflict else 'unsupported_pattern_grammar' if name == 'pattern' and (not p) else 'documented_native_device_decision')
            if name == 'pattern' and p:
                record['interpretation_source'] = SOURCE
                record['decoded_action'] = decoded
        else:
            target = {'protocol': 'transport_protocol', 'src_port': 'src_port_fixed', 'dst_port': 'dst_port_fixed'}.get(name, name)
            present = target in result['facts'] and result['facts'][target] != 65536
            record.update(status='modeled' if present else 'unresolved', target_fields=[target], reason='observed_typed_fact' if present else 'redacted_invalid_or_not_applicable')
        audit['fields'].append(record)
    audit['facts_before'] = before
    audit['facts_after'] = dict(result['facts'])
    return result
