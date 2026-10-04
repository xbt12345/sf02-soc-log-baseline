"""Evidence-preserving adapter over the frozen v39 runtime, Python 3.8+.

Only documented legacy Meraki pattern actions, with all or literal-IP dst scope,
are restored. Device decisions are observations, never security labels.
Raw evidence, unresolved data and wrapper identities stay outside model input.
"""
import hashlib
import ipaddress
import re
import v39_core as old

VERSION = 'v48-evidence-preserving-input-1.0'
SOURCE = 'https://documentation.meraki.com/Platform_Management/Dashboard_Administration/Operate_and_Maintain/Monitoring_and_Reporting/Syslog_Server_Overview_and_Configuration'
NATIVE = re.compile(
    r'<(?P<priority>\d{1,3})>Original Address=(?P<collector>\S+)\s+1\s+'
    r'(?P<clock>\S+)\s+(?P<device>\S+)\s+(?P<kind>flows|l7_firewall)\s+'
    r'(?:(?P<action>deny|allow)\s+)?src=(?P<src_address>\S+)\s+dst=(?P<dst_address>\S+)'
    r'(?:\s+mac=(?P<mac>\S+))?\s+protocol=(?P<protocol>tcp|udp|icmp|icmpv6|[0-9]{1,3})'
    r'(?:\s+sport=(?P<src_port>\S+)\s+dport=(?P<dst_port>\S+))?'
    r'(?:\s+type=(?P<icmp_type>\S+)(?:\s+code=(?P<icmp_code>\S+))?)?'
    r'(?:\s+pattern:\s+(?P<pattern>.+)|\s+decision=(?P<decision>blocked|allowed))?\s*', re.I)
PATTERN = re.compile(r'(?P<action>0|1|allow|deny)\s+(?:all|dst\s+(?P<target>\S+))', re.I)
IDENTITIES = {'collector', 'clock', 'device', 'src_address', 'dst_address', 'mac', 'priority'}


def prepare_record(row):
    return prepare_message(old.previous.prior.base.string(row.get('message_sanitized')))


def prepare_message(raw):
    prior = old.previous.prior.prepare_message(raw)
    result = old.prepare_message(raw)
    result['facts'] = dict(result['facts'])
    result['route'] = prior['route']
    audit = {'raw_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
             'raw_characters': len(raw), 'parser_route': prior['route'],
             'disposition': 'unchanged_frozen_parser', 'fields': [],
             'raw_retained_by_original_row_reference': True,
             'complete_raw_semantics_claimed': False}
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
            record.update(status='unresolved' if conflict or (name == 'pattern' and not p) else 'modeled',
                          target_fields=['action', 'outcome'],
                          reason='conflicting_native_decisions' if conflict else 'unsupported_pattern_grammar' if name == 'pattern' and not p else 'documented_native_device_decision')
            if name == 'pattern' and p:
                record['interpretation_source'] = SOURCE
                record['decoded_action'] = decoded
        else:
            target = {'protocol': 'transport_protocol', 'src_port': 'src_port_fixed', 'dst_port': 'dst_port_fixed'}.get(name, name)
            present = target in result['facts'] and result['facts'][target] != 65536
            record.update(status='modeled' if present else 'unresolved', target_fields=[target],
                          reason='observed_typed_fact' if present else 'redacted_invalid_or_not_applicable')
        audit['fields'].append(record)
    audit['facts_before'] = before
    audit['facts_after'] = dict(result['facts'])
    return result


def fact_dispositions(facts):
    """Fail-visible inventory at the actual fixed encoder boundary."""
    numeric = set(old.learning.NUMERIC) | set(old.learning.BIT_FIELDS) | {'exit_category'}
    enums = dict(old.learning.ENUMS, **old.SEMANTIC_ENUMS)
    result = {}
    for key, value in facts.items():
        if key in enums:
            result[key] = 'encoded' if str(value).casefold() in enums[key] else 'unencoded_enum'
        elif key in numeric:
            result[key] = 'encoded'
        elif key == 'syscall_category':
            result[key] = 'audit_only_architecture_specific_opcode'
        else:
            result[key] = 'unencoded_key'
    return result


def classify_records(bundle, rows):
    """Training and inference call the identical raw-message adapter."""
    import numpy as np
    from scipy import sparse
    if bundle.get('adapter_version') != VERSION:
        raise ValueError('Model requires a different input adapter')
    records = [prepare_record(row) for row in rows]
    for r in records:
        if any(s.startswith('unencoded') for s in fact_dispositions(r['facts']).values()):
            raise ValueError('Unaccounted facts at model boundary')
    x = sparse.hstack([bundle['text_encoder'].transform([r['text'] for r in records]),
                      bundle['fact_encoder'].transform([r['facts'] for r in records])], format='csr')
    return np.asarray(bundle['model'].predict_proba(x)), [r['information_audit'] for r in records]
