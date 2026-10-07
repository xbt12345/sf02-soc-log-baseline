"""Fixed raw-log feature decoder; input-only dependency closure."""
import copy
import json
import re
from . import native_routes as prior
VIEWS = ('A_SEMANTIC', 'B_DESTINATION', 'C_BOTH')
GENERATED = {'asa', 'asa_acl', 'asa_protocol', 'native_flow', 'native_firewall', 'vpc_v2', 'syslog_auth'}
PORT_NAMES = ('src_port', 'dst_port')

def port_range(n):
    if n is None:
        return None
    return 'system' if n <= 1023 else 'user' if n <= 49151 else 'dynamic'

def project(text, facts, route):
    """Transform a verified v3.7 projection. No source identity is returned."""
    facts = json.loads(facts) if isinstance(facts, str) else copy.deepcopy(facts)
    removed = {}
    ports = {}
    for name in PORT_NAMES:
        encoded = facts.pop(name + '_category', None)
        n = None
        if encoded is not None:
            token = encoded.rsplit('|', 1)[-1]
            if not re.fullmatch('[0-9]+', token) or not 0 <= int(token) <= 65535:
                raise ValueError('Invalid cached port category')
            n = int(token)
        ports[name] = n
    for name in ('category',):
        if name in facts:
            removed[name] = facts.pop(name)
    proto = facts.pop('protocol', None)
    if proto is not None:
        if proto in ('tcp', 'udp', 'icmp', 'icmp6') or re.fullmatch('ipproto_[0-9]{1,3}', proto):
            facts['transport_protocol'] = proto
        elif re.fullmatch('(?:tlsv?|https?)[0-9.]*', proto):
            facts['application_protocol'] = proto
        else:
            removed['uninterpreted_protocol'] = proto
    for name in ('src_role', 'dst_role'):
        if facts.get(name) in ('unknown', 'other_interface'):
            removed[name] = facts.pop(name)
    if route in GENERATED:
        text = ''
    elif route == 'authentication':
        text = re.sub('\\b(?:reason|result|factor)\\s+', '', text)
    elif route == 'bounded_payload':
        text = re.sub('\\bbehaviors\\.\\[\\]\\.(?:parent_details\\.)?(parent_)?(cmdline|filename|filepath)\\s+', lambda m: ('parent ' if m[1] else '') + {'cmdline': 'command', 'filename': 'file', 'filepath': 'path'}[m[2]] + ' ', text)
        text = re.sub('^message\\s+', '', text)
    return {'text': text.strip(), 'facts': facts, 'ports': ports, 'removed_audit': removed}

def view_record(projected, view):
    if view not in VIEWS:
        raise ValueError('Unknown v3.8 view')
    facts = dict(projected['facts'])
    for name, n in projected['ports'].items():
        if n is not None:
            facts[name + '_range'] = port_range(n)
    if view in ('B_DESTINATION', 'C_BOTH'):
        facts['dst_port_fixed'] = 65536 if projected['ports']['dst_port'] is None else projected['ports']['dst_port']
    if view == 'C_BOTH':
        facts['src_port_fixed'] = 65536 if projected['ports']['src_port'] is None else projected['ports']['src_port']
    return {'text': projected['text'], 'facts': facts}
