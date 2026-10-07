"""Fixed raw-log feature decoder; input-only dependency closure."""
import csv
import io
import ipaddress
import re
from .denial import parse as v1_parse, port as strict_port, HEADER as SENTENCE_HEADER
HEADER = re.compile('^<\\d{1,3}>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+\\d{1,2}\\s+\\d{2}:\\d{2}:\\d{2}\\s+\\S+\\s+(.*)$', re.I)
DATE = re.compile('(?:\\d{4}|USER-\\d+)/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2}')
ACTION = {'deny': 'deny', 'drop': 'deny', 'drop-all-packets': 'deny', 'block-url': 'deny', 'reset-client': 'deny', 'reset-server': 'deny', 'reset-both': 'deny', 'allow': 'allow', 'alert': 'alert'}
SOURCES = ['https://docs.paloaltonetworks.com/ngfw/help/12-2/traffic-log-fields', 'https://docs.paloaltonetworks.com/ngfw/help/12-2/url-filtering-log-fields']

def port(token):
    if re.fullmatch('(?:[0-9]+|(?:CRED|USER|HOST|ORG)-[0-9]+(?:-[0-9]+)*)+', token) and re.search('(?:CRED|USER|HOST|ORG)-', token):
        return 65536
    return strict_port(token)

def parse(raw):
    if not isinstance(raw, str):
        return None
    result = v1_parse(raw)
    if result:
        return result
    if raw.rstrip().endswith(' due to DNS Response'):
        result = v1_parse(raw.rstrip()[:-len(' due to DNS Response')] + ' on __reason_placeholder__')
        if result:
            result['audit'].pop('interface_text_not_assumed_src_or_dst_role')
            result['audit']['observed_reason_not_encoded'] = 'DNS Response'
            result['audit']['rule'] = 'literal_syslog_Deny_direction_protocol_endpoints_DNS_Response'
            result['audit']['body_span'][1] = len(raw.rstrip())
            return result
    h_acl = SENTENCE_HEADER.fullmatch(raw.strip())
    if h_acl:
        acl = re.fullmatch('Deny (tcp|udp) src (inside|outside|dmz):([^/\\s]+)/([^\\s]+) dst (inside|outside|dmz):([^/\\s]+)/([^\\s]+) by \\S+-group "[^"\\r\\n]+" \\[[^\\]\\r\\n]+\\]', h_acl['body'], re.I)
        if acl:
            try:
                ipaddress.ip_address(acl[3])
                ipaddress.ip_address(acl[6])
                ports = [port(acl[4]), port(acl[7])]
            except ValueError:
                return None
            facts = {'transport_protocol': acl[1].lower(), 'action': 'deny', 'outcome': 'blocked', 'src_role': acl[2].lower(), 'dst_role': acl[5].lower()}
            for name, n in zip(('src_port', 'dst_port'), ports):
                facts[name + '_fixed'] = n
                if n != 65536:
                    facts[name + '_range'] = 'system' if n <= 1023 else 'user' if n <= 49151 else 'dynamic'
            return {'facts': facts, 'audit': {'rule': 'literal_Deny_src_dst_roles_ACL', 'does_not_assign_label': True}}
    h = HEADER.fullmatch(raw.strip())
    if not h:
        return None
    try:
        rows = list(csv.reader(io.StringIO(h[1]), strict=True))
        if len(rows) != 1:
            return None
        f = rows[0]
        if len(f) < 46 or f[0] != '1' or f[3] not in ('TRAFFIC', 'THREAT'):
            return None
        if not DATE.fullmatch(f[1]) or not DATE.fullmatch(f[6]):
            return None
        if not f[15].startswith('vsys') or not re.fullmatch('0x[0-9a-fA-F]+', f[28]):
            return None
        if f[29] not in ('tcp', 'udp', 'icmp', 'icmp6', 'gre') or f[30] not in ACTION:
            return None
        ipaddress.ip_address(f[7])
        ipaddress.ip_address(f[8])
        ports = [port(f[24]), port(f[25])]
    except (ValueError, csv.Error):
        return None
    facts = {'action': ACTION[f[30]]}
    if f[29] != 'gre':
        facts['transport_protocol'] = f[29]
    if facts['action'] in ('deny', 'allow'):
        facts['outcome'] = 'blocked' if facts['action'] == 'deny' else 'allowed'
    for name, n in zip(('src_port', 'dst_port'), ports):
        if f[29] not in ('tcp', 'udp'):
            n = 65536
        facts[name + '_fixed'] = n
        if n != 65536:
            facts[name + '_range'] = 'system' if n <= 1023 else 'user' if n <= 49151 else 'dynamic'
    return {'facts': facts, 'audit': {'rule': 'panos_csv_shared_fields_v1', 'type': f[3], 'subtype': f[4], 'protocol_literal': f[29], 'action_literal': f[30], 'src_zone_not_assumed_role': f[16], 'dst_zone_not_assumed_role': f[17], 'csv_fields': len(f), 'ports_field_indices': [24, 25], 'does_not_assign_label': True, 'schema_sources': SOURCES}}
