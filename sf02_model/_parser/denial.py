"""Fixed raw-log feature decoder; input-only dependency closure."""
import ipaddress
import re
HEADER = re.compile('^<\\d{1,3}>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+\\d{1,2}\\s+(?:(?:\\d{4}|USER-\\d+)\\s+)?\\d{2}:\\d{2}:\\d{2}:?\\s+\\S+\\s+(?P<body>Deny\\s+.*)\\s*$', re.I)
BODY = re.compile('Deny\\s+(inbound|outbound)\\s+(TCP|UDP)\\s+from\\s+([^\\s/]+)/([^\\s/]+)\\s+to\\s+([^\\s/]+)/([^\\s/]+)\\s+on\\s+([^\\r\\n]+)', re.I)
MARKER = re.compile('(?:USER|HOST|CRED|ORG)-\\d+(?:-\\d+)*', re.I)

def port(token):
    if MARKER.fullmatch(token) or token == '-':
        return 65536
    if token.isdecimal() and 0 <= int(token) <= 65535:
        return int(token)
    raise ValueError('Invalid port token')

def parse(raw):
    if not isinstance(raw, str):
        return None
    head = HEADER.fullmatch(raw.strip())
    if not head:
        return None
    m = BODY.fullmatch(head['body'].strip())
    if not m:
        return None
    try:
        for addr in [m[3], m[5]]:
            if not MARKER.fullmatch(addr):
                ipaddress.ip_address(addr)
        ports = [port(m[4]), port(m[6])]
    except ValueError:
        return None
    facts = {'action': 'deny', 'outcome': 'blocked', 'transport_protocol': m[2].lower()}
    for name, n in zip(['src_port', 'dst_port'], ports):
        facts[name + '_fixed'] = n
        if n != 65536:
            facts[name + '_range'] = 'system' if n <= 1023 else 'user' if n <= 49151 else 'dynamic'
    return {'facts': facts, 'audit': {'direction_observed_not_encoded': m[1].lower(), 'interface_text_not_assumed_src_or_dst_role': m[7], 'body_span': [head.start('body'), head.end('body')], 'rule': 'literal_syslog_Deny_direction_protocol_from_endpoint_to_endpoint_on_interface', 'does_not_assign_label': True}}
