"""Versioned observed ICMP field extraction. No threat label or guessed redaction."""
import ipaddress
import re

MARKER = r'(?:USER|HOST|CRED|ORG)-\d+(?:-\d+)*'
HEADER = re.compile(r'^<\d{1,3}>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+'
                    r'(?:(?:\d{4}|' + MARKER + r')\s+)?(?:\d{2}:\d{2}:\d{2}|' + MARKER + r'):?\s+\S+\s+')
BODY = re.compile(r'Deny\s+(?P<protocol>icmp)\s+src\s+(?P<src>[^\s:]+):(?P<sip>\S+)\s+'
                  r'dst\s+(?P<dst>[^\s:]+):(?P<dip>\S+)\s+\(type\s+(?P<type>[^\s,]+),\s*code\s+(?P<code>[^\s)]+)\)'
                  r'\s+by\s+[^\s"]+-group\s+"[^"\r\n]*"\s+\[[^\]\r\n]+\]\s*', re.I)


def parse(raw, route='asa'):
    out = {'facts': {}, 'states': {}, 'observations': [], 'audit_zones': {},
           'format_matched': False, 'assigns_threat_label': False}
    if route != 'asa' or not isinstance(raw, str): return out
    header = HEADER.match(raw)
    if header is None: return out
    body = BODY.fullmatch(raw[header.end():])
    if body is None: return out
    # An endpoint may be visibly redacted, but arbitrary malformed addresses are rejected.
    for field in ['sip', 'dip']:
        value = body[field]
        try:
            if ipaddress.ip_address(value).version != 4: return out
        except ValueError:
            if re.search(MARKER, value) is None: return out
    out['format_matched'] = True
    def add(field, key, value, state='known'):
        start, end = body.span(key); start += header.end(); end += header.end()
        out['states'][field] = state
        if state == 'known': out['facts'][field] = value
        out['observations'].append({'field': field, 'value': value, 'state': state,
                                    'span': [start, end], 'literal': raw[start:end]})
    add('transport_protocol', 'protocol', 'icmp')
    # The outcome is a device observation, never a label for benign/M/S.
    out['facts']['action'] = 'deny'; out['states']['action'] = 'known'
    out['facts']['outcome'] = 'blocked'; out['states']['outcome'] = 'known'
    for field, key in [('src_role', 'src'), ('dst_role', 'dst')]:
        token = body[key].lower(); out['audit_zones'][field] = token
        match = re.fullmatch(r'(inside|outside|dmz)(?:-\d+)?', token)
        if match: add(field, key, match[1])
        else: add(field, key, None, 'unrecognized_zone')
    for field, key in [('icmp_type', 'type'), ('icmp_code', 'code')]:
        token = body[key]
        if re.fullmatch(r'\d{1,3}', token) and 0 <= int(token) <= 255:
            add(field, key, int(token))
        elif re.search(MARKER, token): add(field, key, None, 'redacted')
        else: add(field, key, None, 'invalid')
    return out


def self_check():
    raw = '<164>Jul 26 USER-9546 05:59:56: USER-0010-0324 Deny icmp src outside:100.64.54.242 dst dmz-2:10.190.117.96 (type 3, code 13) by ORG-1738-group "outside_ORG-1738_in" [0x0, 0x0]\n'
    p = parse(raw)
    assert p['facts']['icmp_type'] == 3 and p['facts']['icmp_code'] == 13
    assert p['facts']['dst_role'] == 'dmz' and p['audit_zones']['dst_role'] == 'dmz-2'
    assert parse(raw.replace('05:59:56:', 'USER-3849:'))['facts'] == p['facts']
    assert parse(raw.replace('ORG-1738', 'ORG-9999').replace('100.64.54.242', '192.0.2.9'))['facts'] == p['facts']
    for prefix in ['quoted=', 'message="', 'random ']: assert not parse(prefix + raw)['format_matched']
    assert not parse(raw.replace('Deny', 'Permit'))['format_matched']
    assert not parse(raw.replace('100.64.54.242', '999.1.1.1'))['format_matched']
    assert not parse(raw, 'unsupported')['format_matched']
    assert not parse(None)['format_matched']
    assert 'icmp_code' not in parse(raw.replace('code 13', 'code 1USER-555'))['facts']
    assert 'icmp_code' not in parse(raw.replace('code 13', 'code 999'))['facts']
    assert parse(raw.replace('type 3, code 13', 'type 8, code 0'))['facts']['icmp_type'] == 8
    assert parse(raw.replace('outside:100', 'dmz-1:100'))['facts']['src_role'] == 'dmz'
    assert 'src_role' not in parse(raw.replace('outside:100', 'unknown_zone:100'))['facts']
    assert all(raw[o['span'][0]:o['span'][1]] == o['literal'] for o in p['observations'])
    return {'passed': True, 'scope': 'typed observed facts, boundaries, redaction, identity invariance; no M/S rule'}
