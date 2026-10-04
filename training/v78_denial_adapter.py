"""Literal syslog inbound/outbound denial facts; no labels or source identity."""
import ipaddress
import re

HEADER=re.compile(r'^<\d{1,3}>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(?:(?:\d{4}|USER-\d+)\s+)?\d{2}:\d{2}:\d{2}:?\s+\S+\s+(?P<body>Deny\s+.*)\s*$',re.I)
BODY=re.compile(r'Deny\s+(inbound|outbound)\s+(TCP|UDP)\s+from\s+([^\s/]+)/([^\s/]+)\s+to\s+([^\s/]+)/([^\s/]+)\s+on\s+([^\r\n]+)',re.I)
MARKER=re.compile(r'(?:USER|HOST|CRED|ORG)-\d+(?:-\d+)*',re.I)


def port(token):
    if MARKER.fullmatch(token) or token=='-':return 65536
    if token.isdecimal() and 0<=int(token)<=65535:return int(token)
    raise ValueError('Invalid port token')


def parse(raw):
    if not isinstance(raw,str):return None
    head=HEADER.fullmatch(raw.strip())
    if not head:return None
    m=BODY.fullmatch(head['body'].strip())
    if not m:return None
    try:
        for addr in [m[3],m[5]]:
            if not MARKER.fullmatch(addr):ipaddress.ip_address(addr)
        ports=[port(m[4]),port(m[6])]
    except ValueError:return None
    facts={'action':'deny','outcome':'blocked','transport_protocol':m[2].lower()}
    for name,n in zip(['src_port','dst_port'],ports):
        facts[name+'_fixed']=n
        if n!=65536:facts[name+'_range']='system' if n<=1023 else 'user' if n<=49151 else 'dynamic'
    return {'facts':facts,'audit':{'direction_observed_not_encoded':m[1].lower(),
        'interface_text_not_assumed_src_or_dst_role':m[7],'body_span':[head.start('body'),head.end('body')],
        'rule':'literal_syslog_Deny_direction_protocol_from_endpoint_to_endpoint_on_interface',
        'does_not_assign_label':True}}


def self_check():
    raw='<162>Jan 26 USER-9564 11:13:40: USER-0010-0344 Deny inbound UDP from 10.100.8.199/58169 to 10.100.8.200/9443 on ORG-9508 outside'
    x=parse(raw);assert x['facts']['src_port_fixed']==58169 and x['facts']['dst_port_fixed']==9443
    assert x['facts']['transport_protocol']=='udp' and 'src_role' not in x['facts'] and 'dst_role' not in x['facts']
    assert parse(raw.replace('/9443','/65536')) is None
    assert parse(raw.replace('10.100.8.199','999.100.8.199')) is None
    assert parse(raw.replace('Deny inbound','Permit inbound')) is None
    assert parse('description="'+raw+'"') is None
    assert parse(raw.replace('/58169','/CRED-12345'))['facts']['src_port_fixed']==65536
    assert parse(raw.replace('UDP','TCP').replace('/9443','/443'))['facts']['dst_port_fixed']==443
    assert parse(raw.replace('10.100.8.199','10.10.0.1'))['facts']==x['facts']
    return 9
