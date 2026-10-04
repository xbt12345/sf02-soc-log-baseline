"""Recover literal firewall facts; CSV positions follow PAN-OS public schema.

Neither the traffic action nor the device severity is a ground-truth class.
Unmodelled fields remain in original raw text and in the source archive.
"""
import csv
import io
import ipaddress
import re
from v78_denial_adapter import parse as v1_parse, port as strict_port, self_check as v1_check, HEADER as SENTENCE_HEADER

HEADER = re.compile(r'^<\d{1,3}>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+(.*)$', re.I)
DATE = re.compile(r'(?:\d{4}|USER-\d+)/\d{2}/\d{2} \d{2}:\d{2}:\d{2}')
ACTION = {'deny':'deny','drop':'deny','drop-all-packets':'deny','block-url':'deny',
          'reset-client':'deny','reset-server':'deny','reset-both':'deny',
          'allow':'allow','alert':'alert'}
SOURCES = ['https://docs.paloaltonetworks.com/ngfw/help/12-2/traffic-log-fields',
           'https://docs.paloaltonetworks.com/ngfw/help/12-2/url-filtering-log-fields']


def port(token):
    # A partial redaction destroys the numeric value. Preserve missingness;
    # never concatenate surviving digits or decode the marker as a port.
    if re.fullmatch(r'(?:[0-9]+|(?:CRED|USER|HOST|ORG)-[0-9]+(?:-[0-9]+)*)+',token) and re.search(r'(?:CRED|USER|HOST|ORG)-',token):
        return 65536
    return strict_port(token)


def parse(raw):
    if not isinstance(raw, str): return None
    result = v1_parse(raw)
    if result: return result
    # A distinct, explicitly named reason suffix; do not accept arbitrary prose.
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
        acl = re.fullmatch(r'Deny (tcp|udp) src (inside|outside|dmz):([^/\s]+)/([^\s]+) dst (inside|outside|dmz):([^/\s]+)/([^\s]+) by \S+-group "[^"\r\n]+" \[[^\]\r\n]+\]',h_acl['body'],re.I)
        if acl:
            try:
                ipaddress.ip_address(acl[3]);ipaddress.ip_address(acl[6])
                ports=[port(acl[4]),port(acl[7])]
            except ValueError: return None
            facts={'transport_protocol':acl[1].lower(),'action':'deny','outcome':'blocked',
                   'src_role':acl[2].lower(),'dst_role':acl[5].lower()}
            for name,n in zip(('src_port','dst_port'),ports):
                facts[name+'_fixed']=n
                if n!=65536: facts[name+'_range']='system' if n<=1023 else 'user' if n<=49151 else 'dynamic'
            return {'facts':facts,'audit':{'rule':'literal_Deny_src_dst_roles_ACL', 'does_not_assign_label':True}}
    h = HEADER.fullmatch(raw.strip())
    if not h: return None
    try:
        rows = list(csv.reader(io.StringIO(h[1]), strict=True))
        if len(rows) != 1: return None
        f = rows[0]
        if len(f) < 46 or f[0] != '1' or f[3] not in ('TRAFFIC','THREAT'): return None
        if not DATE.fullmatch(f[1]) or not DATE.fullmatch(f[6]): return None
        if not f[15].startswith('vsys') or not re.fullmatch(r'0x[0-9a-fA-F]+', f[28]): return None
        if f[29] not in ('tcp','udp','icmp','icmp6','gre') or f[30] not in ACTION: return None
        ipaddress.ip_address(f[7]); ipaddress.ip_address(f[8])
        ports = [port(f[24]), port(f[25])]
    except (ValueError, csv.Error): return None
    facts = {'action':ACTION[f[30]]}
    if f[29]!='gre': facts['transport_protocol']=f[29]  # GRE has no corresponding frozen encoder category.
    if facts['action'] in ('deny','allow'):
        facts['outcome'] = 'blocked' if facts['action']=='deny' else 'allowed'
    for name,n in zip(('src_port','dst_port'),ports):
        # ICMP identifier/type fields must never be represented as TCP/UDP ports.
        if f[29] not in ('tcp','udp'): n = 65536
        facts[name+'_fixed'] = n
        if n != 65536: facts[name+'_range'] = 'system' if n<=1023 else 'user' if n<=49151 else 'dynamic'
    return {'facts':facts, 'audit':{'rule':'panos_csv_shared_fields_v1','type':f[3],
        'subtype':f[4], 'protocol_literal':f[29], 'action_literal':f[30], 'src_zone_not_assumed_role':f[16],
        'dst_zone_not_assumed_role':f[17], 'csv_fields':len(f),
        'ports_field_indices':[24,25], 'does_not_assign_label':True,'schema_sources':SOURCES}}


def self_check():
    n=v1_check()
    f=['']*46
    for k,v in {0:'1',1:'USER-9564/03/11 12:32:40',3:'TRAFFIC',4:'drop',
        6:'2020/03/11 12:32:40',7:'10.1.1.1',8:'10.2.2.2',15:'vsys1',
        24:'54321',25:'443',28:'0x0',29:'tcp',30:'deny',11:'name,with,commas'}.items(): f[k]=v
    def make(fields):
        buf=io.StringIO();csv.writer(buf).writerow(fields)
        return '<14>Mar 11 12:32:40 HOST-123 '+buf.getvalue().strip()
    x=parse(make(f)); assert x['facts']['dst_port_fixed']==443 and x['facts']['src_port_fixed']==54321
    assert 'src_role' not in x['facts'] and 'label' not in x['facts']
    bad=f.copy();bad[24]='65536';assert parse(make(bad)) is None
    bad=f.copy();bad.insert(10,'unexpected');assert parse(make(bad)) is None
    bad=f.copy();bad[25]='CRED-12345';assert parse(make(bad))['facts']['dst_port_fixed']==65536
    bad=f.copy();bad[29]='icmp';assert parse(make(bad))['facts']['src_port_fixed']==65536
    bad=f.copy();bad[30]='allow';assert parse(make(bad))['facts']['outcome']=='allowed'
    bad=f.copy();bad[3]='THREAT';bad[30]='block-url';assert parse(make(bad))['facts']['action']=='deny'
    bad=f.copy();bad[7]='10.3.3.3';bad[1]='USER-1234/03/12 01:01:01';assert parse(make(bad))['facts']==x['facts']
    assert parse('quoted='+make(f)) is None
    assert port('4CRED-23673')==65536
    assert port('12CRED-1234CRED-5678')==65536
    try: port('443garbage')
    except ValueError: pass
    else: raise AssertionError('invalid port accepted')
    bad=f.copy();bad[29]='gre';assert parse(make(bad))['facts']['dst_port_fixed']==65536
    acl='<164>Jun 18 USER-9546 08:54:59: HOST-123 Deny tcp src inside:10.1.1.1/CRED-123 dst outside:10.2.2.2/443 by ORG-123-group "inside_acl" [0xABC, 0xDEF]'
    assert parse(acl)['facts']['src_port_fixed']==65536 and parse(acl)['facts']['dst_role']=='outside'
    assert parse(acl.replace('/443','/65536')) is None
    return n+16
