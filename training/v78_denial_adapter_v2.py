"""Recover literal firewall facts; CSV positions follow PAN-OS public schema.

Neither the traffic action nor the device severity is a ground-truth class.
Unmodelled fields remain in original raw text and in the source archive.
"""
import csv
import io
import ipaddress
import re
from v78_denial_adapter import parse as v1_parse, port, self_check as v1_check

HEADER = re.compile(r'^<\d{1,3}>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+(.*)$', re.I)
DATE = re.compile(r'(?:\d{4}|USER-\d+)/\d{2}/\d{2} \d{2}:\d{2}:\d{2}')
ACTION = {'deny':'deny','drop':'deny','drop-all-packets':'deny','block-url':'deny',
          'reset-client':'deny','reset-server':'deny','reset-both':'deny',
          'allow':'allow','alert':'alert'}
SOURCES = ['https://docs.paloaltonetworks.com/ngfw/help/12-2/traffic-log-fields',
           'https://docs.paloaltonetworks.com/ngfw/help/12-2/url-filtering-log-fields']


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
    h = HEADER.fullmatch(raw.strip())
    if not h: return None
    try:
        rows = list(csv.reader(io.StringIO(h[1]), strict=True))
        if len(rows) != 1: return None
        f = rows[0]
        if len(f) < 46 or f[0] != '1' or f[3] not in ('TRAFFIC','THREAT'): return None
        if not DATE.fullmatch(f[1]) or not DATE.fullmatch(f[6]): return None
        if not f[15].startswith('vsys') or not re.fullmatch(r'0x[0-9a-fA-F]+', f[28]): return None
        if f[29] not in ('tcp','udp','icmp','icmp6') or f[30] not in ACTION: return None
        ipaddress.ip_address(f[7]); ipaddress.ip_address(f[8])
        ports = [port(f[24]), port(f[25])]
    except (ValueError, csv.Error): return None
    facts = {'transport_protocol':f[29], 'action':ACTION[f[30]]}
    if facts['action'] in ('deny','allow'):
        facts['outcome'] = 'blocked' if facts['action']=='deny' else 'allowed'
    for name,n in zip(('src_port','dst_port'),ports):
        # ICMP identifier/type fields must never be represented as TCP/UDP ports.
        if f[29] not in ('tcp','udp'): n = 65536
        facts[name+'_fixed'] = n
        if n != 65536: facts[name+'_range'] = 'system' if n<=1023 else 'user' if n<=49151 else 'dynamic'
    return {'facts':facts, 'audit':{'rule':'panos_csv_shared_fields_v1','type':f[3],
        'subtype':f[4], 'action_literal':f[30], 'src_zone_not_assumed_role':f[16],
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
    return n+10
