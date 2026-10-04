"""Recover only observed partial facts; never assign a threat class or decode redaction."""
import re
import ipaddress

MARKER=re.compile(r'(?:CRED|HOST|USER|ORG)-\d+(?:-\d+)*')
ASA=re.compile(r'\bDeny\s+(?P<protocol>tcp|udp)\s+src\s+(?P<src>[^\s:]+):(?P<sip>[^/\s]+)/(?P<sport>[^\s]+)\s+dst\s+(?P<dst>[^\s:]+):(?P<dip>[^/\s]+)/(?P<dport>[^\s]+)\s+by\s+[^\r\n]+?\[[^\]\r\n]+\]\s*$',re.I)
HEADER=re.compile(r'^<\d{1,3}>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(?:(?:\d{4}|USER-\d+)\s+)?\d{2}:\d{2}:\d{2}:?\s+\S+\s+',re.I)
EVENT=re.compile(r'"(?P<field>EventID|EventCode|event_id|code)"\s*:\s*"?(?P<value>\d{1,5})(?="?\s*[,}\]])',re.I)
PRIVILEGES=('SeAssignPrimaryTokenPrivilege','SeTcbPrivilege','SeSecurityPrivilege','SeTakeOwnershipPrivilege',
 'SeLoadDriverPrivilege','SeBackupPrivilege','SeRestorePrivilege','SeDebugPrivilege','SeAuditPrivilege',
 'SeSystemEnvironmentPrivilege','SeImpersonatePrivilege','SeDelegateSessionUserImpersonatePrivilege')


def port(token):
    if re.fullmatch(r'\d{1,5}',token) and 0<=int(token)<=65535:return int(token),'known'
    if MARKER.search(token):return 65536,'redacted'
    return 65536,'invalid_or_unknown'


def parse(raw,route):
    out={'facts':{},'observations':[],'states':{},'does_not_assign_label':True}
    if not isinstance(raw,str) or not raw:return out
    def add(field,value,a,b,state='known'):
        out['facts'][field]=value;out['states'][field]=state
        out['observations'].append({'field':field,'value':value,'span':[a,b],'literal':raw[a:b],'state':state})
    if route=='asa':
        h=HEADER.match(raw);m=ASA.fullmatch(raw[h.end():]) if h else None
        if m:
            offset=h.end()
            # Endpoint identity is only syntax-checked; it never becomes a feature.
            try:
                for k in ['sip','dip']:ipaddress.ip_address(m[k])
            except ValueError:return out
            for field,key in [('transport_protocol','protocol'),('src_role','src'),('dst_role','dst')]:
                token=m[key].lower();value=token
                if key in ['src','dst']:
                    known=re.fullmatch(r'(inside|outside|dmz)(?:-\d+)?',token)
                    if not known:continue
                    value=known[1]
                a,b=m.span(key);add(field,value,a+offset,b+offset)
            a=offset+m.start();add('action','deny',a,a+4);add('outcome','blocked',a,a+4)
            for field,key in [('src_port_fixed','sport'),('dst_port_fixed','dport')]:
                val,state=port(m[key]);a,b=m.span(key);add(field,val,a+offset,b+offset,state)
    elif route in ('windows_message','windows_rendered'):
        matches=[m for m in EVENT.finditer(raw) if m['field'].lower()!='code' or 'winlog' in raw.lower()]
        values={int(m['value']) for m in matches}
        if len(values)==1:
            m=matches[0];add('event_code_observed',int(m['value']),*m.span('value'))
        elif values:out['states']['event_code_observed']='conflicting_observations'
        for privilege in PRIVILEGES:
            m=re.search(r'(?:^|[^A-Za-z0-9_]|\\[nrt])(?P<value>'+privilege+r')(?![A-Za-z0-9_])',raw)
            if m:add('privilege_'+privilege,True,*m.span('value'))
    for obs in out['observations']:
        a,b=obs['span'];assert raw[a:b]==obs['literal']
    return out


def self_check():
    base='<164>Jun 18 USER-9546 08:54:59: HOST-123 Deny tcp src outside:10.1.1.1/5CRED-123 dst dmz-2:10.2.2.2/55292 by ORG-123-group "outside_acl" [0x0, 0x0]'
    p=parse(base,'asa');assert p['facts']['src_port_fixed']==65536 and p['states']['src_port_fixed']=='redacted'
    assert p['facts']['dst_port_fixed']==55292 and p['facts']['dst_role']=='dmz'
    assert parse(base.replace('5CRED-123','12345'),'asa')['facts']['src_port_fixed']==12345
    assert parse(base.replace('/55292','/65536'),'asa')['facts']['dst_port_fixed']==65536
    assert parse('quoted='+base,'asa')['facts']=={}
    assert parse(base.replace('Deny','Permit'),'asa')['facts']=={}
    assert parse(base.replace('10.1.1.1','999.1.1.1'),'asa')['facts']=={}
    assert parse(base,'unsupported')['facts']=={}
    a='{"winlog":{"code":"4672","record_id":1CRED-22,"PrivilegeList":"SeDebugPrivilege SeTcbPrivilege SeHOST-123Privilege"}}'
    q=parse(a,'windows_message');assert q['facts']=={'event_code_observed':4672,'privilege_SeTcbPrivilege':True,'privilege_SeDebugPrivilege':True}
    assert 'event_code_observed' not in parse(a.replace('4672','4CRED-12'),'windows_message')['facts']
    assert 'event_code_observed' not in parse(a.replace('4672','4672garbage'),'windows_message')['facts']
    assert 'event_code_observed' not in parse(a[:-1]+',"EventID":4624}', 'windows_message')['facts']
    assert parse(a,'unsupported')['facts']=={}
    assert parse(None,'asa')['facts']=={} and parse('','windows_message')['facts']=={}
    assert parse(a.replace('1CRED-22','123'),'windows_message')['facts']==q['facts']
    escaped=a.replace('SeDebugPrivilege SeTcbPrivilege',r'\n\tSeDebugPrivilege\n\tSeTcbPrivilege')
    assert parse(escaped,'windows_message')['facts']==q['facts']
    assert 'privilege_SeDebugPrivilege' not in parse(a.replace('SeDebugPrivilege','prefixSeDebugPrivilege'),'windows_message')['facts']
    assert parse(base.replace('HOST-123','HOST-456').replace('08:54:59','22:01:02'),'asa')['facts']==p['facts']
    assert all('label' not in o and 'class' not in o for o in p['observations']+q['observations'])
    return 20


if __name__=='__main__':print({'semantic_checks_passed':self_check()})
