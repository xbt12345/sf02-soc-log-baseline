"""Message-only v3.8 projection; frozen v3.7 is an unchanged parser dependency.

Audit states do not enter prediction. Port precision is explicitly selected by
the model view; no target labels or fitted category vocabulary define a port.
"""
import copy
import functools
import json
import re
import v37_representation as prior

VERSION = 'v38-representation-1.0'
VIEWS = ('A_SEMANTIC', 'B_DESTINATION', 'C_BOTH')
GENERATED = {'asa','asa_acl','asa_protocol','native_flow','native_firewall','vpc_v2','syslog_auth'}
PORT_NAMES = ('src_port','dst_port')

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def port_range(n):
    if n is None:return None
    return 'system' if n<=1023 else 'user' if n<=49151 else 'dynamic'

def project(text, facts, route):
    """Transform a verified v3.7 projection. No source identity is returned."""
    facts = json.loads(facts) if isinstance(facts,str) else copy.deepcopy(facts)
    removed = {}; ports = {}
    for name in PORT_NAMES:
        encoded=facts.pop(name+'_category',None)
        n=None
        if encoded is not None:
            token=encoded.rsplit('|',1)[-1]
            if not re.fullmatch(r'[0-9]+',token) or not 0<=int(token)<=65535:
                raise ValueError('Invalid cached port category')
            n=int(token)
        ports[name]=n
    for name in ('category',):
        if name in facts:removed[name]=facts.pop(name)
    proto=facts.pop('protocol',None)
    if proto is not None:
        if proto in ('tcp','udp','icmp','icmp6') or re.fullmatch(r'ipproto_[0-9]{1,3}',proto):
            facts['transport_protocol']=proto
        elif re.fullmatch(r'(?:tlsv?|https?)[0-9.]*',proto):
            facts['application_protocol']=proto
        else:removed['uninterpreted_protocol']=proto
    for name in ('src_role','dst_role'):
        if facts.get(name) in ('unknown','other_interface'):
            removed[name]=facts.pop(name)
    # These routes were generated exclusively from structured facts: eliminate
    # duplicate fact names, category and decimal values from the lexical branch.
    if route in GENERATED:text=''
    elif route=='authentication':
        text=re.sub(r'\b(?:reason|result|factor)\s+', '',text)
    elif route=='bounded_payload':
        text=re.sub(r'\bbehaviors\.\[\]\.(?:parent_details\.)?(parent_)?(cmdline|filename|filepath)\s+',
                    lambda m:('parent ' if m[1] else '')+{'cmdline':'command','filename':'file','filepath':'path'}[m[2]]+' ',text)
        text=re.sub(r'^message\s+','',text)
    return {'text':text.strip(),'facts':facts,'ports':ports,'removed_audit':removed}

def view_record(projected, view):
    if view not in VIEWS:raise ValueError('Unknown v3.8 view')
    facts=dict(projected['facts'])
    for name,n in projected['ports'].items():
        if n is not None:facts[name+'_range']=port_range(n)
    if view in ('B_DESTINATION','C_BOTH'):
        facts['dst_port_fixed']=65536 if projected['ports']['dst_port'] is None else projected['ports']['dst_port']
    if view=='C_BOTH':
        facts['src_port_fixed']=65536 if projected['ports']['src_port'] is None else projected['ports']['src_port']
    return {'text':projected['text'],'facts':facts}

def token_state(token):
    if token is None:return {'state':'unobserved','value':None}
    token=str(token).strip().strip('"')
    if prior.base.MARKER.search(token):return {'state':'redacted','value':None}
    if token in ('','-'):return {'state':'unobserved','value':None}
    if re.fullmatch(r'[0-9]+',token) and int(token)<=65535:
        return {'state':'observed','value':int(token)}
    return {'state':'invalid','value':None}

def port_audit(raw, route, projected):
    result={name:{'state':'observed' if n is not None else 'unobserved','value':n}
            for name,n in projected['ports'].items()}
    if route=='asa':
        visible=prior.guard.asa_visible_facts(raw)
        if visible is None:raise ValueError('ASA cache no longer parses')
        for side,name in zip(('src','dst'),PORT_NAMES):
            f=visible[side];result[name]={'state':f['port_state'],'value':f['port']}
    elif route=='cef_fields':
        ext=raw[raw.find('CEF:'):].split('|',7)[-1]
        for key,name in zip(('spt','dpt'),PORT_NAMES):
            m=re.search(r'(?<!\S)'+key+r'=("[^"\n]*"|\S+)',ext)
            if m:result[name]=token_state(m[1])
    elif route=='vpc_v2':
        fields=raw.strip().split()
        for i,name in zip((5,6),PORT_NAMES):result[name]=token_state(fields[i])
    elif route in ('native_flow','native_firewall'):
        for key,name in zip(('sport','dport'),PORT_NAMES):
            m=re.search(r'\b'+key+r'=(\S+)',raw)
            if m:result[name]=token_state(m[1])
    elif route=='asa_acl':
        m=re.search(r'denied by ACL from \S+/(\S+) to [^:\s]+:\S+/(\S+)\s*$',raw,re.I)
        if m:
            for i,name in zip((1,2),PORT_NAMES):result[name]=token_state(m[i])
    return result

@functools.lru_cache(maxsize=512)
def prepare_message(raw):
    p=prior.prepare_message(raw)
    return project(p['b1'],p['facts'],p['route'])

def prepare_record(row):
    return prepare_message(prior.base.string(row.get('message_sanitized')))
