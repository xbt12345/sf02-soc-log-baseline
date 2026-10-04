"""Additive literal within-record observations; no absolute time or class rule."""
import re


def value(token):
    if not re.fullmatch(r'[0-9]+',token):return None
    n=int(token)
    return n if n<=2**63-1 else None


def derive_vpc_observations(raw, parser_route):
    result={'values':{},'states':{},'spans':{},'scope':'Observed differences/ratios only; not attack truth or verified anonymization preservation.'}
    if parser_route!='vpc_v2' or not isinstance(raw,str):
        result['states']['grammar']='not_applicable';return result
    tokens=list(re.finditer(r'\S+',raw))
    if len(tokens)!=14 or tokens[0][0]!='2' or tokens[12][0] not in ('ACCEPT','REJECT','-'):
        result['states']['grammar']='mismatch';return result
    fields={k:value(tokens[i][0]) for k,i in [('packets',8),('bytes',9),('start',10),('end',11)]}
    result['spans']={k:[tokens[i].start(),tokens[i].end()] for k,i in [('packets',8),('bytes',9),('start',10),('end',11)]}
    start,end=fields['start'],fields['end'];duration=None
    if start is None or end is None:
        result['states']['interval']='unavailable_literal'
    elif end<start:
        result['states']['interval']='invalid_order'
    else:
        duration=end-start;result['values']['observed_interval_seconds']=duration
        result['states']['interval']='observed'
    packets,bytecount=fields['packets'],fields['bytes']
    if packets is not None and packets>0 and bytecount is not None:
        result['values']['observed_bytes_per_packet']=bytecount/packets
        result['states']['bytes_per_packet']='observed'
    else:result['states']['bytes_per_packet']='unavailable_or_zero_denominator'
    for key,count in [('packets_per_second',packets),('bytes_per_second',bytecount)]:
        if count is not None and duration is not None and duration>0:
            result['values']['observed_'+key]=count/duration;result['states'][key]='observed'
        else:result['states'][key]='unavailable_or_zero_interval'
    return result
