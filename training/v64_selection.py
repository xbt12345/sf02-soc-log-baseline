"""Constrained model selection. Frozen comparators, explicit rejection, no fallback winner."""
import math


PROTECTED = {'M_row': .01, 'M_source': .01, 'row_macro': .005,
             'tcp_M_source': .01, 'udp_M_source': .01}


def flatten(metrics):
    result = {'M_row':metrics['ASA']['recall_B_M_S'][1],
              'M_source':metrics['M_source_recall'],
              'row_macro':metrics['ASA']['macro_f1_M_S'],
              'S_source':metrics['S_source_recall']}
    for protocol in ['tcp', 'udp']:
        for label, name in [('1','M'),('2','S')]:
            result[protocol+'_'+name+'_source'] = metrics['hard'][protocol][label]
    if any(not isinstance(v, (float,int)) or not math.isfinite(v) or not 0 <= v <= 1
           for v in result.values()):
        raise ValueError('All required metric cells must have finite supported values in [0,1]')
    return result


def select(curve, references, min_S_gain=.05):
    if not references:
        raise ValueError('Frozen references are required; a deteriorated candidate cannot define its own floor')
    refs = {name:flatten(m) for name,m in references.items()}
    floor = {k:max(r[k] for r in refs.values())-loss for k,loss in PROTECTED.items()}
    evaluated=[]; feasible=[]
    for item in curve:
        m=flatten(item['metrics'])
        violations={k:{'actual':m[k],'minimum':limit} for k,limit in floor.items() if m[k]<limit-1e-12}
        evaluated.append({'step':item['step'],'violations':violations})
        if not violations: feasible.append(item)
    if not feasible:
        return {'selected':None,'screen_passed':False,'status':'no_feasible_checkpoint',
                'floors':floor,'evaluated':evaluated}
    chosen=max(feasible,key=lambda item:(flatten(item['metrics'])['S_source'],
               item['metrics']['source_balanced'],flatten(item['metrics'])['row_macro'],-item['step']))
    m=flatten(chosen['metrics']); S_anchor=max(r['S_source'] for r in refs.values())
    deltas=[m[p+'_S_source']-max(r[p+'_S_source'] for r in refs.values()) for p in ['tcp','udp']]
    passed=m['S_source']>=S_anchor+min_S_gain-1e-12 and min(deltas)>=-1e-12 and max(deltas)>1e-12
    return {'selected':chosen,'screen_passed':bool(passed),
            'status':'screen_passed_requires_replication' if passed else 'feasible_but_insufficient_gain',
            'S_source_anchor':S_anchor,'minimum_S_gain':min_S_gain,'floors':floor,'evaluated':evaluated}
