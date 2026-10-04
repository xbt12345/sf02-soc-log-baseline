"""Protocol-aware audit keys from the ACTUAL R0 facts; unknown never means same behavior."""
import json
import numpy as np
import pandas as pd
from v89_common import ROOT, OUT, data, read, save, sha
from v96_protocol_audit import DEST, V95


def observed_behavior(facts):
    required=['action','outcome','transport_protocol','src_role','dst_role']
    if any(k not in facts for k in required):return None,'missing_core_fact'
    if facts['src_role'] not in ['inside','outside','dmz'] or facts['dst_role'] not in ['inside','outside','dmz']:
        return None,'unresolved_zone_role'
    protocol=facts['transport_protocol']
    if protocol=='icmp':required+=['icmp_type','icmp_code'];limit=255
    elif protocol in ['tcp','udp']:required+=['dst_port_fixed'];limit=65535
    else:return None,'unsupported_protocol_contract'
    for k in required[5:]:
        value=facts.get(k)
        if not isinstance(value,int) or isinstance(value,bool) or not 0<=value<=limit:
            return None,'missing_or_invalid_protocol_fact'
    return json.dumps({k:facts[k] for k in required},sort_keys=True),'complete_observed_tuple'


def main():
    assert not (DEST/'behavior_contract_audit.json').exists()
    assert observed_behavior({})[0] is None
    base={'action':'deny','outcome':'blocked','transport_protocol':'icmp','src_role':'outside','dst_role':'dmz','icmp_type':3,'icmp_code':13}
    assert observed_behavior(base)[0] is not None
    assert observed_behavior({k:v for k,v in base.items() if k!='icmp_code'})[0] is None
    tcp={k:v for k,v in base.items() if not k.startswith('icmp_')};tcp.update(transport_protocol='tcp',dst_port_fixed=65536)
    assert observed_behavior(tcp)[0] is None
    tcp['dst_port_fixed']=771;assert observed_behavior(tcp)[0] is not None
    r,y,fid,_,old,_,fit=data()
    facts=[json.loads(s) for s in pd.read_parquet(OUT/'projections.parquet').facts]
    keys=[observed_behavior(f) for f in facts]
    roles=pd.read_parquet(V95/'full_format_roles.parquet').role.to_numpy()
    # Eligibility computed ONLY on A/B. V/inner/C/H are not consulted here.
    take=r.route.eq('asa').to_numpy()&np.isin(roles,['A','B'])
    frame=r.loc[take,['row_position','component','projection_id','label_index']].copy()
    frame['role']=roles[take]
    frame['behavior']=[keys[p][0] for p in frame.projection_id]
    frame['parse_status']=[keys[p][1] for p in frame.projection_id]
    coverage=frame.groupby(['role','parse_status','label_index']).size().rename('rows').reset_index()
    coverage.to_csv(DEST/'AB_behavior_contract_coverage.csv',index=False)
    support=frame[frame.behavior.notna()].groupby(['behavior','role','label_index']).agg(rows=('row_position','size'),components=('component','nunique')).reset_index()
    support.to_csv(DEST/'AB_observed_behavior_support.csv',index=False)
    eligible=[]
    for key,g in support.groupby('behavior'):
        cells={(z.role,z.label_index):(z.rows,z.components) for z in g.itertuples()}
        if all((a,c) in cells for a in ['A','B'] for c in [1,2]):
            eligible.append({'behavior':key,'cells':g.drop(columns='behavior').to_dict('records'),
                'has_two_components_per_role_class':all(cells[(a,c)][1]>=2 for a in ['A','B'] for c in [1,2])})
    # Pure format routing, frozen predictions; post-hoc diagnostic, not validation selection.
    base_pred=np.load(V95/'E10_step200_prediction.npy')[fid]
    compose=np.where(r.route.eq('asa').to_numpy(),base_pred,old[fid])
    scope=[]
    for role,mask in [('inner',r.fold.eq(1).to_numpy()),('C',r.fold.eq(2).to_numpy()),('H',r.fold.eq(0).to_numpy()),('fit',fit)]:
        c=compose[mask];truth=y[mask];o=old[fid[mask]];p=base_pred[mask]
        scope.append({'role':role,'old_errors':int((o!=truth).sum()),'E10_errors':int((p!=truth).sum()),
            'scope_restored_errors':int((c!=truth).sum()),'repairs_vs_E10':int(((p!=truth)&(c==truth)).sum()),
            'regressions_vs_E10':int(((p==truth)&(c!=truth)).sum()),
            'negative_flips_vs_old':int(((o==truth)&(c!=truth)).sum()),
            'positive_flips_vs_old':int(((o!=truth)&(c==truth)).sum()),
            'class_support':[int((truth==cl).sum()) for cl in range(3)],
            'class_correct':[int(((truth==cl)&(c==truth)).sum()) for cl in range(3)]})
    save(DEST/'frozen_nonASA_scope_control.json',{'status':'posthoc_diagnostic_not_promotion','new_classifier_fits':0,
        'policy':'Use E10 only for existing ASA route, historic v85 elsewhere; routing does not inspect labels.',
        'results':scope,'limitations':'Historical development diagnosis; V excluded because old teacher saw V. This is not a new unseen validation or deployable promotion.'})
    result={'status':'protocol_aware_contract_audited_no_fit','source_sha256':sha(__file__),
        'A_B_ASA_rows':len(frame),'complete_observed_tuple_rows':int(frame.behavior.notna().sum()),
        'incomplete_rows_preserved_in_main_training':True,'incomplete_rows_eligible_for_behavior_auxiliary':False,
        'eligible_AB_both_class_groups':eligible,
        'minimum_two_components_per_role_class_groups':sum(a['has_two_components_per_role_class'] for a in eligible),
        'new_classifier_fits':0,'new_calibration_fits':0,
        'scope':'Behavior support, not a sufficient condition for stable M/S labels. Two-component screen is a diagnostic for repeatability, not a proved optimal statistical threshold.'}
    save(DEST/'behavior_contract_audit.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
