"""Predeclared gates and paired factorial diagnosis; no fitting or tuning."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from run_v39_prepare import sha,save
from run_v40_train import metrics

PROBS=['p_benign','p_malicious','p_suspicious']
PARTS=['fit_pure_same_label','fit_pure_opposite_label','fit_conflicted','unseen_input']


def load(folder, verify_model=True):
    r=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    for n,k in [('model.joblib','model_sha256'),('evaluation.parquet','predictions_sha256'),('report.json','report_sha256')]:
        if n=='model.joblib' and not verify_model:continue
        assert sha(folder/n)==r[k]
    return pq.read_table(folder/'evaluation.parquet').to_pandas()


def main(a):
    root=Path(a.root);run=Path(a.run);stage=run/'prepared';out=run/('primary_review_stage1' if a.stage1 else 'primary_review')
    assert not out.exists();out.mkdir()
    config=json.loads((stage/'configuration.json').read_text(encoding='utf-8'))
    old=root/'artifacts/v39_local_r2_20260913';v40=root/'artifacts/v40_local_r1_20260913'
    tables=[]
    for fold in range(3):
        b=load(old/('primary/fold_%s/SEMANTIC'%fold));b['fold']=fold
        aux=load(v40/('primary/fold_%s/R'%fold),False)
        assert np.array_equal(b.row_position,aux.row_position)
        for col in ['baseline_partition','observation_mask','unseen_parameter','unseen_joint_value','unsupported_joint_value']:
            b[col]=aux[col].to_numpy()
        tables.append(b)
    b=pd.concat(tables,ignore_index=True).sort_values('row_position').reset_index(drop=True);del tables
    assert len(b)==1378650 and not b.row_position.duplicated().any()
    y=b.label_index.to_numpy(dtype=int);bp=b[PROBS].to_numpy();bc=bp.argmax(1)==y;asa=(b.route=='asa').to_numpy()
    native=set(pq.read_table(stage/'native_scope.parquet').column('row_position').to_pylist());target=b.row_position.isin(native).to_numpy()
    canary=set(r['row_position'] for r in json.loads((root/'evidence/2026-09-13/v41_planning/native_status_probe.json').read_text(encoding='utf-8'))['records'])
    critical=(b.route=='authentication').to_numpy()|((b['product']=='Barracuda WAF').to_numpy()&(y==2))|(b.route=='bounded_payload').to_numpy()|b.row_position.isin(canary).to_numpy()
    slices={'ASA':asa,'Duo':(b['product']=='Duo').to_numpy(),'Windows_native_target':target,
            'Windows_18_canary':b.row_position.isin(canary).to_numpy(),'native_flow':(b.route=='native_flow').to_numpy(),
            'WAF_suspicious':(b['product']=='Barracuda WAF').to_numpy()&(y==2)}
    for col in ['unseen_parameter','unseen_joint_value','unsupported_joint_value']:slices['ASA_'+col]=asa&b[col].to_numpy()
    for i,n in enumerate(PARTS):slices['ASA_'+n]=asa&(b.baseline_partition.to_numpy()==i)
    base=metrics(y,bp);result={'baseline':base,'baseline_slices':{n:metrics(y[s],bp[s]) for n,s in slices.items()},'candidates':{},'scope':'Repeatedly inspected official development, not blind or external'}
    slice_indexes=[]
    for grouping in [['route','label_index'],['route','observation_mask','label_index']]:
        for key,idx in b.groupby(grouping).indices.items():
            ix=np.asarray(idx);bodies=b.iloc[ix].body_group.nunique()
            if len(ix)<=config['small_slice_maximum_class_rows'] or bodies<=config['small_slice_maximum_class_body_keys']:
                slice_indexes.append((dict(zip(grouping,[int(v) if isinstance(v,(np.integer,int)) else v for v in key])),ix,int(bodies)))
    pair_outcomes={'B':~bc};pair_losses={'B':-np.log(np.maximum(bp[np.arange(len(y)),y],1e-300))}
    views=['P','F','D'] if a.stage1 else ['P','F','D','A']
    eligible=[]
    for view in views:
        parts=[load(run/('primary/fold_%s/%s'%(fold,view))) for fold in range(3)]
        d=pd.concat(parts,ignore_index=True).sort_values('row_position').reset_index(drop=True);del parts
        assert np.array_equal(d.row_position,b.row_position) and np.array_equal(d.label_index,y)
        assert np.array_equal(d.baseline_partition,b.baseline_partition)
        p=d[PROBS].to_numpy();correct=p.argmax(1)==y;m=metrics(y,p);pair_outcomes[view]=~correct
        pair_losses[view]=-np.log(np.maximum(p[np.arange(len(y)),y],1e-300))
        details={}
        for n,s in slices.items():
            details[n]={'metrics':metrics(y[s],p[s]),'fixed':int((s&~bc&correct).sum()),'regressed':int((s&bc&~correct).sum()),
                        'body_keys':int(b.loc[s,'body_group'].nunique()),'class_support':np.bincount(y[s],minlength=3).tolist()}
        tails=[]
        for key,ix,bodies in slice_indexes:
            tails.append({'slice':key,'rows':len(ix),'body_keys':bodies,'B_correct':int(bc[ix].sum()),'new_correct':int(correct[ix].sum()),
                          'fixed':int((~bc[ix]&correct[ix]).sum()),'regressed':int((bc[ix]&~correct[ix]).sum()),
                          'new_total_loss':bool(bc[ix].any() and not correct[ix].any())})
        route_class=[]
        non_target_regression=False
        for (route,c),ix0 in b.groupby(['route','label_index']).indices.items():
            ix=np.asarray(ix0);outside=ix[~target[ix]]
            rb=int((~bc[outside]).sum());rn=int((~correct[outside]).sum())
            non_target_regression |= rn>rb
            route_class.append({'route':route,'class':int(c),'rows':len(ix),'body_keys':int(b.iloc[ix].body_group.nunique()),
                'B_errors':int((~bc[ix]).sum()),'new_errors':int((~correct[ix]).sum()),'outside_target_B_errors':rb,'outside_target_new_errors':rn})
        fold_asa=[int((asa&(b.fold.to_numpy()==f)&~correct).sum()-(asa&(b.fold.to_numpy()==f)&~bc).sum()) for f in range(3)]
        fold_native=[int((target&(b.fold.to_numpy()==f)&~correct).sum()-(target&(b.fold.to_numpy()==f)&~bc).sum()) for f in range(3)]
        gates={'no_critical_previously_correct_regression':not (critical&bc&~correct).any(),
               'no_new_small_slice_total_loss':not any(t['new_total_loss'] for t in tails),
               'M_recall_guard':m['class_recall'][1]>=base['class_recall'][1]-config['recall_drop_limit'],
               'S_recall_guard':m['class_recall'][2]>=base['class_recall'][2]-config['recall_drop_limit'],
               'normal_false_alert_guard':m['false_alerts_per_10000_normal']<=base['false_alerts_per_10000_normal']+config['normal_errors_per_10000_increase_limit']}
        if view=='A':
            gates.update(native_errors_improve=details['Windows_native_target']['metrics']['errors']<result['baseline_slices']['Windows_native_target']['errors'],
                         at_least_two_native_folds_nonworse=sum(v<=0 for v in fold_native)>=2,all_non_target_route_class_error_totals_nonworse=not non_target_regression,
                         ASA_nonworse=details['ASA']['metrics']['errors']<=6018)
        else:
            gates.update(ASA_two_folds_improve=sum(v<0 for v in fold_asa)>=2,ASA_pooled_improve=details['ASA']['metrics']['errors']<6018)
            for k,limit in config['P_asa_unseen_error_limits'].items():gates['ASA_'+k+'_guard']=details['ASA_'+k]['metrics']['errors']<=limit
        gates={k:bool(v) for k,v in gates.items()}
        changes=d.loc[bc!=correct,['row_position','projection_id','body_group','route','label_index']+PROBS].copy()
        for j,n in enumerate(PROBS):changes['B_'+n]=bp[bc!=correct,j]
        changes['change']=np.where(correct[bc!=correct],'fixed','regressed')
        pq.write_table(pa.Table.from_pandas(changes,preserve_index=False),out/('changes_%s.parquet'%view),compression='zstd')
        item={'evaluation':m,'slices':details,'fold_ASA_error_changes':fold_asa,'fold_native_error_changes':fold_native,'gates':gates,
              'eligible_for_stress':view in ('P','A') and all(gates.values()),'small_slices':tails,'route_class':route_class,
              'critical_regressed_rows':b.loc[critical&bc&~correct,'row_position'].tolist(),
              'fixed':int((~bc&correct).sum()),'regressed':int((bc&~correct).sum()),
              'per_fold':[metrics(y[b.fold.to_numpy()==f],p[b.fold.to_numpy()==f]) for f in range(3)],
              'new_ASA_partitions':{n:{'rows':int((asa&(d.new_partition.to_numpy()==k)).sum()),'errors':int((asa&(d.new_partition.to_numpy()==k)&~correct).sum())} for k,n in enumerate(PARTS)}}
        result['candidates'][view]=item;save(out/('candidate_%s.json'%view),item)
        if item['eligible_for_stress']:eligible.append(view)
        print(json.dumps({'view':view,'errors':m['errors'],'macro_f1':m['macro_f1'],'ASA_errors':details['ASA']['metrics']['errors'],'critical_regressions':len(item['critical_regressed_rows']),'gates':gates,'eligible':item['eligible_for_stress']}),flush=True)
        del d,p,changes;gc.collect()
    oldn=pd.concat([load(v40/('primary/fold_%s/N'%f),False) for f in range(3)],ignore_index=True).sort_values('row_position').reset_index(drop=True)
    assert np.array_equal(oldn.row_position,b.row_position)
    npb=oldn[PROBS].to_numpy();pair_outcomes['N']=npb.argmax(1)!=y;pair_losses['N']=-np.log(np.maximum(npb[np.arange(len(y)),y],1e-300))
    factorial={}
    for name,mask in {**slices,'all':np.ones(len(y),dtype=bool)}.items():
        e={v:int(e[mask].sum()) for v,e in pair_outcomes.items() if v in ('B','F','D','N')}
        loss={v:float(z[mask].mean()) for v,z in pair_losses.items() if v in ('B','F','D','N')}
        factorial[name]={'errors':e,'mean_log_loss':loss,'F_minus_B':e['F']-e['B'],'D_minus_B':e['D']-e['B'],
                         'error_interaction_N_minus_F_minus_D_plus_B':e['N']-e['F']-e['D']+e['B']}
    result['factorial']=factorial;result['quality_accepted']=False
    save(out/'primary_review.json',result)
    selected=next((v for v in ['P','A'] if v in eligible),None)
    save(out/'selection.json',{'selected_view':selected,'proceed_to_stress':selected is not None and not a.stage1,'eligible_views':eligible,
        'rule':config['selection'],'configuration_sha256':sha(stage/'configuration.json'),'primary_review_sha256':sha(out/'primary_review.json'),
        'script_sha256':sha(__file__),'stage1_only':a.stage1,'no_stress_results_used_for_selection':True,'quality_accepted':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True);p.add_argument('--stage1',action='store_true');main(p.parse_args())
