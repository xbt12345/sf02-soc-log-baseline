"""Fit-only source-excluded evidence audit and one frozen context-rule probe."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v61_common import FIELDS, read, save, sha
from train_v63_factorial import metric_summary
from v64_selection import select

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/v61_source_factorial_20260914_r2'
PREV=ROOT/'artifacts/v63_factorial_20260920'


def keys(frame, columns):
    return frame[columns].astype(str).apply(lambda row:json.dumps(row.tolist(),separators=(',',':')),axis=1)


def support_table(frame, column):
    # One vote per key/source/class; copies of a log cannot manufacture support.
    return (frame.drop_duplicates([column,'group','label'])
            .groupby([column,'label']).size().unstack(fill_value=0)
            .reindex(columns=[1,2],fill_value=0))


def pure_rule(frame, column, min_sources=3):
    table=support_table(frame,column)
    result={}
    for label in [1,2]:
        good=(table[label]>=min_sources)&(table[3-label]==0)
        result.update({key:label for key in table.index[good]})
    return result


def leave_source_out(frame, column, min_sources=3):
    table=support_table(frame,column)
    membership=frame.drop_duplicates([column,'group','label'])
    own={label:set(zip(membership.loc[membership.label.eq(label),column],
                       membership.loc[membership.label.eq(label),'group'])) for label in [1,2]}
    total={label:frame[column].map(table[label]).to_numpy() for label in [1,2]}
    for label in [1,2]:
        total[label]-=np.array([(k,g) in own[label] for k,g in zip(frame[column],frame.group)],dtype=int)
    prediction=np.full(len(frame),-1,dtype=int)
    for label in [1,2]:prediction[(total[label]>=min_sources)&(total[3-label]==0)]=label
    return prediction


def context_shapes(stats):
    # Structural presence only: no literal identity, timestamp, count magnitude,
    # numeric port value, label, or learned representation.
    names=['has_other_state','has_same_destination','has_same_destination_port',
           'has_tcp','has_udp','has_icmp','has_missing_src_port','has_missing_dst_port',
           'has_outside_destination','has_dmz_destination','has_inside_destination',
           'multiple_protocols','multiple_destination_roles']
    mask=np.column_stack([stats[:,0]>0,stats[:,4:14]>0,stats[:,14]>1,stats[:,15]>1])
    assert mask.shape[1]==len(names)
    return pd.DataFrame(mask.astype(int),columns=names)


def coverage_metrics(frame, prediction):
    result={}
    for label in [1,2]:
        part=frame[frame.label.eq(label)].copy();p=prediction[frame.label.eq(label)]
        part['covered']=p>=0;part['correct']=p==label
        result[str(label)]={'rows':len(part),'sources':part.group.nunique(),
             'covered_rows':int((p>=0).sum()),'wrong_covered_rows':int(((p>=0)&(p!=label)).sum()),
             'source_mean_coverage':float(part.groupby('group').covered.mean().mean()),
             'source_mean_correct_coverage':float(part.groupby('group').correct.mean().mean())}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Do not overwrite evidence')
    config=read(DATA/'configuration.json')
    for name in ['records.parquet','context.npz']:
        assert sha(DATA/name)==config['data_bindings'][name]
    frame=pd.read_parquet(DATA/'records.parquet',filters=[('role','in',['fit','selection'])]).sort_values('row_position').reset_index(drop=True)
    assert frame.groupby('group').role.nunique().max()==1
    assert frame.groupby('body_group').role.nunique().max()==1
    frame['behavior']=keys(frame,FIELDS[:5])
    # Global row positions are read for array indexing only, not outer labels/text.
    positions=pd.read_parquet(DATA/'records.parquet',columns=['row_position']).row_position
    ix=pd.Series(np.arange(len(positions)),index=positions).loc[frame.row_position].to_numpy()
    global_to_local=np.full(len(positions),-1,dtype=int);global_to_local[ix]=np.arange(len(frame))
    with np.load(DATA/'context.npz') as context:
        stats=context['stats'][ix];neighbors=context['neighbors'][ix]
    valid=neighbors>=0;mapped=global_to_local[neighbors.clip(min=0)]
    assert ((mapped>=0)|~valid).all()
    assert ((frame.group.to_numpy()[mapped.clip(min=0)]==frame.group.to_numpy()[:,None])|~valid).all()
    assert ((frame.role.to_numpy()[mapped.clip(min=0)]==frame.role.to_numpy()[:,None])|~valid).all()
    shape=context_shapes(stats);frame['shape']=keys(shape,shape.columns.tolist())
    views={'coarse':FIELDS[:5], 'source_port':FIELDS[:5]+['src_port_fixed'],
           'destination_port':FIELDS[:5]+['dst_port_fixed'],
           'port_pair':FIELDS[:5]+['src_port_fixed','dst_port_fixed'],
           'port_ranges':FIELDS[:5]+['src_port_range','dst_port_range'],
           'text_facts':['text']+FIELDS,'context_shape':['behavior','shape']}
    for name,columns in views.items():frame[name]=keys(frame,columns)
    fit=frame[frame.role.eq('fit')&frame.label.gt(0)].copy().reset_index(drop=True)
    dev=frame[frame.role.eq('selection')].copy().reset_index(drop=True)
    args.out.mkdir(parents=True)
    # Fixed before evaluation of the new probe; NOT before historical data inspection.
    registration={'status':'development_probe_fixed_before_execution_not_blind_preregistration',
      'method':'Fit-only pure class rules supported by >=3 independent sources; context_shape rules override fit coarse row-majority, no probability threshold search.',
      'min_sources':3,'views':views,'shape_features':shape.columns.tolist(),
      'unknown_or_conflicting_rule':'Fallback to fit coarse majority; no row removed from evaluation.',
      'source_sha256':sha(__file__),'selection_source_sha256':sha(ROOT/'training/v64_selection.py'),
      'inputs':{str((DATA/n).relative_to(ROOT)):sha(DATA/n) for n in ['records.parquet','context.npz']},
      'scope':'ASA targeted diagnosis; source equality proxy is not a verified real host or domain; all selection previously inspected.'}
    save(args.out/'protocol.json',registration)
    audits={};rows=[];view_support=[]
    for name in views:
        pred=leave_source_out(fit,name)
        audits[name]=coverage_metrics(fit,pred)
        for behavior,part in fit.groupby('behavior'):
            loc=part.index.to_numpy();m=coverage_metrics(part,pred[loc])
            for label,detail in m.items():
                rows.append({'view':name,'behavior':behavior,'label':int(label),**detail})
            table=support_table(part,name)
            view_support.append({'view':name,'behavior':behavior,'keys':len(table),
                'mixed_keys':int(table.gt(0).all(axis=1).sum()),
                'S_keys_ge3_sources':int((table[2]>=3).sum()),
                'pure_S_keys_ge3_sources':int(((table[2]>=3)&(table[1]==0)).sum())})
        table=support_table(fit,name)
        table.rename(columns={1:'M_sources',2:'S_sources'}).reset_index().to_parquet(args.out/(name+'_support.parquet'),index=False)
    pd.DataFrame(rows).to_csv(args.out/'source_excluded_coverage.csv',index=False)
    pd.DataFrame(view_support).to_csv(args.out/'view_support.csv',index=False)
    skeleton=fit.text.str.replace(r'/(?:opaque_port|[0-9]+)', '/[port]',regex=True)
    grammar=fit.assign(skeleton=skeleton).groupby('behavior').agg(
        rows=('label','size'),text_variants=('text','nunique'),port_masked_text_variants=('skeleton','nunique')).reset_index()
    grammar.to_csv(args.out/'text_variation.csv',index=False)
    # A transparent contextual alternative, one configuration, no tuning on dev labels.
    full_fit=frame[frame.role.eq('fit')]
    majority=full_fit.groupby(['coarse','label']).size().unstack(fill_value=0).idxmax(axis=1)
    default=int(fit.label.mode().iloc[0]);base=dev.coarse.map(majority).fillna(default).to_numpy(dtype=int)
    rule=pure_rule(fit,'context_shape');override=dev.context_shape.map(rule)
    pred=np.where(override.notna(),override.fillna(default),base).astype(int)
    prediction=dev[['row_position','role','label','group','behavior','context_shape']].copy()
    prediction['baseline_prediction']=base;prediction['prediction']=pred;prediction['rule_applied']=override.notna()
    prediction.to_parquet(args.out/'context_probe.parquet',index=False)
    base_metrics=metric_summary(dev,np.eye(3)[base]);probe_metrics=metric_summary(dev,np.eye(3)[pred])
    refs={'A0_row_peak':read(PREV/'additional_baseline_guard.json')['row_peak_A0']['metrics'],
          'fit_coarse_majority':base_metrics}
    replay={arm:select(read(PREV/arm/'curve.json'),refs) for arm in ['A0','A1','P0','P1']}
    probe_gate=select([{'step':0,'metrics':probe_metrics}],refs)
    by_behavior=[]
    for (behavior,label),part in prediction[prediction.label.gt(0)].groupby(['behavior','label']):
        before=part.baseline_prediction.eq(label);after=part.prediction.eq(label)
        by_behavior.append({'behavior':behavior,'label':int(label),'rows':len(part),'sources':part.group.nunique(),
              'fixed':int((~before&after).sum()),'broken':int((before&~after).sum()),
              'base_source_recall':float(before.groupby(part.group).mean().mean()),
              'source_recall':float(after.groupby(part.group).mean().mean())})
    pd.DataFrame(by_behavior).to_csv(args.out/'context_probe_effects.csv',index=False)
    save(args.out/'selection_replay.json',{'scope':'Retrospective comparison only; historical selections and training are unchanged. New policy must be frozen before a future training run.','references':refs,'arms':replay})
    summary={'status':'evidence_audit_selection_repair_and_context_probe_completed','quality_acceptance':False,
      'no_neural_retraining':True,'fit_rows':len(fit),'selection_rows':len(dev),
      'source_excluded_support':audits,'context_probe':{'rule_count':len(rule),'baseline':base_metrics,'candidate':probe_metrics,'selection_gate':probe_gate},
      'context_isolation_checked':True,'old_outer_labels_or_text_used':False,
      'limitations':['Exact-key support is a conservative diagnostic, not a proof that nonlinear or semantic generalization is impossible.',
        'Leave-one-source-out rule evidence remains development evidence, not external validation.',
        'Pure rules with three sources have no guaranteed future precision; duplicates within a source cannot increase support.',
        'Context uses role-local source-symbol cooccurrence, not qualified sessions or event rates.',
        'Fixed context shape ignores magnitudes and may discard relevant information; it is a probe, not a replacement representation.',
        'No full SOC or operational false-positive validation; no model promotion.'],
      'protocol_sha256':sha(args.out/'protocol.json')}
    summary['outputs']={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}
    save(args.out/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'context_probe_passed':probe_gate['screen_passed'],
        'probe_S_source':probe_metrics['S_source_recall'],'baseline_S_source':base_metrics['S_source_recall'],
        'reselected':{k:None if x['selected'] is None else x['selected']['step'] for k,x in replay.items()}},indent=2),flush=True)


if __name__=='__main__':main()
