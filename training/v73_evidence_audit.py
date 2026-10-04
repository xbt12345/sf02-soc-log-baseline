"""Bounded support/collision and official-record case audit, no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run.resolve()
    out=run/'evidence_audit.json'
    if out.exists():raise FileExistsError(out)
    train=ROOT/'data/official/train.parquet';valid=ROOT/'data/official/valid_input.parquet'
    prepared=ROOT/'artifacts/v39_local_r2_20260913/prepared/rows.parquet'
    fitmanifest=ROOT/'artifacts/v51_fact_residual_20260914/pressure/training_manifest.parquet'
    rows=pd.read_parquet(prepared);rawmeta=pd.read_parquet(train,columns=['event_id','label_binary','src_ip'])
    assert len(rows)==len(rawmeta) and np.array_equal(rows.row_position,np.arange(len(rows)))
    assert np.array_equal(rows.event_id,rawmeta.event_id)
    label_names=['benign','malicious','suspicious']
    np.testing.assert_array_equal(rows.label_index,rawmeta.label_binary.map({n:i for i,n in enumerate(label_names)}))
    positions=set(pd.read_parquet(fitmanifest,columns=['row_position']).row_position)
    rows['actually_fit']=rows.row_position.isin(positions)
    rows['source_symbol']=rawmeta.src_ip.fillna('MISSING_SOURCE')
    support=[]
    for (route,y),z in rows.groupby(['route','label_index']):
        support.append({'route':route,'label':label_names[int(y)],'official_train_rows':len(z),
            'body_groups':int(z.body_group.nunique()),'nonmissing_source_symbols':int(z.loc[z.source_symbol.ne('MISSING_SOURCE'),'source_symbol'].nunique()),
            'actual_fit_rows':int(z.actually_fit.sum()),'actual_fit_bodies':int(z.loc[z.actually_fit,'body_group'].nunique())})
    pred=pd.read_parquet(run/'predictions.parquet')
    answer=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
    d=pred.merge(answer,on='event_id',how='left',validate='one_to_one',sort=False)
    assert d.label_binary.notna().all()
    ct=pd.crosstab(d.effective_input_sha256,d.label_binary).reindex(columns=label_names,fill_value=0)
    minimum=(ct.sum(axis=1)-ct.max(axis=1))
    mixed=(ct>0).sum(axis=1)>1
    consistency=d.groupby('effective_input_sha256').agg(v48=('v48_pred','nunique'),v51=('v51_pred','nunique'))
    assert (consistency.max()<=1).all()
    emptykey=d.loc[d.route.eq('empty'),'effective_input_sha256'].unique();assert len(emptykey)==1
    collapsed=d[d.effective_input_sha256.eq(emptykey[0])]
    collapsed_counts=collapsed.groupby(['route','label_binary']).size().reset_index(name='rows')
    floor={'scope':'Empirical minimum mistakes for deterministic classifiers on this exact frozen model input, on inspected development labels; not a Bayes limit for original raw logs',
        'unique_effective_inputs':len(ct),'mixed_inputs':int(mixed.sum()),'minimum_errors':int(minimum.sum()),
        'same_input_as_empty_message_rows':len(collapsed),'same_input_as_empty_counts':collapsed_counts.to_dict('records')}
    ct.assign(minimum_errors=minimum).to_parquet(run/'effective_input_label_counts.parquet')
    # Three fixed old problem categories. Selecting cases uses labels for audit,
    # never to create training samples, weights, model features or a router.
    facts_path=ROOT/'artifacts/v61_source_factorial_20260914_r2/records.parquet'
    targetpath=ROOT/'artifacts/v67_targeted_20260920/manifest.parquet'
    f=pd.read_parquet(facts_path);m=pd.read_parquet(targetpath)
    target=m[m.target_578].merge(f,on=['row_position','label','group'],suffixes=('_audit',''))
    defs=[('UDP拒绝',target.transport_protocol.eq('udp')),('外部到DMZ的TCP拒绝',target.transport_protocol.eq('tcp')),
          ('无合格邻居',target.empty_context)]
    keys=['action','outcome','transport_protocol','src_role','dst_role','src_port_fixed','dst_port_fixed']
    samples=[];wanted=set()
    for name,mask in defs:
        t=target[mask].sort_values('row_position').iloc[0];fold=int(t['fold'])
        rolepath=ROOT/f'artifacts/v69_support_control_20260921/fold{fold}_roles.parquet'
        rr=pd.read_parquet(rolepath);np.testing.assert_array_equal(rr.row_position,f.row_position)
        fit=f[rr.B_small_sources.to_numpy()]
        behavior=fit.copy()
        for key in keys[:5]:behavior=behavior[behavior[key].eq(t[key])]
        opposite=behavior[behavior.label.eq(1)].copy()
        match=sum(opposite[k].eq(t[k]).astype(int) for k in keys)
        order=opposite.assign(matches=match).sort_values(['matches','row_position'],ascending=[False,True])
        chosen=order.iloc[0]
        case={'category':name,'target_train_row_position':int(t.row_position),'target_fold':fold,'target_official_label':'suspicious',
            'target_neighbor_slots':int(t.neighbor_slots_used),'paired_fit_M_row_position':int(chosen.row_position),
            'same_seven_event_fields_not_full_context':bool(all(chosen[k]==t[k] for k in keys)),
            'differences':{k:{'S':str(t[k]),'M':str(chosen[k])} for k in keys if chosen[k]!=t[k]},
            'same_coarse_behavior_fit_support':[{'label':label_names[int(y)],'rows':len(z),'source_symbols':int(z.group.nunique())} for y,z in behavior.groupby('label')],
            'interpretation':'Nearest observed coarse-behavior counterexample, not a causal pair or proof of label noise. Raw endpoint/context differences may matter.'}
        wanted.update([int(t.row_position),int(chosen.row_position)]);samples.append(case)
    # Also retain one training example for each existing unsupported-M body.
    positive=rows[rows.route.eq('unsupported')&rows.label_index.eq(1)].drop_duplicates('body_group').sort_values('row_position')
    wanted.update(positive.row_position.astype(int));raw={};offset=0
    for batch in pq.ParquetFile(train).iter_batches(batch_size=16384,columns=['message_sanitized'],use_threads=False):
        for pos in sorted(wanted):
            if offset<=pos<offset+len(batch):raw[str(pos)]=batch.column(0)[pos-offset].as_py()
        offset+=len(batch)
    assert len(raw)==len(wanted)
    (run/'official_case_messages.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'new_fits':0,'labels_changed':False,'support_by_route':support,'effective_input_floor':floor,'ASA_case_pairs':samples,
        'unsupported_M_training_representatives':positive[['row_position','body_group','actually_fit']].to_dict('records'),
        'source_caveat':'Literal source symbols and body groups do not prove independent physical incidents.',
        'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [train,valid,prepared,fitmanifest,facts_path,targetpath,run/'predictions.parquet',ROOT/'data/official/valid_answer_private.parquet']},
        'script_sha256':sha(__file__)}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'input_floor':floor,'case_pairs':samples},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
