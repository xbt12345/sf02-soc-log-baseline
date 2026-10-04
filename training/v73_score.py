"""Separate post-inference scoring; answers never reach prediction code."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
LABELS=['benign','malicious','suspicious']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def measure(y,p):
    cm=confusion_matrix(y,p,labels=[0,1,2]);support=cm.sum(axis=1);called=cm.sum(axis=0)
    precision=np.divide(cm.diagonal(),called,out=np.zeros(3,dtype=float),where=called!=0)
    recall=np.divide(cm.diagonal(),support,out=np.zeros(3,dtype=float),where=support!=0)
    f1=np.divide(2*cm.diagonal(),called+support,out=np.zeros(3,dtype=float),where=called+support!=0)
    return dict(rows=int(cm.sum()),correct=int(cm.trace()),errors=int(cm.sum()-cm.trace()),accuracy=float(cm.trace()/cm.sum()),
        confusion_B_M_S=cm.tolist(),class_support=support.tolist(),class_precision=precision.tolist(),
        class_recall=recall.tolist(),class_f1=f1.tolist(),macro_f1_all_three=float(f1.mean()) if (support>0).all() else None,
        normal_to_threat=int(cm[0,1:].sum()),threat_to_normal=int(cm[1:,0].sum()),
        false_alerts_per_10000_normal=float(cm[0,1:].sum()/support[0]*10000) if support[0] else None,
        malicious_to_suspicious=int(cm[1,2]),suspicious_to_malicious=int(cm[2,1]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run.resolve()
    if (run/'scoring.json').exists():raise FileExistsError('Do not overwrite scores')
    done=json.loads((run/'inference_complete.json').read_text(encoding='utf-8'))
    for name,h in done['output_sha256'].items():assert sha(run/name)==h
    ans_path=ROOT/'data/official/valid_answer_private.parquet'
    pred=pd.read_parquet(run/'predictions.parquet');answer=pd.read_parquet(ans_path)
    assert len(answer)==len(pred) and answer.event_id.is_unique and pred.event_id.is_unique
    d=pred.merge(answer,on='event_id',how='left',validate='one_to_one',sort=False)
    assert d.label_binary.notna().all() and set(d.label_binary.unique())==set(LABELS)
    y=d.label_binary.map({n:i for i,n in enumerate(LABELS)}).to_numpy()
    assert np.array_equal(d.event_id,pred.event_id)
    source=ROOT/'data/official/valid_input.parquet'
    meta=pd.read_parquet(source,columns=['event_id','src_ip','product_name'])
    assert np.array_equal(meta.event_id,d.event_id)
    names=meta.src_ip.fillna('').astype(str)
    # Scoring-only identities: never imported or passed into inference.
    d['source_group']=np.where(names.str.strip().eq(''), 'missing_source',names)
    train_sources=set(pd.read_parquet(ROOT/'data/official/train.parquet',columns=['src_ip']).src_ip.dropna().astype(str))
    d['unseen_nonempty_source']=names.ne('')&~names.isin(train_sources)
    outputs={'scope':'Historical official development regression using already inspected private answers; not new blind or external validation',
        'new_fits':0,'primary_metric':'exact three-class row accuracy; macro-F1 and per-class recalls/f1 are mandatory safeguards',
        'answers_read_only_by_scorer':True,'model_quality_promoted':False,'full':{},'slices':[],
        'inputs_sha256':{'predictions':sha(run/'predictions.parquet'),'answers':sha(ans_path),'input':sha(source)},
        'source_sha256':sha(__file__)}
    models={'all_benign':np.zeros(len(y),dtype=int),'v48':d.v48_pred.to_numpy(),'v51':d.v51_pred.to_numpy()}
    for name,p in models.items():outputs['full'][name]=measure(y,p)
    slices=[('all',np.ones(len(d),bool)),('missing_product',d.missing_product.to_numpy()),
        ('empty_message',d.empty_message.to_numpy()),('unseen_nonempty_src_ip',d.unseen_nonempty_source.to_numpy()),
        ('text_encoded_zero',d.text_encoded_zero.to_numpy()),('residual_applied',d.residual_applied.to_numpy())]
    slices += [('route:'+str(n),d.route.eq(n).to_numpy()) for n in sorted(d.route.unique())]
    slices += [('protocol:'+str(n),d.protocol.eq(n).to_numpy()) for n in ['tcp','udp','icmp']]
    hard=d.action.eq('deny')&d.src_role.eq('outside')&d.dst_role.eq('dmz')&d.protocol.isin(['tcp','udp'])
    slices += [('ASA_outside_dmz_deny',hard.to_numpy())]
    for label,k in slices:
        if not k.any():outputs['slices'].append({'slice':label,'rows':0});continue
        out={'slice':label,'rows':int(k.sum()),'models':{n:measure(y[k],p[k]) for n,p in models.items() if n!='all_benign'}}
        outputs['slices'].append(out)
    b=models['v48'];p=models['v51'];fixed=(b!=y)&(p==y);broken=(b==y)&(p!=y)
    outputs['paired']={'fixed':int(fixed.sum()),'broken':int(broken.sum()),'changed':int((p!=b).sum()),
        'by_class':{n:{'fixed':int((fixed&(y==i)).sum()),'broken':int((broken&(y==i)).sum())} for i,n in enumerate(LABELS)}}
    source_rows=[]
    for name in ['v48','v51']:
        z=pd.DataFrame({'source':d.source_group,'y':y,'ok':models[name]==y,'route':d.route})
        sr=z.groupby(['source','y']).ok.agg(['size','mean']).reset_index()
        outputs['full'][name]['class_source_mean_recall']={LABELS[int(i)]:float(g['mean'].mean()) for i,g in sr.groupby('y')}
        sr['model']=name;source_rows.append(sr)
    pd.concat(source_rows).to_parquet(run/'source_class_metrics.parquet',index=False)
    errors=d.loc[p!=y,['row_position','event_id','route','protocol','action','src_role','dst_role','missing_product','empty_message','source_group','v48_pred','v51_pred','label_binary']]
    errors.to_parquet(run/'errors.parquet',index=False)
    outputs['outputs_sha256']={n:sha(run/n) for n in ['errors.parquet','source_class_metrics.parquet']}
    (run/'scoring.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'full':outputs['full'],'paired':outputs['paired']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
