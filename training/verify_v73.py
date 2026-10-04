"""Independent full CSV/score checks plus canonical raw-record model replay."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from v73_inference import ROOT,load_bound,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run.resolve()
    done=json.loads((run/'inference_complete.json').read_text(encoding='utf-8'))
    reg=json.loads((run/'inference_contract.json').read_text(encoding='utf-8'))
    score=json.loads((run/'scoring.json').read_text(encoding='utf-8'))
    for name,h in done['output_sha256'].items():assert sha(run/name)==h,name
    for p,h in reg['source_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in reg['model_and_runtime_sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/'training/v73_score.py')==score['source_sha256']
    evidence=json.loads((run/'evidence_audit.json').read_text(encoding='utf-8'))
    assert evidence['script_sha256']==sha(ROOT/'training/v73_evidence_audit.py')
    for p,h in evidence['input_sha256'].items():assert sha(ROOT/p)==h,p
    p=pd.read_parquet(run/'predictions.parquet');answer=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
    x=pd.read_parquet(ROOT/'data/official/valid_input.parquet',columns=['event_id'])
    assert len(p)==len(x)==2014052 and p.event_id.is_unique and answer.event_id.is_unique
    np.testing.assert_array_equal(p.event_id,x.event_id)
    labels=['benign','malicious','suspicious'];y=answer.set_index('event_id').loc[p.event_id,'label_binary'].map({v:i for i,v in enumerate(labels)}).to_numpy()
    matrices={}
    for name in ['v48','v51']:
        csv=pd.read_csv(run/f'res_{name}_development.csv',dtype=str)
        assert list(csv)==['event_id','pred_label'] and len(csv)==len(x) and csv.event_id.is_unique
        np.testing.assert_array_equal(csv.event_id.to_numpy(),x.event_id.astype(str).to_numpy())
        pred=p[name+'_pred'].to_numpy();np.testing.assert_array_equal(csv.pred_label,np.asarray(labels)[pred])
        cm=np.zeros((3,3),dtype=np.int64);np.add.at(cm,(y,pred),1)
        assert cm.tolist()==score['full'][name]['confusion_B_M_S']
        assert int((pred!=y).sum())==score['full'][name]['errors'];matrices[name]=cm.tolist()
    # Replay real original entry on deterministic per-route and edge-case rows.
    samples=set(p.groupby('route').head(4).row_position.tolist())
    samples.update(p[p.missing_product].row_position.head(4));samples.update([0,len(p)-1])
    raw={};offset=0
    for batch in pq.ParquetFile(ROOT/'data/official/valid_input.parquet').iter_batches(batch_size=16384,use_threads=False):
        take=[i for i in samples if offset<=i<offset+len(batch)]
        if take:
            d=batch.to_pandas()
            for i in take:raw[i]=d.iloc[i-offset].to_dict()
        offset+=len(batch)
    bundle,adapter,residual,_=load_bound();ix=sorted(raw);records=[raw[i] for i in ix]
    qb,_=adapter.classify_records(bundle['base'],records);qc,_=residual.classify_records(bundle,records)
    for name,q in [('v48',qb),('v51',qc)]:
        np.testing.assert_allclose(q,p.iloc[ix][[name+'_p_'+c for c in ['B','M','S']]].to_numpy(),rtol=0,atol=1e-12)
    # A label-bearing file must be refused before a run directory is created.
    with tempfile.TemporaryDirectory(prefix='v73_contract_',dir=run) as td:
        td=Path(td);test=pd.DataFrame([records[0]]);test['label_binary']='malicious';test.to_parquet(td/'labeled.parquet',index=False)
        z=subprocess.run([sys.executable,str(ROOT/'training/v73_inference.py'),'--input',str(td/'labeled.parquet'),'--out',str(td/'forbidden_run')],capture_output=True,text=True)
        assert z.returncode!=0 and 'twelve official unlabeled input columns' in z.stderr and not (td/'forbidden_run').exists()
    old=ROOT/'artifacts/v73_full_task_20260921';same_old=None
    if old.exists():
        same_old=all(sha(old/f'res_{m}_development.csv')==sha(run/f'res_{m}_development.csv') for m in ['v48','v51'])
        assert same_old
    floor=pd.read_parquet(run/'effective_input_label_counts.parquet')
    bound=int((floor[labels].sum(axis=1)-floor[labels].max(axis=1)).sum())
    assert bound==evidence['effective_input_floor']['minimum_errors']==6332
    result={'all_checks_passed':True,'new_fits':0,'rows':len(p),'canonical_raw_replay_rows':len(ix),
        'csv_ids_and_allowed_labels_exact':True,'labels_rejected_at_inference':True,'scores_recomputed_from_predictions':True,
        'corrected_audit_metadata_predictions_unchanged':same_old,'actual_input_empirical_error_floor':bound,
        'source_bindings_unchanged':len(reg['model_and_runtime_sha256']),'confusion_B_M_S':matrices,
        'quality_acceptance':False,'scope':'Full-input historical development regression and inference integrity; not model improvement or blind/generalization acceptance'}
    (run/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
