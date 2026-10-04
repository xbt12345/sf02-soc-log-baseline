"""Independent full-output and source integrity checks, never fits a model."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT,OUT,read,save,sha,NAMES


def main():
    cfg=read(OUT/'configuration.json')
    for n,h in cfg['input_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in cfg['source_sha256'].items():assert sha(ROOT/'training'/n)==h,n
    for stage in ['four_arm','full']:
        rec=read(OUT/stage/'complete.json')
        for n,h in rec['outputs'].items():assert sha(OUT/stage/n)==h,n
    full=read(OUT/'full/complete.json')
    assert full['total_exposures']['FULL']==cfg['rows']*cfg['epochs']
    support=full['fit_support']
    native=[s for s in support if s['route']=='native_flow' and s['label']=='malicious']
    assert len(native)==1 and native[0]['fit_rows']==32596
    assert sum(s['fit_rows'] for s in support)==2056871
    dest=OUT/'official_replay';rec=read(dest/'complete.json')
    for n,h in rec['output_sha256'].items():assert sha(dest/n)==h,n
    p=pd.read_parquet(dest/'predictions.parquet');csv=pd.read_csv(dest/'res_development.csv')
    original=pd.read_parquet(ROOT/'data/official/valid_input.parquet',columns=['event_id'])
    np.testing.assert_array_equal(original.event_id,p.event_id)
    np.testing.assert_array_equal(csv.event_id,p.event_id)
    np.testing.assert_array_equal(csv.pred_label.to_numpy(),np.asarray(NAMES)[p.pred])
    assert len(p)==2014052 and p.event_id.is_unique
    q=p[['p0','p1','p2']].to_numpy();assert np.isfinite(q).all()
    np.testing.assert_allclose(q.sum(1),1,atol=1e-6)
    scores=read(dest/'scoring.json')
    ans=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet').set_index('event_id')
    y=ans.loc[p.event_id,'label_binary'].map(dict(zip(NAMES,range(3)))).to_numpy()
    cm=np.bincount(y*3+p.pred.to_numpy(),minlength=9).reshape(3,3)
    assert cm.tolist()==scores['after']['cm']
    error=int((y!=p.pred.to_numpy()).sum());assert error==scores['after']['errors']
    result={'all_checks_passed':True,'official_training_rows':2056871,'native_flow_malicious_all_used':32596,
        'duplicates_removed':0,'full_fit_epochs':cfg['epochs'],'full_fit_original_exposures':full['total_exposures']['FULL'],
        'whole_input_ids_order_csv_and_probabilities_match':True,'rows_replayed':len(p),'independently_recomputed_cm':cm.tolist(),
        'original_data_hashes_rechecked':True,'frozen_sources_and_model_outputs_rechecked':True,
        'quality_acceptance':False,'scope':'Execution integrity and inspected-answer development scoring, not blind transfer acceptance.',
        'source_sha256':sha(__file__)}
    save(OUT/'release_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
