"""Independent reload/scikit-learn count verification and component uncertainty."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from threadpoolctl import threadpool_limits
from v79_execute import DEST,WIDTH,rows,load,weights,FOLDS,SEED
from run_v75 import ROOT,read,save,sha,NAMES


def uncertainty(r,fid):
    b=np.load(DEST/'P1_B_ovr_feature_prediction.npy');p=np.load(DEST/'P2_B_ms_feature_prediction.npy');results={}
    rng=np.random.default_rng(7911)
    for role,mask in [('calibration',r.fold.eq(2).to_numpy()),('evaluation',r.fold.eq(0).to_numpy())]:
        component,ids=pd.factorize(r.loc[mask,'component'],sort=False);y=r.label_index.to_numpy()[mask];bp=b[fid[mask]];pp=p[fid[mask]]
        vals=np.column_stack([(y==1),(y==1)&(bp==y),(y==1)&(pp==y),(y==2),(y==2)&(bp==y),(y==2)&(pp==y),bp!=y,pp!=y]).astype(float)
        units=np.zeros((len(ids),8));np.add.at(units,component,vals)
        intervals=[]
        for _ in range(400):
            s=units[rng.integers(0,len(ids),len(ids))].sum(0)
            intervals.append([(s[2]-s[1])/s[0] if s[0] else np.nan,(s[5]-s[4])/s[3] if s[3] else np.nan,s[7]-s[6]])
        a=np.array(intervals)
        results[role]={'proxy_components':len(ids),'bootstrap_draws':400,'MS_aux_minus_OVR_95pct':
            {'M_recall':np.nanquantile(a[:,0],[.025,.975]).tolist(),'S_recall':np.nanquantile(a[:,1],[.025,.975]).tolist(),
             'row_errors':np.nanquantile(a[:,2],[.025,.975]).tolist()},'scope':'Component resampling, previously inspected development; components are isolation proxies, not verified independent organizations.'}
    save(DEST/'component_uncertainty.json',results)


def main():
    r,x,fid=load();checks={};fits=[];prediction_rows=0
    with threadpool_limits(limits=4):
        for path in sorted(DEST.glob('*_fit.json')):
            report=read(path);name=report['name'];modelpath=DEST/(name+'.joblib');model=joblib.load(modelpath)
            assert sha(modelpath)==report['model_sha256'];assert model['coef'].shape==(WIDTH,3)
            assert np.isfinite(model['coef']).all() and np.isfinite(model['intercept']).all()
            pred=(x@model['coef']+model['intercept']).argmax(1)
            saved=np.load(DEST/(name+'_feature_prediction.npy'));assert np.array_equal(saved,pred)
            h,c=(1,3) if name=='P4_source_1' else ((4,1) if name=='P4_source_4' else (0,2))
            scores=read(DEST/(name+'_scores.json'))
            for role,mask in [('fit',~r.fold.isin([h,c]).to_numpy()),('calibration',r.fold.eq(c).to_numpy()),('evaluation',r.fold.eq(h).to_numpy())]:
                y=r.label_index.to_numpy()[mask];pp=pred[fid[mask]]
                cm=confusion_matrix(y,pp,labels=[0,1,2]);assert cm.tolist()==scores[role]['cm']
                prec,rec,f1,support=precision_recall_fscore_support(y,pp,labels=[0,1,2],zero_division=0)
                for key,val in [('precision',prec),('recall',rec),('f1',f1)]:
                    for k,expected in enumerate(scores[role][key]):
                        if expected is not None:assert abs(val[k]-expected)<1e-12,(name,role,key)
                prediction_rows+=len(y)
            # Recompute each actual train draw and ensure withheld components/format supervision never entered it.
            fit=~r.fold.isin([h,c]).to_numpy()
            if name=='P5_shadow':fit=~r.fold.eq(0).to_numpy()
            if name.startswith('P4_') and ('_zero_M' in name or '_whole_format' in name):
                suffix='_zero_M' if '_zero_M' in name else '_whole_format';carrier=name[3:-len(suffix)]
                remove=r.route.eq(carrier).to_numpy()
                if suffix=='_zero_M':remove=remove&r.label_index.eq(1).to_numpy()
                fit=fit&~remove
            arm=report['exposure']['arm'];seed=7902 if name=='P4_second_seed' else SEED
            _,den,e=weights(r,fid,x.shape[0],fit,arm,seed)
            assert den==report['exposure']['denominator'];assert e['selected_rows_sha256']==report['exposure']['selected_rows_sha256']
            fits.append({'name':name,'converged':report['converged'],'seconds':report['seconds'],
                         'valid_coverage_experiment':name!='P4_cef_zero_M'})
            print(json.dumps({'verified_model':name}),flush=True)
    uncertainty(r,fid)
    checks['all_model_hashes_finite_and_reload_predictions']=True
    checks['all_fold_score_tables_match_sklearn']=True
    checks['all_fit_population_and_sampling_receipts_match']=True
    checks['registered_source_unchanged']=read(DEST/'registration.json')['source_sha256']==sha(ROOT/'training/v79_execute.py')
    assert checks['registered_source_unchanged']
    receipt=read(DEST/'development_inference_receipt.json');assert receipt['prediction_sha256']==sha(DEST/'development_predictions.parquet')
    pred=pd.read_parquet(DEST/'development_predictions.parquet');answers=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
    d=pred.merge(answers,on='event_id',validate='one_to_one',sort=False);y=d.label_binary.map(dict(zip(NAMES,range(3)))).to_numpy()
    cm=confusion_matrix(y,d.prediction,labels=[0,1,2]);reg=read(DEST/'development_regression.json');assert cm.tolist()==reg['metrics']['cm']
    checks['complete_2014052_development_cm_matches_sklearn']=len(d)==2014052
    checks['development_inference_without_answers']=receipt['answers_read'] is False
    save(DEST/'independent_verification.json',{'all_checks_passed':all(checks.values()),'checks':checks,'model_fits':fits,
        'actual_classifier_fits':len(fits),'valid_protocol_classifier_fits':sum(v['valid_coverage_experiment'] for v in fits),
        'training_role_prediction_rows_recomputed':prediction_rows,'development_rows':len(d),'development_cm':cm.tolist(),
        'new_calibration_fits':0,'quality_acceptance':False,'source_sha256':sha(__file__),
        'invalid_coverage_note':'P4_cef_zero_M used nonexistent cef route and withheld0 rows. Preserved as one actual no-op fit, excluded as carrier evidence; corrected cef_fields zero/whole fits separately executed.'})


if __name__=='__main__':main()
