"""Execute frozen carrier protocols once where two masks are exactly equal."""
import numpy as np
import joblib
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from v79_execute import *


def main():
    r,x,fid=load();selected,arm,kind=selected_spec()
    # Coverage/efficiency amendment precedes this stage's first fit and sees no model metrics.
    support={}
    for carrier in FORMATS+['cef_fields']:
        support[carrier]=r[r.route.eq(carrier)].groupby(['fold','label_index']).size().rename('rows').reset_index().to_dict('records')
    binding={'source_sha256':sha(__file__),'primary_source_sha256':sha(ROOT/'training/v79_execute.py'),
         'new_P4_fits_before_binding':0,'support':support,'cef':'Supplementary carrier in accepted plan, never sole proof of VPC transfer',
         'equal_masks':'One fitted model reused for identical zero-M and whole-format train masks; distinct generic-input test remains separate',
         'selection_frozen':selected,'no_new_candidate_selection':True}
    if not (DEST/'P4_execution_binding.json').exists():save(DEST/'P4_execution_binding.json',binding)
    else:save(DEST/'P4_mask_compatibility_repair.json',{'old_source':sha(DEST/'frozen_development/transfer_before_mask_fix.py'),
          'new_source':sha(__file__),'change':'Copy read-only pandas bool array before in-place conjunction; resume already fitted source models without refitting',
          'selection_changed':False,'objectives_changed':False})
    result={'selected':selected,'source_folds':[],'formats':{},'seed_confirmation':None}
    with threadpool_limits(limits=4):
        for index,(h,c) in enumerate(FOLDS):
            name=selected if index==0 else 'P4_source_'+str(h)
            if index:
                if (DEST/(name+'_scores.json')).exists():scores=read(DEST/(name+'_scores.json'))
                else:
                    fit=~r.fold.isin([h,c]).to_numpy(); cc,den,e=weights(r,fid,x.shape[0],fit,arm);e['arm']=arm
                    model=solve(x,cc,den,kind,name,e);scores=evaluate(r,x,fid,model,h,c,name)
            else:scores=read(DEST/(name+'_scores.json'))
            result['source_folds'].append({'H':h,'C':c,'name':name,'metrics':scores['evaluation']})
            save(DEST/'P4_transfer_progress.json',result)
        h,c=FOLDS[0];fit=~r.fold.isin([h,c]).to_numpy()
        for carrier in FORMATS+['cef_fields']:
            info={};cache={}
            for protocol in ['zero_M','whole_format']:
                remove=r.route.eq(carrier).to_numpy().copy()
                if protocol=='zero_M':remove &= r.label_index.eq(1).to_numpy()
                restricted=fit&~remove;mask_hash=__import__('hashlib').sha256(restricted.tobytes()).hexdigest()
                if mask_hash in cache:
                    name,model,pp,e=cache[mask_hash];shared=True
                else:
                    cc,den,e=weights(r,fid,x.shape[0],restricted,arm);e.update({'arm':arm,'removed_supervision_rows':int((fit&remove).sum())})
                    name='P4_'+carrier+'_'+protocol
                    if (DEST/(name+'_joblib')).exists():raise AssertionError('invalid checkpoint filename')
                    if (DEST/(name+'_scores.json')).exists():
                        model=joblib.load(DEST/(name+'.joblib'))
                        assert read(DEST/(name+'_fit.json'))['exposure']['selected_rows_sha256']==e['selected_rows_sha256']
                    else:
                        model=solve(x,cc,den,kind,name,e);evaluate(r,x,fid,model,h,c,name)
                    pp=np.load(DEST/(name+'_feature_prediction.npy'));cache[mask_hash]=(name,model,pp,e);shared=False
                hm=r.fold.eq(h).to_numpy()&r.route.eq(carrier).to_numpy();cm=r.fold.eq(c).to_numpy()&r.route.eq(carrier).to_numpy()
                entry={'model':name,'shared_identical_training_mask':shared,'removed_fit_rows':int((fit&remove).sum()),
                       'H_metrics':score(r,fid,pp,hm),'C_metrics':score(r,fid,pp,cm),'H_components':int(r.loc[hm,'component'].nunique()),
                       'missing_classes_not_evidence':True,'known_schema':True}
                if protocol=='whole_format':
                    enc=joblib.load(OUT/'facts_encoder.joblib');default=normalize(enc.transform([{}]).astype(np.float32),copy=False)
                    ids=np.unique(fid[hm|cm]);gp=pp.copy()
                    for beg in range(0,len(ids),2048):
                        ix=ids[beg:beg+2048];xx=x[ix];meta=xx[:,-18:].tolil();meta[:,17]=0
                        generic=sparse.hstack([xx[:,:65792],sparse.vstack([default]*len(ix),format='csr'),meta.tocsr()],format='csr')
                        gp[ix]=(generic@model['coef']+model['intercept']).argmax(1)
                    entry['unknown_syntax_no_adapter']={'H_metrics':score(r,fid,gp,hm),'C_metrics':score(r,fid,gp,cm),
                        'scope':'Heldout carrier facts disabled; universal residual + record port only. No novel attack or arbitrary-format guarantee.'}
                info[protocol]=entry;emit(stage='carrier_finished',carrier=carrier,protocol=protocol,H=entry['H_metrics'])
            result['formats'][carrier]=info;save(DEST/'P4_transfer_progress.json',result)
        cc,den,e=weights(r,fid,x.shape[0],fit,arm,seed=7902);e.update({'arm':arm,'seed':7902})
        name='P4_second_seed'
        if (DEST/(name+'_scores.json')).exists():scores=read(DEST/(name+'_scores.json'))
        else:
            model=solve(x,cc,den,kind,name,e);scores=evaluate(r,x,fid,model,h,c,name)
        result['seed_confirmation']=scores['evaluation']
    save(DEST/'P4_transfer.json',result)


if __name__=='__main__':main()
