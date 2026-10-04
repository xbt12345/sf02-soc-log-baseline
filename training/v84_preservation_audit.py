"""No-fit audit of backward-compatible predictions and teacher-lock limitations."""
import json
import joblib
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, read, save, sha, load_sparse
from v79_execute import rows
from v82_capacity import DEST as OLD, LAST

DEST=ROOT/'artifacts/v84_preservation_20260927'


def quantile(a):
    return np.quantile(a,[0,.25,.5,.75,1]).tolist() if len(a) else []


def main():
    if (DEST/'review_receipt.json').exists():raise FileExistsError('Published review is frozen')
    DEST.mkdir(exist_ok=True)
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X')
    k=np.load(OLD/'primary_kernel.npy',mmap_mode='r');models={};scores={}
    for arm in ['linear','nonlinear']:
        state=read(OLD/('primary_'+arm+'_fit.json'))['states'][-1]['name']
        models[arm]=m=joblib.load(OLD/(state+'.joblib'))
        z=np.asarray(x@m['coef'])+m['intercept']
        if len(m['kernel_coef']):z+=k@m['kernel_coef']
        assert np.array_equal(z.argmax(1),np.load(OLD/(state+'_prediction.npy')))
        scores[arm]=z
    za,zb=scores['linear'],scores['nonlinear'];pa,pb=za.argmax(1)[fid],zb.argmax(1)[fid]
    selected=np.zeros(len(r),bool);selected[np.load(OLD/'selected_rows.npy')]=True
    n=x.shape[0];cc=np.bincount(fid[selected]*3+y[selected],minlength=n*3).reshape(-1,3)
    oldpred=za.argmax(1);oldcorrect_count=cc[np.arange(n),oldpred];protected=oldcorrect_count>0
    oldwrong_count=cc.sum(1)-oldcorrect_count
    protected_conflict=(oldcorrect_count>0)&(oldwrong_count>0)
    lock_floor=int(oldwrong_count[protected].sum()+ (cc.sum(1)-cc.max(1))[~protected].sum())
    usual_floor=int((cc.sum(1)-cc.max(1)).sum())
    cohort=np.load(OLD/'fit_457_cohort.npy');remaining=cohort[pb[cohort]!=2]
    assert len(remaining)==439 and not protected[fid[remaining]].any()
    roles={'selected_fit':selected,'fit':~r.fold.isin([0,2]).to_numpy(),'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    table=[];details=[]
    oldgap=np.sort(za,axis=1)[:,-1]-np.sort(za,axis=1)[:,-2]
    for role,mask in roles.items():
        oc=(pa==y)&mask;nc=(pb==y)&mask;pf=~(pa==y)&nc;nf=oc&~(pb==y);change=(pa!=pb)&mask
        table.append({'role':role,'rows':int(mask.sum()),'old_errors':int((mask&~oc).sum()),
           'new_errors':int((mask&~nc).sum()),'positive_flips':int(pf.sum()),'negative_flips':int(nf.sum()),
           'changed_decisions':int(change.sum()),'wrong_to_different_wrong':int((change&~oc&~nc).sum()),
           'negative_flip_rate_all_rows':float(nf.sum()/mask.sum()),
           'negative_flip_rate_among_old_correct':float(nf.sum()/oc.sum()),
           'correct_fraction_of_changed_decisions':float(pf.sum()/change.sum()) if change.any() else None,
           'negative_flips_by_class':np.bincount(y[nf],minlength=3).tolist(),
           'positive_flips_by_class':np.bincount(y[pf],minlength=3).tolist()})
        for group,take in [('old_wrong',mask&(pa!=y)),('old_correct',oc),('negative_flip',nf),('positive_flip',pf)]:
            details.append({'role':role,'group':group,'rows':int(take.sum()),'old_top_two_gap_quantiles':quantile(oldgap[fid[take]])})
    # Infinitesimal gradient interference only; never apply a weight update.
    # Compare mean OVR gradients of the 439 known S errors and selected fit
    # records correctly classified by the old model, using linear coordinates.
    def gradient(ix):
        zz=za[fid[ix]];res=expit(zz);res[np.arange(len(ix)),y[ix]]-=1
        return np.r_[(np.asarray(x[fid[ix]].T@res)/len(ix)).ravel(),res.mean(0)]
    gerr=gradient(remaining);grads={}
    for cls,name in [(0,'B'),(1,'M'),(2,'S')]:
        take=np.flatnonzero(selected&(pa==y)&(y==cls));g=gradient(take)
        dot=float(gerr@g);norm=float(np.linalg.norm(gerr)*np.linalg.norm(g))
        grads[name]={'protected_rows':len(take),'gradient_cosine':dot/norm if norm else None,
           'directional_protected_loss_change_per_unit_error_descent':-dot/float(np.linalg.norm(gerr))}
    # A confidence gate cannot observe correctness on unknown inputs. Record
    # model score gaps, not calibrated probabilities or an H-tuned threshold.
    positive_feature_groups=np.unique(fid[selected&(pa!=y)&(pb==y)])
    result={'status':'no_fit_preservation_diagnosis','new_classifier_fits':0,'new_calibration_fits':0,
      'target_answers_read':False,'source_sha256':sha(__file__),
      'transition_table':table,'gap_diagnostics':details,
      'fit_input_lock_feasibility':{'selected_rows':int(selected.sum()),'protected_encoded_inputs':int(protected.sum()),
         'protected_inputs_with_other_labels':int(protected_conflict.sum()),
         'wrong_rows_in_protected_inputs':int(oldwrong_count[protected_conflict].sum()),
         'empirical_minimum_errors_with_unrestricted_deterministic_input_lookup':usual_floor,
         'empirical_minimum_errors_keeping_every_old_correct_fit_decision':lock_floor,
         'remaining_439_exact_input_conflicts_with_protected_decisions':int(protected[fid[remaining]].sum()),
         'actual_positive_flip_groups_conflicting_with_old_correct_fit':int(protected[positive_feature_groups].sum()),
         'scope':'Finite selected fit under current R0 only; not a raw-data Bayes bound, attainable learned-model score, or unseen-input guarantee.'},
      'local_gradient_interference':{'coordinates':'Current full-R0 linear weights and intercept; no kernel, regularizer or update applied.',
         'error_rows':439,'protected_classes':grads,
         'scope':'First-order average OVR loss interaction at the old model; not proof of future label flips or of generalization.'},
      'scope':'Frozen v82 model replay and local derivative analysis only. C/H are inspected development, never protection-training labels.'}
    save(DEST/'diagnosis.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
