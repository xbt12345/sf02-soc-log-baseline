"""Prospective acceptance contract; does not reselect the frozen v81 trajectory."""
import numpy as np


def compare(base, candidate, strict_improvement):
    a,b=np.asarray(base),np.asarray(candidate)
    for cm in (a,b):
        if cm.shape!=(3,3) or not np.issubdtype(cm.dtype,np.integer) or np.any(cm<0):
            raise ValueError('Expected nonnegative integer 3x3 confusion matrices')
    if not np.array_equal(a.sum(1),b.sum(1)):
        raise ValueError('Different true-class supports cannot be paired')
    if a.sum()==0:
        return {'eligible':False,'reason':'no_support','checks':{}}
    aa,bb=a.diagonal(),b.diagonal();ap,bp=a.sum(0),b.sum(0);support=a.sum(1)
    ae=int(a.sum()-aa.sum());be=int(b.sum()-bb.sum())
    checks={'total_errors':be<ae if strict_improvement else be<=ae,
            'class_recalls':bool(np.all(bb>=aa)),
            'class_precisions':bool(np.all(bb*ap>=aa*bp)),
            'class_f1':bool(np.all(bb*(support+ap)>=aa*(support+bp))),
            'absent_class_false_predictions':bool(np.all(bp[support==0]<=ap[support==0])),
            'normal_false_alerts':int(b[0,1:].sum())<=int(a[0,1:].sum())}
    return {'eligible':all(checks.values()),'checks':checks,'base_errors':ae,'candidate_errors':be}


def accept_candidate(base, candidate, paired_subgroups):
    """Overall gain + supported subgroup non-regression, not improvement everywhere."""
    main=compare(base,candidate,True)
    groups={name:compare(a,b,False) for name,(a,b) in paired_subgroups.items()}
    missing=[name for name,r in groups.items() if r.get('reason')=='no_support']
    return {'eligible':main['eligible'] and not missing and all(v['eligible'] for v in groups.values()),
            'overall':main,'subgroups':groups,'unsupported_required_subgroups':missing,
            'scope':'Prospective development selection only; final engineering and transfer gates remain separate.'}
