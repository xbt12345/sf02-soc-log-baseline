"""Legal-FIT-only algebraic prior, never chosen from outer quality."""
import numpy as np

def safe_legacy_prior(current,legacy,truth,legal_fit):
    if current.shape!=legacy.shape or current.shape!=(len(truth),3):raise ValueError('All three classes required')
    if not np.isfinite(current).all() or not np.isfinite(legacy).all():raise ValueError('Nonfinite saved probabilities')
    legal_fit=np.asarray(legal_fit,dtype=bool);truth=np.asarray(truth,dtype=np.int64)
    ids=np.flatnonzero(legal_fit&(current.argmax(1)==truth))
    if not len(ids):raise ValueError('No protected legal FIT rows')
    margins=current[ids,truth[ids],None]-current[ids]
    previous=legacy[ids,truth[ids],None]-legacy[ids]
    slope=previous-margins;harm=slope<0
    bound=float(np.min(margins[harm]/-slope[harm],initial=1.))
    alpha=.25*min(1.,bound)
    if alpha<=0:raise ValueError('No positive safe prior; stop, do not scan a different initialization')
    # Derived margin guarantee, not a prediction or threshold quality search.
    assert np.all((margins+alpha*slope)>=.75*margins-1e-15)
    return dict(legacy_prior=alpha,linear_class_preserving_bound=bound,protected_legal_FIT_rows=len(ids),safety_fraction=.25)
