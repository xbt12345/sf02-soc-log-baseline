"""Matched A/B directions, one common scalar displacement scale, no optimizer."""
import numpy as np
from v159_mgda_synthetic_qualification import exact_two_gradient_direction


def class_direction(g_m, g_s, mass_m, mass_s, arm):
    gm, gs = np.asarray(g_m, np.float64), np.asarray(g_s, np.float64)
    if gm.shape != gs.shape or gm.size == 0 or not np.isfinite(gm).all() or not np.isfinite(gs).all():
        raise ValueError('Complete finite matched class gradients required')
    if mass_m <= 0 or mass_s <= 0 or arm not in ('A','B'):
        raise ValueError('Both original classes and registered arm required')
    if arm == 'A':
        alpha = float(mass_m / (mass_m + mass_s))
        raw = -(alpha * gm + (1-alpha) * gs)
    else:
        alpha, raw = exact_two_gradient_direction(gm,gs)
    scale = float(np.max(np.abs(raw)))
    if scale == 0:
        return dict(status='zero_direction_no_automatic_fallback',M_weight=alpha,direction=raw,raw_infinity_norm=0.,class_slopes=[0.,0.])
    direction = raw / scale
    slopes = [float(np.vdot(g,direction)) for g in (gm,gs)]
    if not np.isfinite(direction).all() or not np.isfinite(slopes).all():
        raise ValueError('Nonfinite normalized direction')
    if arm == 'B' and max(slopes) >= 0:
        return dict(status='no_strict_common_descent_no_automatic_fallback',M_weight=alpha,direction=direction,raw_infinity_norm=scale,class_slopes=slopes)
    return dict(status='direction_qualified_for_finite_guarded_proposal',M_weight=alpha,direction=direction,raw_infinity_norm=scale,class_slopes=slopes)


def finite_armijo(base_class_risks, proposal_class_risks, class_slopes, mass_m, mass_s, arm, step, classification_guard):
    before,after,slopes=[np.asarray(v,np.float64) for v in (base_class_risks,proposal_class_risks,class_slopes)]
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in (before,after,slopes)):
        return False
    if step <= 0 or arm not in ('A','B') or not classification_guard:
        return False
    bounds=before+1e-4*step*slopes
    if arm == 'B':return bool(max(slopes)<0 and np.all(after<=bounds))
    weights=np.array([mass_m,mass_s],np.float64)/(mass_m+mass_s)
    return bool(float(weights@slopes)<0 and float(weights@after)<=float(weights@bounds))
