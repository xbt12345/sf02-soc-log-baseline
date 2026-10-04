"""Create new immutable candidate versions; old failed source/results stay."""
from experiment_review import ROOT
def main():
    old=ROOT/'training/v159_float64_repeat_policy.py';target=old.with_name('v159_float64_repeat_policy_v2.py');assert not target.exists()
    text=old.read_text(encoding='utf-8');start=text.index('def repeat_direction(');end=text.index('\ndef finite_step_review',start)
    replacement='''def resolved_class_direction(g_m,g_s,mass_m,mass_s,arm):
    result=class_direction(g_m,g_s,mass_m,mass_s,arm)
    direction=result['direction'];dnorm=float(np.linalg.norm(direction))
    margins=[STEP_EPS*EPS*float(np.linalg.norm(g))*dnorm for g in [g_m,g_s]]
    result['slope_resolution']=margins
    if arm=='B' and result['raw_infinity_norm']>0 and not all(s<-e for s,e in zip(result['class_slopes'],margins)):
        result['status']='no_resolved_common_descent_no_automatic_fallback'
    if arm=='A' and result['raw_infinity_norm']>0:
        w=float(mass_m/(mass_m+mass_s));g=w*np.asarray(g_m)+(1-w)*np.asarray(g_s)
        margin=STEP_EPS*EPS*float(np.linalg.norm(g))*dnorm
        result['mean_slope_resolution']=margin
        if not w*result['class_slopes'][0]+(1-w)*result['class_slopes'][1]<-margin:
            result['status']='no_resolved_mean_descent_no_automatic_fallback'
    return result

def repeat_direction(g0,g1,mass_m,mass_s,arm):
    a,b=[resolved_class_direction(*g,mass_m,mass_s,arm) for g in [g0,g1]]
    if a['status']!=b['status']:return dict(passed=False,reason='resolved_direction_status_changed')
    reports={k:repeat_values(a[k],b[k],k) for k in ['direction','M_weight']}
    slopes_a,slopes_b=[np.asarray(r['class_slopes']) for r in [a,b]]
    margins=np.maximum(a['slope_resolution'],b['slope_resolution'])
    gaps=np.abs(slopes_a-slopes_b)
    resolved=(np.maximum(np.abs(slopes_a),np.abs(slopes_b))>margins)
    signs=bool(np.array_equal(np.sign(slopes_a[resolved]),np.sign(slopes_b[resolved])))
    reports['class_slopes']=dict(passed=bool(np.all(gaps<=2*margins) and signs),maximum_absolute=float(gaps.max()),dot_product_scale_limits=(2*margins).tolist(),resolved_descent_signs_exact=signs,raw_descent_signs_exact=bool(np.array_equal(np.sign(slopes_a),np.sign(slopes_b))))
    return dict(passed=all(r['passed'] for r in reports.values()),status=a['status'],descent_qualified=a['status']=='direction_qualified_for_finite_guarded_proposal',comparisons=reports)
'''
    target.write_text(text[:start]+replacement+text[end:],encoding='utf-8')
    oldq=ROOT/'training/v159_nonzero_real_dimension_numeric_qualification.py';newq=oldq.with_name('v159_nonzero_real_dimension_numeric_qualification_v2.py');assert not newq.exists()
    text=oldq.read_text(encoding='utf-8').replace('from v159_float64_repeat_policy import','from v159_float64_repeat_policy_v2 import').replace("training/v159_float64_repeat_policy.py","training/v159_float64_repeat_policy_v2.py").replace("v159_nonzero_real_dimension_numeric_qualification_20261002'","v159_nonzero_real_dimension_numeric_qualification_v2_20261002'")
    text=text.replace("ROOT/'training/v159_class_direction.py',","ROOT/'training/v159_class_direction.py',ROOT/'training/v159_float64_repeat_policy.py',ROOT/'artifacts/v159_nonzero_real_dimension_numeric_qualification_20261002/failure.json',Path(__file__).with_name('v159_build_resolved_direction_numeric_revision.py'),")
    text=text.replace("assert report['complete_parameter_gradient_nonzero'] and (scale==0 or report['hidden_gradient_nonzero'])","assert report['complete_parameter_gradient_nonzero'] and (scale==0 or report['hidden_gradient_nonzero'])\n        if scale==8.:assert dirs[1]['status']=='no_resolved_common_descent_no_automatic_fallback' and not dirs[1]['descent_qualified']\n        else:assert dirs[1]['descent_qualified']")
    newq.write_text(text,encoding='utf-8');print('new v2 policy and qualification sources created; no calls')
if __name__=='__main__':main()
