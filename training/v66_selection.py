"""One comparator for both arms, normal controls fail closed, no replacement of old results."""
from v64_selection import select as old_select


def guarded_select(curve,references):
    if not references:raise ValueError('References required')
    supports=[r.get('normal_support') for r in references.values()]
    if any(not isinstance(n,int) or n<=0 for n in supports):
        return {'selected':None,'status':'normal_validation_unsupported','screen_passed':False,
                'quality_acceptance':False}
    if len(set(supports))!=1:raise ValueError('References must evaluate identical normal controls')
    for m in list(references.values())+[c['metrics'] for c in curve]:
        errors=m.get('normal_control_errors')
        if not isinstance(errors,int) or not 0<=errors<=supports[0]:
            raise ValueError('Invalid normal error count')
    cap=min(r['normal_control_errors'] for r in references.values())
    kept=[];rejected=[]
    for candidate in curve:
        m=candidate['metrics']
        if m.get('normal_support')!=supports[0]:raise ValueError('Candidate control membership/count mismatch')
        if m['normal_control_errors']>cap:rejected.append(candidate['step'])
        else:kept.append(candidate)
    result=old_select(kept,references)
    result.update(normal_error_cap=cap,normal_rejected_steps=rejected,
                  quality_acceptance=False,scope='Control regression screening only; small controls cannot validate FPR')
    return result


def select_both(arms,references):
    # No H0 row-first versus H1 S-first asymmetry here.
    if set(arms)!={'H0','H1'}:raise ValueError('Exactly the two comparison arms are required')
    if len(arms['H0'])!=len(arms['H1']):raise ValueError('Checkpoint search budgets must match')
    if [c['step'] for c in arms['H0']]!=[c['step'] for c in arms['H1']]:
        raise ValueError('Checkpoint schedules must match')
    return {arm:guarded_select(curve,references) for arm,curve in arms.items()}
