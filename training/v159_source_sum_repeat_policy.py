"""Propagate fixed per-row CE repeat envelopes; integer quality stays exact."""
import math
import numpy as np
from v159_float64_repeat_policy_v2 import EPS,REPEAT_EPS,repeat_values

def source_sum_repeat_review(actual_rows,saved_rows,actual_source,saved_source):
    identity=['row_position','root','truth','pred']
    if not np.array_equal(actual_rows[identity].to_numpy(),saved_rows[identity].to_numpy()):return dict(passed=False,reason='row_identity_truth_or_prediction')
    integer=['root','truth','support','errors']
    if not np.array_equal(actual_source[integer].to_numpy(),saved_source[integer].to_numpy()):return dict(passed=False,reason='source_identity_support_or_errors')
    quantities=[('stable_CE','stable_CE_sum'),('probability_clip_CE','probability_clip_CE_sum')]
    row_reviews={name:repeat_values(actual_rows[name].to_numpy(),saved_rows[name].to_numpy(),'row_'+name) for name,_ in quantities}
    if not all(z['passed'] for z in row_reviews.values()):return dict(passed=False,reason='row_CE_repeat',row_reviews=row_reviews)
    groups=actual_rows.groupby(['root','truth'],sort=True).indices;results=[]
    for j,(key,indices) in enumerate(groups.items()):
        assert tuple(actual_source.iloc[j][['root','truth']].to_numpy())==key
        for rowname,srcname in quantities:
            a,b=[frame[rowname].to_numpy()[indices] for frame in [actual_rows,saved_rows]]
            sa,sb=math.fsum(a),math.fsum(b);av,bv=float(actual_source.iloc[j][srcname]),float(saved_source.iloc[j][srcname])
            # Independently certify the stored and replayed reductions before
            # propagating their already fixed per-row error envelopes.
            reduction_a=2*EPS*max(1.,abs(sa));reduction_b=2*EPS*max(1.,abs(sb))
            sum_envelope=math.fsum(REPEAT_EPS*EPS*max(1.,abs(x),abs(y)) for x,y in zip(a,b))
            bound=sum_envelope+reduction_a+reduction_b;gap=abs(av-bv)
            passed=abs(av-sa)<=reduction_a and abs(bv-sb)<=reduction_b and gap<=bound
            results.append(dict(root=int(key[0]),truth=int(key[1]),quantity=srcname,support=len(indices),actual_difference=gap,propagated_row_envelope=sum_envelope,reduction_envelope=reduction_a+reduction_b,passed=bool(passed)))
    return dict(passed=all(z['passed'] for z in results),reason='propagated_fixed_row_CE_and_independently_certified_reductions',row_reviews=row_reviews,groups=results,learning_acceptance_relaxed=False)
