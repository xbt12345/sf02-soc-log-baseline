"""Final factual counts, immutable source checks and zero-fit direction closeout."""
from pathlib import Path
import numpy as np
import pandas as pd
from v135_runtime import ROOT,OUT,read,save,sha,require_run_seal,load_data,fit_context
from v136_verify_design import validate,PLAN,DEST

def main():
    require_run_seal(ROOT/'training/v135_train.py');validate(read(PLAN))
    preflight=read(DEST/'design_preflight.json')
    if sha(ROOT/'training/v136_verify_design.py')!=preflight['source_sha256']:raise ValueError('Preflight source changed')
    x,d=load_data();counts=np.zeros(3,dtype=np.int64);bound={Path(__file__),PLAN,DEST/'design_preflight.json'}
    originals=0
    for fold in range(3):
        fit,c,pure,totals,ids=fit_context(d,fold);frozen_path=DEST/f'fold{fold}_R_decay_frozen.npz'
        a=np.load(frozen_path);q=a['mean_probability'];mp=a['member_pred'];ii=fit.local.to_numpy();y=fit.truth.to_numpy()
        mask=pure[ii].astype(bool)&(q[ii].argmax(1)!=y);n=(mp[ii[mask]]==y[mask,None]).sum(1)
        counts+=np.array([int(((n>0)&(n<8)).sum()),int((n>8).sum()),int((n==8).sum())]);originals+=int(mask.sum())
        bound.add(frozen_path)
        for arm in ['R_const','R_decay','O_const','O_decay']:
            folder=OUT/f'fold{fold}_{arm}';cp=read(folder/'checkpoints.json')[-1]
            if sha(folder/'sealed_all_prob.npy')!=cp['all_prob_sha256']:raise ValueError('V135 all-input probabilities changed')
    if originals!=80 or counts.tolist()!=[72,6,2]:raise ValueError('Wrong actual correct-member counts')
    doc=ROOT/'docs/V136_ROOT_DIAGNOSIS_AND_TARGETED_TRAINING_PLAN.md';text=doc.read_text(encoding='utf-8')
    if '72/80' not in text or '70/80' in text or '2/80有8个' not in text:raise ValueError('Narrative counts inconsistent')
    delivery=read(DEST/'review_delivery.json')
    if delivery['direction_sha256']!=sha(doc) or delivery['plan_sha256']!=sha(PLAN):raise ValueError('Design delivery hash mismatch')
    catalog=read(ROOT/'mcp_readonly/catalog.json')
    if catalog['project']['authoritative_delivery_id']!='v135-delivery' or catalog['project']['authoritative_direction_id']!='v136-review':raise ValueError('Wrong source authority')
    bound.update([doc,ROOT/'README.md',ROOT/'mcp_readonly/catalog.json',DEST/'review_delivery.json',
        ROOT/'training/v136_publish_design.py',ROOT/'mcp_readonly/tests/test_readonly_mcp.py'])
    require_run_seal(ROOT/'training/v135_train.py')
    target=DEST/'completion_receipt.json'
    if target.exists():raise FileExistsError(target)
    save(target,{'status':'zero_fit_review_design_closed','new_fits':0,'new_updates':0,'latest_actual_training':'V135',
        'model_promoted':False,'quality_acceptance':False,'new_runtime_ready':False,
        'actual_R_decay_residual_roles':80,'minority_correct_members_roles':72,'majority_correct_members_roles':6,'tie_roles':2,
        'plan_negative_cases_rejected':12,'historical_review_tests_passed':14,'readonly_MCP_tests_passed':10,
        'old_training_source_seal_intact':True,'cloud_MCP_connection_verified':False,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(bound)},
        'limits':['Checks support evidence and design constraints, not future training correctness or quality.',
                  'Earlier commentary minority count 70 was corrected to 72 after actual-row recomputation.']})
    print({'new_fits':0,'new_updates':0,'member_counts':counts.tolist(),'old_source_seal_intact':True},flush=True)

if __name__=='__main__':main()
