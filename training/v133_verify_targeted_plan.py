"""V133 zero-fit evidence/plan validation. This is not a training runtime guard."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v133_training_design_review_20260930'
PLAN=ROOT/'training/review_policy/v133_targeted_learning_plan.json'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def validate_plan(p):
    require(p['status']=='designed_not_trained','Plan is not execution evidence')
    arms=p['primary']['arms']
    require(arms=={'R_const':{'loss':'row_CE','schedule':'constant'},
                  'R_decay':{'loss':'row_CE','schedule':'late_cosine'},
                  'O_const':{'loss':'focused_CE','schedule':'constant'},
                  'O_decay':{'loss':'focused_CE','schedule':'late_cosine'}},'Incomplete/changed factorial')
    require(p['model']['hidden']==128 and p['model']['members']==16,'Model change confounds factors')
    require(p['primary']['folds']==[0,1,2] and p['primary']['fixed_epochs']==100,'Population or endpoint changed')
    require(p['primary']['early_stop'] is False,'Premature learning stop')
    s=p['primary']['late_cosine']
    require(s=={'constant_through_epoch':80,'end_epoch':100,'initial_lr':.002,'final_lr':.00002},'Changed schedule')
    require(p['primary']['fresh_control_fits'] is True,'Historical baseline is not the new matched control')
    require(p['primary']['fits']==12 and p['primary']['updates']==71200,'Wrong primary budget')
    require(p['fit_budget']['maximum_fits']==15 and p['fit_budget']['maximum_updates']==89000,'Wrong confirmation budget')
    require(p['selection']['task_candidate']=='R_decay','Posthoc task candidate swap')
    require(p['selection']['may_promote_other_arms_using_held_scores'] is False,'Held-driven selection')
    require(p['selection']['freeze_all_training_qualifications_before_held_read'] is True,'Held exposure before qualification')
    require(p['mastery_gate']['fixed_epochs']==[96,97,98,99,100],'Sliding window acceptance')
    require(p['mastery_gate']['pure_TRAIN_M_errors_max']==0 and p['mastery_gate']['pure_TRAIN_S_errors_max']==0,'Panel-only qualification')
    require(p['record']['no_source_history_overwrite'] is True and p['record']['every_epoch_original_row_predictions'] is True,'Incomplete histories')
    require(p['confirmation']['requires_training_and_quality'] is True and p['confirmation']['seed']==13701,'Unqualified confirmation')
    require(p['confirmation']['automatic_promotion'] is False,'Confirmation is not deployment')
    require(p['runtime_readiness']['trainer_implemented'] is False and p['runtime_readiness']['run_seal_created'] is False,'Do not fabricate readiness')
    q=p['matched_control_quality_gate']
    require(q=={'M_errors_do_not_increase':True,'S_errors_do_not_increase':True,
                'at_least_one_class_strictly_improves':True,'improved_outer_folds_min':2},'Matched-control protection missing')


def adversaries(plan):
    cases=[]
    edits=[('missing original objective',lambda p:p['primary']['arms'].pop('R_const')),
           ('premature decay',lambda p:p['primary']['late_cosine'].update(constant_through_epoch=60)),
           ('posthoc arm switch',lambda p:p['selection'].update(task_candidate='O_decay')),
           ('width confound',lambda p:p['model'].update(hidden=256)),
           ('source history overwrite',lambda p:p['record'].update(no_source_history_overwrite=False)),
           ('weak 95pct qualifier',lambda p:p['mastery_gate'].update(pure_TRAIN_S_errors_max=26)),
           ('historical matched control substitution',lambda p:p['primary'].update(fresh_control_fits=False)),
           ('quality-failed confirmation',lambda p:p['confirmation'].update(requires_training_and_quality=False))]
    for name,edit in edits:
        bad=copy.deepcopy(plan);edit(bad)
        try:
            validate_plan(bad)
        except ValueError:
            cases.append({'case':name,'rejected':True})
        else:
            raise AssertionError('Invalid plan accepted: '+name)
    return cases


def main():
    p=read(PLAN);validate_plan(p)
    for path,h in p['evidence_sha256'].items():
        require(sha(ROOT/path)==h,'Evidence changed: '+path)
    a=read(OUT/'supervision_audit.json');t=read(OUT/'learning_trajectory_evidence.json')
    for receipt in (a,t):
        for path,h in receipt['source_sha256'].items():
            require(sha(ROOT/path)==h,'Audited source changed: '+path)
    # Old actual training source seal stays intact, including old checkers/trainers.
    for path,h in read(ROOT/'artifacts/v131_learning_trial_20260930/run_seal.json')['source_sha256'].items():
        require(sha(ROOT/path)==h,'V131 sealed source changed: '+path)
    d=pd.read_parquet(ROOT/'artifacts/v132_training_mastery_review_20260930/training_mastery_ledger.parquet')
    labels=pq.read_table(ROOT/'data/official/train.parquet',columns=['label_binary']).column(0).to_pandas().map(
        {'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    require(len(labels)==2056871 and not pd.isna(labels).any(),'Incomplete reference truth')
    require(np.array_equal(d.truth,labels[d.row_position.to_numpy()]),'Official reference mismatch')
    for row in a['roles']:
        f=d[d.arm.eq('R') & d.outer_fit_role.eq(row['fold'])]
        require(not f.row_position.duplicated().any(),'Duplicated original rows')
        pure=f.canonical_key.map(f.groupby('canonical_key').truth.nunique()).eq(1)
        n=len(f);ns=int((f.truth.eq(2)&pure).sum());nm=int((f.truth.eq(1)&pure).sum())
        require(abs(row['O_S_coefficient_mass']-(.5*f.truth.eq(2).mean()+.25))<1e-12,'S coefficient recount mismatch')
        require(abs(row['pure_S_per_row_coefficient_multiplier']-(.5+.25*n/ns))<1e-12,'Pure S coefficient mismatch')
        require(abs(row['pure_M_per_row_coefficient_multiplier']-(.5+.25*n/nm))<1e-12,'Pure M coefficient mismatch')
    h=pd.read_parquet(ROOT/'artifacts/v131_learning_trial_20260930/ASA_prediction_ledger.parquet')
    require(np.array_equal(h.truth,labels[h.row_position.to_numpy()]),'Held reference mismatch')
    for row in a['fold1_R_O_paired_errors']:
        cls=row['truth'];f=h[h.fold.eq(1)&h.truth.eq(cls)]
        r=f.pred_R.ne(cls);o=f.pred_O.ne(cls)
        require(int((r&~o).sum())==row['O_repairs_vs_R'],'Repair mismatch')
        require(int((~r&o).sum())==row['O_new_errors_vs_R'],'Regression mismatch')
        require(int(o.sum()-r.sum())==row['O_new_errors_vs_R']-row['O_repairs_vs_R'],'Net hides gross error accounting')
    keyed={(v['fit'],v['epoch']):v for v in t['rows']}
    require(keyed['fold1_O',60]['M_errors']==244 and keyed['fold1_O',80]['M_errors']==28,'Learning trajectory identity mismatch')
    cases=adversaries(p)
    for e in (81,90,100):
        lr=.00002+.5*(.002-.00002)*(1+math.cos(math.pi*(e-80)/20))
        require(.00002-1e-12<=lr<.002,'Invalid late schedule')
    result={'status':'v133_evidence_and_design_checks_passed_not_training_ready',
            'new_fits':0,'new_updates':0,'independent_official_reference_rows':len(labels),
            'plan_sha256':sha(PLAN),'checker_sha256':sha(__file__),'adversaries':cases,
            'task_candidate':'R_decay','training_solution_achieved':False,'model_promoted':False,
            'limits':['Design mutation rejection is not a implemented training runtime guard.',
                      'No new model fit, fixed-window replay or source-generalization acceptance.',
                      'No statistical causal proof that O coefficient mass explains all observed errors.']}
    (OUT/'plan_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
