"""No new fits: one preregistered-in-code equal B69/R71 score average.

Historical adaptive development diagnostic only. Common fit/cal/eval sources.
No weight/gate search; oracle union is NOT a deployable model or score upper bound.
"""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training'))
from run_v69_support_control import load,hard,threshold,rates,ranking,BUDGETS
from run_v67_targeted import counts
from verify_v69_support_control import independent_cut


def main():
    inputs={}
    def get(path):
        path=ROOT/path;inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_parquet(path)
    def getjson(path):
        path=ROOT/path;inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding='utf-8'))
    f,ctx,m=load();original=f.set_index('row_position').loc[m.row_position].reset_index()
    h=hard(original);base=m.baseline_pred_with_normal_gate.to_numpy();target=m.target_578.to_numpy()
    score=np.full(len(m),np.nan);cuts={};geometry=[]
    allarm={}
    for arm,path in [('B','artifacts/v69_support_control_20260921/B_small_sources_budget0.01_oof.parquet'),
                     ('Q','artifacts/v71_resolution_20260921/Q_old_bins_budget0.01_oof.parquet'),
                     ('R','artifacts/v71_resolution_20260921/R_observed_budget0.01_oof.parquet')]:
        d=get(path);assert np.array_equal(d.row_position,m.row_position);allarm[arm]=d.pred.to_numpy()
    for fold in range(3):
        rb=get(f'artifacts/v69_support_control_20260921/fold{fold}_roles.parquet')
        rr=get(f'artifacts/v71_resolution_20260921/fold{fold}_roles.parquet')
        for col in ['row_position','B_small_sources','calibration','evaluation']:np.testing.assert_array_equal(rb[col],rr[col])
        frames={}
        for role in ['fit','calibration','evaluation']:
            b=get(f'artifacts/v69_support_control_20260921/fold{fold}/B_small_sources/{role}.parquet')
            r=get(f'artifacts/v71_resolution_20260921/fold{fold}/R_observed/{role}.parquet')
            for col in ['row_position','label','group','transport_protocol']:np.testing.assert_array_equal(b[col],r[col])
            b=b.copy();b['score']=.5*b.score.to_numpy()+.5*r.score.to_numpy();frames[role]=b
        cuts[fold]={}
        for budget in BUDGETS:
            cuts[fold][str(budget)]={}
            for proto in ['tcp','udp']:
                d=frames['calibration'];d=d[d.transport_protocol.eq(proto)]
                cut=threshold(d,d.score,budget);assert cut==independent_cut(d,budget)
                cuts[fold][str(budget)][proto]=cut
        for role,d in frames.items():
            for proto,z in d.groupby('transport_protocol'):
                cut=cuts[fold]['0.01'][proto];oc=independent_cut(z,.01)
                geometry.append({'fold':fold,'role':role,'protocol':proto,
                    'frozen_cut_rates':rates(z,np.where(z.score>=cut,2,1)),
                    'own_role_oracle_rates':rates(z,np.where(z.score>=oc,2,1)), 'ranking':ranking(z,z.score)})
        idx=pd.Index(m.row_position).get_indexer(frames['evaluation'].row_position);assert (idx>=0).all()
        score[idx]=frames['evaluation'].score
    assert np.isfinite(score[h]).all() and np.isnan(score[~h]).all()
    results={}
    for budget in BUDGETS:
        pred=base.copy();cells=[]
        for fold in range(3):
            for proto in ['tcp','udp']:
                ix=h & original.fold.eq(fold).to_numpy() & original.transport_protocol.eq(proto).to_numpy()
                pred[ix]=np.where(score[ix]>=cuts[fold][str(budget)][proto],2,1)
                cells.append({'fold':fold,'protocol':proto,**rates(original[ix],pred[ix])})
        results[str(budget)]={'vs_historical':counts(original,pred,base,target),
             'vs_B69':counts(original,pred,allarm['B'],target),'hard_rates':rates(original[h],pred[h]),'cells':cells,
             'confusion_B_M_S':confusion_matrix(original.label,pred,labels=[0,1,2]).tolist(),
             'row_report':classification_report(original.label,pred,labels=[0,1,2],output_dict=True,zero_division=0)}
        if budget==.01:
            allarm['BR_equal']=pred
            pd.DataFrame({'row_position':m.row_position,'label':m.label,'group':m.group,'score':score,'pred':pred}).to_parquet(Path(__file__).with_name('fixed_blend_decisions.parquet'),index=False)
    union={}
    for names in [('B','R'),('B','Q','R')]:
        flags=np.column_stack([allarm[a]==original.label.to_numpy() for a in names])
        union['+'.join(names)]={'scope':'Label-aware union of existing decisions only, not trainable routing or bound on all blends',
          'target_correct_by_any':int(flags[target].any(axis=1).sum()),'target_common_wrong':int((~flags[target].any(axis=1)).sum()),
          'all_rows_oracle_errors':int((~flags.any(axis=1)).sum()),'all_rows_all_models_correct':int(flags.all(axis=1).sum())}
    pair=[]
    for col in [0,1,2]:
        mask=original.label.eq(col).to_numpy();bc=allarm['B'][mask]==col;rc=allarm['R'][mask]==col
        pair.append({'label':col,'rows':int(mask.sum()),'both_wrong':int((~bc&~rc).sum()),'B_only_correct':int((bc&~rc).sum()),
                     'R_only_correct':int((~bc&rc).sum()),'both_correct':int((bc&rc).sum())})
    primary=results['0.01'];cells=primary['cells']
    result={'new_fits':0,'experiment':'Exactly one fixed equal score blend B69+R71; same frozen source roles; no weights selected',
      'scope':'Already inspected adaptive development. Recalibrated fixed blend is a diagnostic candidate, not an accepted model.',
      'primary_budget':.01,'results':results,'role_geometry':geometry,'calibration_thresholds':cuts,
      'primary_all_M_budgets_pass':all(max(1-v['M']['row_recall'],1-v['M']['source_recall'])<=.01+1e-12 for v in cells),
      'decision_oracle_unions':union,'B_R_error_overlap':pair,'quality_acceptance':False,
      'inputs_sha256':inputs,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'primary':primary,'decision_unions':union,'overlap':pair},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
