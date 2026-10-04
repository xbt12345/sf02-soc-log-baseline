"""Replay the two separate-fit calibration controls and independently rank cutpoints."""
import json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/v55_risk_validation_20260914';RUN=ROOT/'artifacts/v55_calibration_control_r2_20260914';OUT=ROOT/'evidence/2026-09-14/v55_calibration_verification'
sys.path.insert(0,str(BASE));import run_v55 as n;import run_v54 as v
from verify_v55 import manual

def score_ratio(a):
 p=a[['p_0','p_1','p_2']].to_numpy()
 # Preserve the fitted output's float32 division, then compare cutpoints in float64.
 return (p[:,1]/np.maximum(p[:,1]+p[:,2],1e-30)).astype(np.float64)

def independent_choice(d):
 groups=[];all_scores=[]
 for view in n.VIEWS:
  a=d[(d.scenario==view)&(d.route=='asa')&d.eligible_stress];s=score_ratio(a);y=a.label_index.to_numpy();keep=a.pred.to_numpy()!=0
  groups.append((s,y,keep));all_scores.extend(s[keep])
 thresholds=np.unique(np.r_[0.,.5,np.nextafter(np.unique(all_scores),np.inf)]);columns=[];support=[]
 for s,y,keep in groups:
  for label in [1,2]:
   z=np.sort(s[(y==label)&keep]);lower=np.searchsorted(z,thresholds,side='left');count=int((y==label).sum());support.append(count)
   columns.append(lower+int(((y==label)&~keep).sum()) if label==1 else count-lower)
 errors=np.column_stack(columns);rates=errors/np.array(support);best=np.lexsort((thresholds,np.abs(thresholds-.5),rates.mean(1),rates.max(1)))[0]
 return float(thresholds[best]),errors[best],support,len(thresholds)

def main():
 assert not OUT.exists();OUT.mkdir(parents=True);c=v.read(RUN/'configuration.json');assert v.sha(RUN/'configuration.json')==v.read(RUN/'preregistered.json')['sha256']
 for base,key in [(ROOT,'source_bindings'),(RUN,'local_bindings')]:
  for p,h in c[key].items():assert v.sha(base/p)==h
 assert v.sha(ROOT/'training/probe_v55_calibration.py')==v.sha(RUN/'probe_v55_calibration.py')
 _,r,x=n.load(ROOT,BASE);report=[];maxgap=0.;replayed=0
 for split in n.SPLITS:
  m=pd.read_parquet(RUN/(split+'_split.parquet'));orig=pd.read_parquet(BASE/(split+'_split.parquet'));idx=np.flatnonzero(orig.role.eq('fit'));rr=r.iloc[idx].reset_index(drop=True)
  a,b=next(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20260915).split(rr,rr.label_index,rr.body_group));expected=np.full(len(r),'evaluation',dtype=object);expected[idx[a]]='fit';expected[idx[b]]='calibration'
  np.testing.assert_array_equal(m.role,expected);pd.testing.assert_frame_equal(m[['row_position','label_index','body_group']],r[['row_position','label_index','body_group']])
  for left,right in [('fit','calibration'),('fit','evaluation'),('calibration','evaluation')]:assert set(m.loc[m.role==left,'body_group']).isdisjoint(m.loc[m.role==right,'body_group'])
  np.testing.assert_array_equal(m.role.eq('evaluation'),orig.role.eq('validation'))
  folder=RUN/split
  for p,h in v.read(folder/'complete.json')['bindings'].items():assert v.sha(folder/p)==h
  before=v.read(folder/'before_evaluation.json');assert not before['evaluation_generated'];assert set(before['bindings'])=={'model.joblib','calibration.parquet','selection.json'}
  for p,h in before['bindings'].items():assert v.sha(folder/p)==h
  bundle=joblib.load(folder/'model.joblib');model=bundle['model'];assert bundle['configuration_sha256']==v.sha(RUN/'configuration.json');assert model.alpha==1. and model.random_state==c['seed'] and model.t_==int((m.role=='fit').sum())*10
  for j,field in enumerate(v.FIELDS):assert bundle['input']['seen'][j]==set(x.loc[m.role=='fit',field])
  cal=pd.read_parquet(folder/'calibration.parquet');ev=pd.read_parquet(folder/'evaluation.parquet')
  for role,d in [('calibration',cal),('evaluation',ev)]:
   mask=m.role.eq(role).to_numpy()
   for view,cols in n.VIEWS.items():
    z=d[d.scenario==view];pd.testing.assert_frame_equal(z[r.columns].reset_index(drop=True),r.loc[mask].reset_index(drop=True));xx=x.loc[mask].copy();xx[cols]=v.MISSING
    p=manual(bundle,xx);gap=float(abs(p-z[['p_0','p_1','p_2']].to_numpy()).max());maxgap=max(maxgap,gap);np.testing.assert_allclose(p,z[['p_0','p_1','p_2']],atol=3e-6,rtol=0);np.testing.assert_array_equal(p.argmax(1),z.pred)
    eligible=np.ones(mask.sum(),dtype=bool) if not cols else x.loc[mask,cols].ne(v.MISSING).any(axis=1).to_numpy();np.testing.assert_array_equal(z.eligible_stress,eligible);replayed+=len(z)
  t,errors,support,nchoices=independent_choice(cal);selected=v.read(folder/'selection.json');assert t==selected['threshold'];np.testing.assert_array_equal(errors,selected['calibration_errors_full_src_dst_M_S']);assert support==selected['calibration_class_support_full_src_dst_M_S'];assert nchoices==selected['candidates']
  pred=np.where(ev.pred.to_numpy()==0,0,np.where(score_ratio(ev)>=t,1,2));np.testing.assert_array_equal(pred,ev.threshold_pred);assert np.array_equal(pred==0,ev.pred.to_numpy()==0)
  # Direct chosen-cutpoint counting catches score ties and off-by-one frontier errors.
  direct=[]
  for view in n.VIEWS:
   a=cal[(cal.scenario==view)&(cal.route=='asa')&cal.eligible_stress];yp=np.where(a.pred.to_numpy()==0,0,np.where(score_ratio(a)>=t,1,2))
   direct.extend(int(((a.label_index==label)&(yp!=label)).sum()) for label in [1,2])
  np.testing.assert_array_equal(direct,errors);stored=v.read(folder/'summary.json');results={}
  for view in n.VIEWS:
   a=ev[(ev.scenario==view)&ev.eligible_stress&(ev.route=='asa')];b=ev[(ev.scenario==view)&(ev.route=='asa_acl')];item={}
   for key,frame,column in [('argmax',a,'pred'),('threshold',a,'threshold_pred'),('normal_argmax',b,'pred'),('normal_threshold',b,'threshold_pred')]:
    mm=v.metrics(frame.label_index.to_numpy(),frame[column].to_numpy());assert mm==stored[view][key];item[key]=mm
   results[view]=item
  report.append({'split':split,'threshold':t,'calibration_class_errors':direct,'rows_by_role':m.role.value_counts().to_dict(),'evaluation':results,'threshold_frozen_before_scoring':True,'same_model_no_post_calibration_refit':True});print(json.dumps({'split':split,'all_checks_passed':True}),flush=True)
 v.save(OUT/'verification.json',{'all_checks_passed':True,'actual_fits':2,'runs':report,'probability_rows_replayed':replayed,'max_probability_difference':maxgap,'scope':'Source/role/model replay and independent calibration-only cutpoint ranking. Two post-hoc development controls, not blind evaluation, probability calibration, external transfer or main-gate promotion.'})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}})
if __name__=='__main__':main()
