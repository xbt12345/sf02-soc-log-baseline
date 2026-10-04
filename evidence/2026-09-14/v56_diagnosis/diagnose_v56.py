"""Post-hoc repair/regression and correction-level ablation; no new training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from scipy.special import softmax
import run_v56 as t
import run_v54 as v
from verify_v56 import independent_keys,lookup_codes,logits
ROOT=t.ROOT;RUN=t.RUN;OUT=ROOT/'evidence/2026-09-14/v56_diagnosis'

def main():
 assert not OUT.exists();OUT.mkdir(parents=True);c,r,x=t.load();items=[]
 for protocol in c['protocols']:
  folder=RUN/protocol;selected=v.read(folder/'selection.json')['support']['selected'];bundle=joblib.load(folder/(selected+'.joblib'));ev=pd.read_parquet(RUN/(protocol+'_split.parquet')).role.eq('evaluation').to_numpy();tr=pd.read_parquet(RUN/(protocol+'_split.parquet')).role.eq('fit').to_numpy();base=pd.read_parquet(folder/'base_evaluation.parquet');new=pd.read_parquet(folder/'support_evaluation.parquet');labels={k:set(a.label) for k,a in pd.DataFrame({'key':v.frame_keys(x.loc[tr]),'label':r.loc[tr,'label_index'].to_numpy()}).groupby('key')}
  for view,cols in v.VIEWS.items():
   xx=x.loc[ev].copy();xx[cols]=v.MISSING;codes=lookup_codes(bundle['design'],xx);b=base[base.scenario==view].reset_index(drop=True);a=new[new.scenario==view].reset_index(drop=True);y=a.label_index.to_numpy();keep=(a.route=='asa').to_numpy()&a.eligible_stress.to_numpy();bp=b[['p_0','p_1','p_2']].to_numpy();repair=(b.pred!=y).to_numpy()&(a.pred==y).to_numpy();regress=(b.pred==y).to_numpy()&(a.pred!=y).to_numpy();k=independent_keys(xx)['exact'];strata=np.array(['unseen' if z not in labels else ('conflicted' if len(labels[z])>1 else 'pure_seen') for z in k]);ablations={}
   for levelset in [('coarse',),('coarse','fine'),('coarse','fine','exact')]:
    pp=softmax(logits(bp,{k:z for k,z in codes.items() if k in levelset},bundle['weights']),axis=1);yp=pp.argmax(1);ablations['+'.join(levelset)]=v.metrics(y[keep],yp[keep])
    if len(levelset)==3:np.testing.assert_allclose(pp,a[['p_0','p_1','p_2']],atol=3e-6,rtol=0);np.testing.assert_array_equal(yp,a.pred)
   inactive=np.ones(len(xx),dtype=bool)
   for z in codes.values():inactive &= z<0
   exactsupport=np.zeros(len(xx));hit=codes['exact']>=0;exactsupport[hit]=bundle['design']['bodies'][codes['exact'][hit]]
   items.append({'protocol':protocol,'view':view,'repaired':int((repair&keep).sum()),'regressed':int((regress&keep).sum()),'no_correction_level_active_rows':int((inactive&keep).sum()),'inactive_errors':int(((b.pred!=y).to_numpy()&inactive&keep).sum()),'probabilities_changed_rows':int(((abs(a[['p_0','p_1','p_2']].to_numpy()-bp).max(1)>1e-8)&keep).sum()),'repair_regression_by_support':{s:{'rows':int(((strata==s)&keep).sum()),'repair':int((repair&keep&(strata==s)).sum()),'regression':int((regress&keep&(strata==s)).sum())} for s in np.unique(strata)},'repaired_exact_support_le_two':int((repair&keep&(exactsupport>0)&(exactsupport<=2)).sum()),'ablations':ablations})
 v.save(OUT/'diagnosis.json',{'results':items,'scope':'Post-selection diagnostics, not selection or a refitted ablation. Removing fitted correction levels is not equivalent to retraining without them. Exact-key support is fit body-count proxy, not known independent incidents.'});(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}});print(json.dumps([a for a in items if a['view']=='full']),flush=True)
if __name__=='__main__':main()
