"""Independent shared-effect objective, selection and output verification."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import softmax,logsumexp
import v56_shared_control as s
import run_v54 as v
from verify_v56 import independent_keys,independent_rank,logits
from verify_v55 import manual
ROOT=s.ROOT;RUN=s.RUN;OUT=ROOT/'evidence/2026-09-14/v56_shared_verification'

def keyset(x,fields):return {**{'field:'+f:x[f].to_numpy() for f in fields},**independent_keys(x)}
def codes(b,x,fields):return {level:np.array([b['maps'][level].get(k,-1) for k in values]) for level,values in keyset(x,fields).items()}

def main():
 assert not OUT.exists();OUT.mkdir(parents=True);c,r,x=s.load();records=[];nrows=0;maxp=0.;maxloss=0.
 for protocol in c['protocols']:
  folder=RUN/protocol;deadline=time.monotonic()+1800
  while not (folder/'complete.json').exists():
   if time.monotonic()>deadline:raise TimeoutError(protocol)
   time.sleep(2)
  for p,h in v.read(folder/'complete.json')['bindings'].items():assert v.sha(folder/p)==h
  before=v.read(folder/'before_evaluation.json');assert not before['evaluation_generated'] and all('evaluation' not in p for p in before['bindings'])
  for p,h in before['bindings'].items():assert v.sha(folder/p)==h
  roles=pd.read_parquet(s.BASE/(protocol+'_split.parquet')).role;tr=roles.eq('fit').to_numpy();base=joblib.load(s.BASE/protocol/'base.joblib');pfit=manual(base,x.loc[tr]);y=r.loc[tr,'label_index'].to_numpy();maps={};rows=[];bodies=[];offset=0;feature_keys=[]
  for level,values in keyset(x.loc[tr],c['shared_fields']).items():
   z=pd.DataFrame({'key':values,'body':r.loc[tr,'body_group'].to_numpy()}).groupby('key',sort=True).agg(rows=('body','size'),bodies=('body','nunique'));maps[level]={k:i+offset for i,k in enumerate(z.index)};offset+=len(z);rows.extend(z.rows);bodies.extend(z.bodies);feature_keys.extend([[level,k] for k in z.index])
  b={'maps':maps};rows=np.array(rows);bodies=np.array(bodies);fitcodes=codes(b,x.loc[tr],c['shared_fields']);cache={};selection=v.read(folder/'selection.json');scores={};fits=[]
  for role in ['calibration','evaluation']:
   dd=pd.read_parquet(s.BASE/protocol/('base_'+role+'.parquet'));mask=roles.eq(role).to_numpy()
   for view,cols in v.VIEWS.items():
    xx=x.loc[mask].copy();xx[cols]=v.MISSING;d=dd[dd.scenario==view].reset_index(drop=True);pd.testing.assert_frame_equal(d[r.columns],r.loc[mask].reset_index(drop=True));cache[(role,view)]={'basep':d[['p_0','p_1','p_2']].to_numpy(),'codes':codes(b,xx,c['shared_fields']),'rows':d[r.columns],'eligible':d.eligible_stress}
  for strength in c['strengths']:
   name='shared_'+str(strength);bundle=joblib.load(folder/(name+'.joblib'));design=bundle['design'];assert design['maps']==maps and design['feature_keys']==feature_keys;np.testing.assert_array_equal(design['rows'],rows);np.testing.assert_array_equal(design['bodies'],bodies);assert v.sha(ROOT/bundle['base_source'])==bundle['base_sha256'];w=bundle['weights'];prec=strength*rows/tr.sum()/np.sqrt(bodies);z=logits(pfit,fitcodes,w);pp=softmax(z,axis=1);loss=float((logsumexp(z,axis=1)-z[np.arange(len(y)),y]).mean()+.5*np.sum(prec[:,None]*w*w));gap=abs(loss-bundle['optimizer']['objective']);maxloss=max(maxloss,gap);assert gap<1e-10;dz=pp;dz[np.arange(len(y)),y]-=1;dz/=len(y);grad=prec[:,None]*w
   for level,ids in fitcodes.items():
    for k in range(3):grad[:,k]+=np.bincount(ids,weights=dz[:,k],minlength=len(w))
   gradmax=float(abs(grad/np.sqrt(prec)[:,None]).max());assert abs(gradmax-bundle['optimizer']['scaled_gradient'])<1e-9;assert selection['candidates'][name]['eligible']==(bundle['optimizer']['success'] and gradmax<=5e-6)
   cal=pd.read_parquet(folder/(name+'_calibration.parquet'));scores[name]=independent_rank(cal);np.testing.assert_allclose(scores[name],selection['candidates'][name]['rank'],atol=1e-12,rtol=0)
   for role in ['calibration']+(['evaluation'] if name==selection['selected'] else []):
    dd=cal if role=='calibration' else pd.read_parquet(folder/'evaluation.parquet')
    for view in v.VIEWS:
     a=cache[(role,view)];p=softmax(logits(a['basep'],a['codes'],w),axis=1);d=dd[dd.scenario==view].reset_index(drop=True);pd.testing.assert_frame_equal(d[r.columns],a['rows']);np.testing.assert_array_equal(d.eligible_stress,a['eligible']);gap=float(abs(p-d[['p_0','p_1','p_2']].to_numpy()).max());maxp=max(maxp,gap);np.testing.assert_allclose(p,d[['p_0','p_1','p_2']],rtol=0,atol=3e-6);np.testing.assert_array_equal(p.argmax(1),d.pred);nrows+=len(p)
   fits.append({'candidate':name,'full_original_rows_objective':loss,'scaled_gradient':gradmax})
  eligible=[k for k,a in selection['candidates'].items() if a['eligible']];best=min(eligible,key=lambda k:scores[k]+[-selection['candidates'][k]['strength']]) if eligible else None;assert best==selection['selected'];records.append({'protocol':protocol,'fits':fits,'selection_verified':True});print({'verified_shared':protocol},flush=True)
 v.save(OUT/'verification.json',{'all_checks_passed':True,'runs':records,'score_rows_replayed':nrows,'max_probability_difference':maxp,'max_original_row_objective_difference':maxloss,'scope':'12 new convex shared corrections; objective/gradient on uncompressed real rows, independent support maps, all calibration and selected evaluation replay. Base cached scores rely transitively on bound v56 independent replay. No blind or deployment claim.'});(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}})
if __name__=='__main__':main()
