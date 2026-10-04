"""Independent full-row objective and saved-score replay for support pooling."""
import json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax,logsumexp
from sklearn.model_selection import StratifiedGroupKFold
import run_v56 as t
import run_v54 as v
from verify_v55 import manual
ROOT=t.ROOT;RUN=t.RUN;OUT=ROOT/'evidence/2026-09-14/v56_verification'

def independent_keys(x):
 coarse=[];fine=[];exact=[]
 for row in x[v.FIELDS].itertuples(index=False,name=None):
  a=dict(zip(v.FIELDS,row));basic=[a['transport_protocol'],a['src_role'],a['dst_role']];flags=[str(a['src_port_fixed']!=v.MISSING),str(a['dst_port_fixed']!=v.MISSING)]
  coarse.append('|'.join(basic+flags));fine.append('|'.join(basic+[a['icmp_type'],a['icmp_code']]+flags));exact.append(json.dumps(list(row),separators=(',',':')))
 return {'coarse':coarse,'fine':fine,'exact':exact}

def lookup_codes(design,xx):
 return {level:np.array([design['maps'][level].get(k,-1) for k in values]) for level,values in independent_keys(xx).items()}

def logits(base_probability,codes,weights):
 out=np.log(np.maximum(base_probability.astype(np.float64),1e-30))
 for level,ids in codes.items():
  hit=ids>=0;out[hit]+=weights[ids[hit]]
 return out

def independent_rank(d):
 bad=[];rates=[];normal=[];ce=[];errors=0
 for view in ['full','no_src_role','no_dst_role']:
  a=d[(d.scenario==view)&(d.route=='asa')&d.eligible_stress];b=d[(d.scenario==view)&(d.route=='asa_acl')];y=a.label_index.to_numpy();yp=a.pred.to_numpy();bad.append(float((yp==0).mean()));rates.extend([float((yp[y==k]!=k).mean()) for k in [1,2]]);normal.append(float((b.pred!=b.label_index).mean()));p=a[['p_0','p_1','p_2']].to_numpy();ce.append(float(-np.log(np.maximum(p[np.arange(len(a)),y],1e-30)).mean()));errors+=int((yp!=y).sum())
 return [max(bad),max(rates),max(normal),float(np.mean(ce)),errors]

def main():
 assert not OUT.exists();OUT.mkdir(parents=True);c,r,x=t.load();old=pd.read_parquet(t.BASE/'body_split.parquet');fine=independent_keys(x)['fine'];import hashlib
 bucket=np.array([int(hashlib.sha256(('v55-behavior|'+k).encode()).hexdigest()[:8],16)%3 for k in fine]);normal=r.route.eq('asa_acl').to_numpy();runs=[];rows_replayed=0;maxgap=0.;maxobjectivegap=0.
 for protocol in c['protocols']:
  folder=RUN/protocol;deadline=time.monotonic()+1800
  while not (folder/'complete.json').exists():
   if time.monotonic()>deadline:raise TimeoutError(protocol)
   time.sleep(2)
  for p,h in v.read(folder/'complete.json')['bindings'].items():assert v.sha(folder/p)==h
  manifest=pd.read_parquet(RUN/(protocol+'_split.parquet'));ev=old.role.eq('validation').to_numpy(copy=True) if protocol=='body' else bucket==int(protocol[-1]);ev[normal]=old.role.eq('validation').to_numpy()[normal];idx=np.flatnonzero(~ev);rr=r.iloc[idx]
  a,b=next(StratifiedGroupKFold(3,shuffle=True,random_state=20260915).split(rr,rr.label_index,rr.body_group));roles=np.full(len(r),'evaluation',dtype=object);roles[idx[a]]='fit';roles[idx[b]]='calibration';np.testing.assert_array_equal(manifest.role,roles)
  for left,right in [('fit','calibration'),('fit','evaluation'),('calibration','evaluation')]:assert set(r.loc[roles==left,'body_group']).isdisjoint(r.loc[roles==right,'body_group'])
  if protocol!='body':assert set(np.array(fine)[~ev&~normal]).isdisjoint(np.array(fine)[ev&~normal])
  tr=roles=='fit';y=r.loc[tr,'label_index'].to_numpy();base=joblib.load(folder/'base.joblib');assert base['model'].t_==int(tr.sum())*10 and base['model'].alpha==1 and base['model'].random_state==20260914
  for j,field in enumerate(v.FIELDS):assert base['input']['seen'][j]==set(x.loc[tr,field])
  before=v.read(folder/'before_evaluation.json');assert not before['evaluation_generated'];assert all('evaluation' not in p for p in before['bindings'])
  for p,h in before['bindings'].items():assert v.sha(folder/p)==h
  origin=v.read(folder/'base_origin.json')
  if not origin['new_base_fit']:
   assert v.sha(ROOT/origin['reused'])==origin['sha256'];ref=joblib.load(ROOT/origin['reused'])
   for aa,bb in zip(base['model'].coefs_+base['model'].intercepts_,ref['model'].coefs_+ref['model'].intercepts_):np.testing.assert_array_equal(aa,bb)
  fitprob=manual(base,x.loc[tr]);fitkeys=independent_keys(x.loc[tr]);offset=0;counts=[];bodies=[];keylist=[];maps={}
  for level,kk in fitkeys.items():
   z=pd.DataFrame({'key':kk,'body':r.loc[tr,'body_group'].to_numpy()}).groupby('key',sort=True).agg(rows=('body','size'),bodies=('body','nunique'));maps[level]={k:i+offset for i,k in enumerate(z.index)};offset+=len(z);counts.extend(z.rows);bodies.extend(z.bodies);keylist.extend([[level,k] for k in z.index])
  design={'maps':maps,'feature_keys':keylist};counts=np.array(counts);bodies=np.array(bodies);fitcodes=lookup_codes(design,x.loc[tr]);cache={}
  for role in ['calibration','evaluation']:
   mask=roles==role;d=pd.read_parquet(folder/('base_'+role+'.parquet'))
   for view,cols in v.VIEWS.items():
    xx=x.loc[mask].copy();xx[cols]=v.MISSING;p=manual(base,xx);z=d[d.scenario==view];pd.testing.assert_frame_equal(z[r.columns].reset_index(drop=True),r.loc[mask].reset_index(drop=True));np.testing.assert_allclose(p,z[['p_0','p_1','p_2']],atol=3e-6,rtol=0);np.testing.assert_array_equal(p.argmax(1),z.pred);eligible=np.ones(mask.sum(),dtype=bool) if not cols else x.loc[mask,cols].ne(v.MISSING).any(axis=1).to_numpy();np.testing.assert_array_equal(z.eligible_stress,eligible)
    cache[(role,view)]={'p':p,'codes':lookup_codes(design,xx),'rows':r.loc[mask].reset_index(drop=True),'eligible':eligible};rows_replayed+=len(p)
  selected=v.read(folder/'selection.json');checked=[]
  for kind in c['kinds']:
   ranks={}
   for strength in c['strengths']:
    name=kind+'_'+str(strength);bundle=joblib.load(folder/(name+'.joblib'));w=bundle['weights'];bd=bundle['design'];assert bd['maps']==maps and bd['feature_keys']==keylist;np.testing.assert_array_equal(bd['rows'],counts);np.testing.assert_array_equal(bd['bodies'],bodies);assert bundle['kind']==kind and bundle['strength']==strength
    precision=strength*counts/tr.sum()/(np.sqrt(bodies) if kind=='support' else 1);z=logits(fitprob,fitcodes,w);p=softmax(z,axis=1);ce=float((logsumexp(z,axis=1)-z[np.arange(len(y)),y]).mean());loss=ce+.5*float(np.sum(precision[:,None]*w*w));dz=p;dz[np.arange(len(y)),y]-=1;dz/=len(y);gradient=precision[:,None]*w
    for level,ids in fitcodes.items():
     for k in [0,1,2]:gradient[:,k]+=np.bincount(ids,weights=dz[:,k],minlength=len(w))
    gap=abs(loss-bundle['optimizer']['objective_end']);maxobjectivegap=max(maxobjectivegap,gap);assert gap<1e-10;gmax=float(abs(gradient/np.sqrt(precision)[:,None]).max());assert abs(gmax-bundle['optimizer']['max_abs_scaled_gradient'])<1e-9;assert loss<=bundle['optimizer']['objective_start']+1e-12
    cal=pd.read_parquet(folder/(name+'_calibration.parquet'));ranks[name]=independent_rank(cal);np.testing.assert_allclose(ranks[name],selected[kind]['candidates'][name]['rank'],rtol=0,atol=1e-12)
    for role in ['calibration']+(['evaluation'] if selected[kind]['selected']==name else []):
     dd=cal if role=='calibration' else pd.read_parquet(folder/(kind+'_evaluation.parquet'))
     for view in v.VIEWS:
      cached=cache[(role,view)];pp=softmax(logits(cached['p'],cached['codes'],w),axis=1);zz=dd[dd.scenario==view];pd.testing.assert_frame_equal(zz[r.columns].reset_index(drop=True),cached['rows']);np.testing.assert_array_equal(zz.eligible_stress,cached['eligible']);gap=float(abs(pp-zz[['p_0','p_1','p_2']].to_numpy()).max());maxgap=max(maxgap,gap);np.testing.assert_allclose(pp,zz[['p_0','p_1','p_2']],atol=3e-6,rtol=0);np.testing.assert_array_equal(pp.argmax(1),zz.pred);rows_replayed+=len(pp)
    checked.append({'candidate':name,'full_original_rows_objective_checked':len(y),'scaled_gradient':gmax})
   eligible=[k for k,a in selected[kind]['candidates'].items() if a['eligible']];best=min(eligible,key=lambda k:ranks[k]+[-selected[kind]['candidates'][k]['strength']]) if eligible else None;assert best==selected[kind]['selected']
  runs.append({'protocol':protocol,'checked':checked,'selection_verified':True});print(json.dumps({'verified':protocol,'residuals':len(checked)}),flush=True)
 v.save(OUT/'verification.json',{'all_checks_passed':True,'protocols':runs,'score_rows_replayed':rows_replayed,'max_probability_difference':maxgap,'max_original_row_objective_difference':maxobjectivegap,'scope':'All 24 residual objectives replayed on original rows, independent support counts and correction lookup, all calibration and selected evaluation scores, split/model bindings. Does not establish real incident independence, label semantics, posterior calibration or blind transfer.'});(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}})
if __name__=='__main__':main()
