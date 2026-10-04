"""Independent layer/support-vector replay; nested split and input checks."""
import argparse,json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v54_method_change_20260914'
sys.path.insert(0,str(RUN));import run_v54 as v

def independent_encode(bundle,x):
 inp=bundle['input'];sizes=[len(t) for t in inp['encoder'].categories_];offset=np.r_[0,np.cumsum(sizes)]
 rows=[];cols=[]
 for j,k in enumerate(v.FIELDS):
  dictionary={str(s):i+offset[j] for i,s in enumerate(inp['encoder'].categories_[j])}
  for i,value in enumerate(x[k]):
   value=value if value in inp['seen'][j] or value==v.MISSING else v.UNKNOWN
   rows.append(i);cols.append(dictionary[value])
 a=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(x),int(offset[-1])))
 assert (a!=v.encode(inp,x)).nnz==0
 return a
def kernel_manual(m,a):
 dual=m.dual_coef_.toarray();edges=np.r_[0,np.cumsum(m.n_support_)];out=[]
 for start in range(0,a.shape[0],256):
  xx=a[start:start+256];k=(xx@m.support_vectors_.T).toarray()
  if m.kernel=='rbf':k=np.exp(-m._gamma*(2*len(v.FIELDS)-2*k))
  elif m.kernel!='linear':raise ValueError(m.kernel)
  vote=np.zeros((len(k),3));confidence=np.zeros((len(k),3));pair=0
  for i in range(3):
   for j in range(i+1,3):
    score=k[:,edges[i]:edges[i+1]]@dual[j-1,edges[i]:edges[i+1]]+k[:,edges[j]:edges[j+1]]@dual[i,edges[j]:edges[j+1]]+m.intercept_[pair]
    vote[:,i]+=score>=0;vote[:,j]+=score<0;confidence[:,i]+=score;confidence[:,j]-=score;pair+=1
  out.append(vote+confidence/(3*(abs(confidence)+1)))
 return np.concatenate(out)
def neural_manual(m,a):
 z=a.astype(np.float32)
 for i,(coef,intercept) in enumerate(zip(m.coefs_,m.intercepts_)):
  z=z@coef+intercept
  if i<len(m.coefs_)-1:z=np.maximum(z,0)
 return softmax(z,axis=1)
def manual(bundle,x):
 codes,_=pd.factorize(v.frame_keys(x),sort=False);first=np.unique(codes,return_index=True)[1]
 a=independent_encode(bundle,x.iloc[first])
 if bundle['family']=='kernel':q=kernel_manual(bundle['model'],a)
 else:q=np.mean([neural_manual(m,a) for m in bundle['models']],axis=0)
 return q[codes]
def loss_floor(x,y):
 k=v.frame_keys(x);c=pd.crosstab(k,y).reindex(columns=[0,1,2],fill_value=0).to_numpy();n=c.sum(1)
 q=c/n[:,None];parts=np.zeros_like(q);nz=c>0;parts[nz]=c[nz]*np.log(q[nz])
 return {'rows':len(y),'observation_keys':len(c),'minimum_empirical_errors':int((n-c.max(1)).sum()),'minimum_empirical_nll':float(-parts.sum()/len(y))}
def main(family):
 out=ROOT/'evidence/2026-09-14'/('v54_verification_'+family);assert not out.exists();out.mkdir(parents=True)
 cfg=v.read(RUN/'configuration.json');r=pd.read_parquet(RUN/'rows.parquet');x=pd.read_parquet(RUN/'observations.parquet');y=r.label_index.to_numpy()
 for p,h in cfg['local_bindings'].items():assert v.sha(RUN/p)==h,p
 for p,h in cfg['source_bindings'].items():assert v.sha(ROOT/p)==h,p
 all_reports=[];rows_checked=0;maxdiff=0.;fit_reports={}
 for protocol in cfg['protocols']:
  folder=RUN/(protocol+'_'+family);receipt=v.read(folder/'evaluation_complete.json')
  for p,h in receipt['bindings'].items():assert v.sha(folder/p)==h,p
  split=pd.read_parquet(RUN/(protocol+'_split.parquet'));outer=v.select(r,protocol);ix=np.flatnonzero(outer);rr=r.iloc[ix].reset_index(drop=True)
  a,b=next(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20260914).split(rr,rr.label_index,rr.body_group))
  np.testing.assert_array_equal(split[split.role=='inner_fit'].row_position,r.iloc[ix[a]].row_position)
  np.testing.assert_array_equal(split[split.role=='inner_validation'].row_position,r.iloc[ix[b]].row_position)
  fit=split.role=='inner_fit';val=split.role=='inner_validation'
  assert set(r.loc[fit,'body_group']).isdisjoint(r.loc[val,'body_group']) and set(r.loc[outer,'body_group']).isdisjoint(r.loc[~outer,'body_group'])
  if protocol=='pressure':assert set(r.loc[outer,'union_group']).isdisjoint(r.loc[~outer,'union_group'])
  seen=set(v.frame_keys(x.loc[fit]));unseen=np.array([k not in seen for k in v.frame_keys(x.loc[val])]);rankings={}
  # Reconstruct model selection from saved inner predictions, with no outer labels.
  for path in folder.glob('inner_*_decisions.parquet'):
   name=path.name[len('inner_'):-len('_decisions.parquet')]
   d=pd.read_parquet(path);np.testing.assert_array_equal(d.row_position,r.loc[val,'row_position']);np.testing.assert_array_equal(d.label_index,y[val]);np.testing.assert_array_equal(d.unseen_fit_key,unseen)
   pred=d.pred.to_numpy();full=v.metrics(y[val],pred);u=v.metrics(y[val][unseen],pred[unseen]);worst=min(full['recall_B_M_S'][1:]+u['recall_B_M_S'][1:])
   rankings[name]=[full['ASA_to_normal'],-worst,-full['macro_f1_M_S'],full['class_errors_B_M_S'][0],full['errors']]
   ib=joblib.load(folder/('inner_'+name+'.joblib'))
   for j,k in enumerate(v.FIELDS):assert ib['input']['seen'][j]==set(x.loc[fit,k])
   # Learned inner outputs are also replayed independently on the original inner roles.
   q=manual(ib,x.loc[val]);np.testing.assert_array_equal(q.argmax(1),pred)
   np.testing.assert_array_equal(np.array(ib['model'].classes_ if family=='kernel' else ib['models'][0].classes_),[0,1,2])
   rows_checked+=len(q)
  order=list(cfg['kernel_candidates']) if family=='kernel' else [str(z) for z in cfg['neural_epochs']]
  picked=min(order,key=lambda n:rankings[n]);selection=v.read(folder/'selection.json')
  assert selection['ranking']==rankings
  assert (picked==selection['selected']) if family=='kernel' else (int(picked)==selection['selected_epochs'])
  if family=='kernel':
   for title,mask in [('inner',fit),('outer',outer)]:
    d=pd.read_parquet(folder/(title+'_compression.parquet'));raw=pd.DataFrame({'input_key':v.frame_keys(x.loc[mask]),'label':y[mask]})
    count=raw.groupby(['input_key','label']).size().sort_index();saved=d.set_index(['input_key','label']).weight.sort_index()
    pd.testing.assert_index_equal(count.index,saved.index);np.testing.assert_array_equal(count.to_numpy(),saved.to_numpy())
  all_reports.append({'protocol':protocol,'selected':picked,'selection_recomputed':True,'inner_fit_rows':int(fit.sum()),'inner_validation_rows':int(val.sum()),'inner_unseen_rows':int(unseen.sum())})
  fit_reports[protocol]={'floor':loss_floor(x.loc[outer],y[outer]),'models':{}}
  for path in folder.glob('final_*.joblib'):
   name=path.stem[6:];bundle=joblib.load(path);d=pd.read_parquet(folder/(name+'_evaluation.parquet'))
   for j,k in enumerate(v.FIELDS):assert bundle['input']['seen'][j]==set(x.loc[outer,k])
   if family=='neural':
    mat=independent_encode(bundle,x.loc[outer]);cold=np.flatnonzero(np.asarray(mat.sum(0)).ravel()==0)
    for m in bundle['models']:assert np.all(m.coefs_[0][cold]==0)
   for scenario,cols in cfg['views'].items():
    xx=x.loc[~outer].copy();xx[cols]=v.MISSING;q=manual(bundle,xx)
    dd=d[d.scenario==scenario];np.testing.assert_array_equal(dd.row_position,r.loc[~outer,'row_position']);np.testing.assert_array_equal(dd.label_index,y[~outer])
    ref=dd[['score_0','score_1','score_2']].to_numpy();delta=float(abs(q-ref).max());maxdiff=max(maxdiff,delta)
    np.testing.assert_allclose(q,ref,atol=1e-7 if family=='kernel' else 3e-6,rtol=0);np.testing.assert_array_equal(q.argmax(1),dd.pred)
    expected=np.ones(len(xx),dtype=bool) if not cols else x.loc[~outer,cols].ne(v.MISSING).any(axis=1).to_numpy();np.testing.assert_array_equal(dd.eligible_stress,expected)
    rows_checked+=len(q)
   # Post-fit errors and NLL distinguish remaining fit capacity from generalization.
   q=manual(bundle,x.loc[outer]);report=v.metrics(y[outer],q.argmax(1))
   if family=='neural':report['nll']=float(-np.log(np.maximum(q[np.arange(len(q)),y[outer]],1e-30)).mean())
   else:report['nll']=None
   fit_reports[protocol]['models'][name]=report
   rows_checked+=len(q)
  print(json.dumps({'verified':protocol,'family':family,'score_rows':rows_checked,'max_difference':maxdiff}),flush=True)
 v.save(out/'verification.json',{'all_checks_passed':True,'family':family,'selection':all_reports,'score_rows_independently_replayed':rows_checked,'max_score_difference':maxdiff,'scope':'Exact split, fit vocabulary, input-loss stress, saved coefficients/layers and inner-selection replay; not calibration or transfer validation.'})
 v.save(out/'fit_capacity.json',fit_reports);(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 v.save(out/'receipt.json',{'files':{p.name:v.sha(p) for p in out.iterdir() if p.is_file()}})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--family',required=True,choices=['kernel','neural']);main(p.parse_args().family)
