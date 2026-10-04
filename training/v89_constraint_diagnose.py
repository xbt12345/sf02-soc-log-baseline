"""Derive an independent infeasibility witness from verified positive slack duals."""
import json
import numpy as np,pandas as pd
from v89_common import ROOT,OUT,DEST,OLD,read,save,sha


def main():
    z=np.load(OLD/'teacher_scores.npy',mmap_mode='r');r=pd.read_parquet(OUT/'rows.parquet');fid=np.load(ROOT/'artifacts/v79_execution_20260927/row_feature_id.npy');records=[];summary={}
    for name in ['R00','R01','R10','R11','F00']:
        rec=read(DEST/(name+'_fit.json'));p=DEST/(name+'_slack_linear_solution.npz')
        if not p.exists():continue
        a=np.load(p);V=np.load(DEST/(name+'_basis.npy'),mmap_mode='r');orig=np.load(DEST/'F00_group_keys.npy')[:,0] if name=='F00' else np.arange(len(z));z0=np.asarray(z[orig]);old=z0.argmax(1);gid=np.load(DEST/'F00_row_group.npy') if name=='F00' else fid
        margin=z0[np.arange(len(old)),old]-np.max(np.where(np.eye(3,dtype=bool)[old],-np.inf,z0),axis=1);eps=np.minimum(.001,margin/2)+np.minimum(1e-5,margin/4)
        cuts=a['cuts'];lam=a['dual'][:len(cuts)].copy();lam/=lam.sum();D=np.zeros((len(cuts),V.shape[1]*3));b=np.zeros(len(cuts))
        for i,(kind,row,cls,other,j) in enumerate(cuts):
            D[i,cls*V.shape[1]:(cls+1)*V.shape[1]]=V[row];D[i,other*V.shape[1]:(other+1)*V.shape[1]]=-V[row]
            b[i]=(.01 if kind else eps[row])-(z0[row,cls]-z0[row,other])
            if lam[i]>1e-10:
                ix=np.flatnonzero((gid==row)&~r.fold.isin([0,1,2]).to_numpy());cls_counts=np.bincount(r.label_index.iloc[ix],minlength=3)
                records.append({'model':name,'constraint':'error_target' if kind else 'old_correct_protection','input_group':int(row),'true_class':int(cls),'competitor_class':int(other),'normalized_multiplier':float(lam[i]),'original_fit_class_counts':json.dumps(cls_counts.tolist()),'representative_row':int(ix[0]),'route':r.route.iloc[ix[0]]})
        eq=float(np.abs(D.T@lam).max());rhs=float(b@lam);assert eq<1e-9 and rhs>1e-7 and lam.min()>=-1e-10 and abs(lam.sum()-1)<1e-8
        summary[name]={'verified':True,'equality_residual':eq,'positive_rhs':rhs,'witness_nonzero_constraints':int((lam>1e-10).sum()),'via':'Dual of minimum equal-class hinge slack; nonnegativity rows have zero core coefficients and RHS, removed, remaining multipliers normalized. Contradicts a solution with zero E slack and full P.',
            'direct_P_solver_status':rec['P_feasibility']['status'],'scope':'Fixed finite-dimensional basis at numerical tolerance, not all model families or raw information.'}
        np.savez_compressed(DEST/(name+'_dual_derived_witness.npz'),D=D,b=b,multiplier=lam)
    pd.DataFrame(records).to_csv(DEST/'certificate_active_constraints.csv',index=False)
    save(DEST/'dual_derived_certificates.json',{'status':'verified','source_sha256':sha(__file__),'new_classifier_fits':0,'solver_calls':0,'certificates':summary})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
