"""Finite-dimensional LP/QP readout with full-population constraint separation.

Every solver result is independently checked against all constraints. A resource
limit is unresolved, not infeasibility. Farkas checks are numerical certificates
at the registered tolerance, not a statement about other representations.
"""
import time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import expit
import osqp
from v89_common import DEST,TOL,DELTA,ALPHA,save,ledger_start,ledger_end,emit

MAX_CUTS=12000
MAX_ROUNDS=40
SOLVE_SECONDS=90
PROBLEM_SECONDS=900
REFINE_STEPS=25


def lp(c,A,b,name):
    n=ledger_start('linear_solver_call',name);start=time.monotonic()
    result=linprog(c,A_ub=-A,b_ub=-b,bounds=[(None,None)]*len(c),method='highs',
        options={'time_limit':SOLVE_SECONDS,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    ledger_end(n,'completed',solver_status=int(result.status),seconds=time.monotonic()-start)
    return result


def certificate(A,b,name):
    # For D u >= b, lambda>=0, D.T lambda=0, b.T lambda>0 contradicts feasibility.
    eq=np.vstack([A.T,np.ones(len(b))]);rhs=np.r_[np.zeros(A.shape[1]),1.]
    n=ledger_start('linear_solver_call',name+'_farkas');start=time.monotonic()
    res=linprog(-b,A_eq=eq,b_eq=rhs,bounds=(0,None),method='highs',
        options={'time_limit':SOLVE_SECONDS,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    stat={'status':int(res.status),'verified':False,'seconds':time.monotonic()-start}
    if res.success:
        l=res.x;stat.update(min_multiplier=float(l.min()),multiplier_sum=float(l.sum()),
            equality_residual_inf=float(np.max(np.abs(A.T@l))),positive_rhs=float(b@l))
        stat['verified']=bool(l.min()>=-1e-10 and abs(l.sum()-1)<1e-8 and stat['equality_residual_inf']<1e-9 and stat['positive_rhs']>1e-7)
        np.save(DEST/(name+'_farkas_multiplier.npy'),l)
        stat['scope']='Numerical Farkas relation at registered tolerance; no capacity or raw semantic impossibility claim.'
    ledger_end(n,'completed',solver_status=stat['status'],**{k:v for k,v in stat.items() if k!='status'})
    return stat


class Problem:
    def __init__(self,V,z0,P,epsilon,E,truth,weights,name):
        self.V=V;self.z0=z0;self.P=np.asarray(P);self.epsilon=epsilon
        self.E=np.asarray(E);self.truth=np.asarray(truth);self.weights=weights;self.name=name
        self.d=V.shape[1];self.n=self.d*3;self.old=z0.argmax(1)
        self.slack=False;self.slack_cap=None;self.cuts=[];self.cutset=set()

    def initial(self,protect):
        self.cuts=[];self.cutset=set()
        for j,row in enumerate(self.E):
            for other in range(3):
                if other!=self.truth[j]:self.add((1,int(row),int(self.truth[j]),other,j))
        if protect:
            m=self.z0[self.P];g=m[np.arange(len(m)),self.old[self.P],None]-m
            g[np.arange(len(m)),self.old[self.P]]=np.inf
            choose=np.argsort(g.min(1),kind='stable')[:256]
            for row in self.P[choose]:
                for other in range(3):
                    if other!=self.old[row]:self.add((0,int(row),int(self.old[row]),other,-1))

    def add(self,t):
        if t not in self.cutset:self.cutset.add(t);self.cuts.append(t)

    def matrix(self):
        n=self.n+(len(self.E) if self.slack else 0);D=np.zeros((len(self.cuts),n));b=np.zeros(len(D))
        for i,(kind,row,cls,other,j) in enumerate(self.cuts):
            D[i,cls*self.d:(cls+1)*self.d]=self.V[row]
            D[i,other*self.d:(other+1)*self.d]=-self.V[row]
            b[i]=(DELTA if kind else self.epsilon[row])-(self.z0[row,cls]-self.z0[row,other])
            if kind and self.slack:D[i,self.n+j]=1
        if self.slack:
            s=np.zeros((len(self.E),n));s[:,self.n:]=np.eye(len(self.E));D=np.vstack([D,s]);b=np.r_[b,np.zeros(len(self.E))]
        if self.slack_cap is not None:
            c=np.r_[np.zeros(self.n),-self.weights];D=np.vstack([D,c]);b=np.r_[b,-self.slack_cap]
        return D,b

    def scores(self,u,ids=None):
        w=u[:self.n].reshape(3,self.d).T
        if ids is None:return self.z0+self.V@w
        return self.z0[ids]+self.V[ids]@w

    def scan(self,u,protect):
        issues=[];worst=0.;wrong_p=0
        if protect:
            for beg in range(0,len(self.P),16384):
                ids=self.P[beg:beg+16384];z=self.scores(u,ids);cl=self.old[ids]
                diff=self.epsilon[ids,None]-(z[np.arange(len(ids)),cl,None]-z);diff[np.arange(len(ids)),cl]=-np.inf
                worst=max(worst,float(diff.max()));wrong_p+=int((z.argmax(1)!=cl).sum())
                a,b=np.where(diff>TOL)
                if len(a):
                    order=np.argsort(-diff[a,b],kind='stable')[:1024]
                    issues.extend((float(diff[a[j],b[j]]),(0,int(ids[a[j]]),int(cl[a[j]]),int(b[j]),-1)) for j in order)
        z=self.scores(u,self.E);diff=DELTA-(z[np.arange(len(self.E)),self.truth,None]-z)
        if self.slack:diff-=u[self.n:,None]
        diff[np.arange(len(self.E)),self.truth]=-np.inf
        worst=max(worst,float(diff.max()))
        a,b=np.where(diff>TOL)
        issues.extend((float(diff[i,j]),(1,int(self.E[i]),int(self.truth[i]),int(j),int(i))) for i,j in zip(a,b))
        if self.slack:
            worst=max(worst,float(-u[self.n:].min()))
            if self.slack_cap is not None:worst=max(worst,float(self.weights@u[self.n:]-self.slack_cap))
        issues.sort(key=lambda t:-t[0]);added=0
        for violation,t in issues[:1024]:
            if t not in self.cutset:self.add(t);added+=1
        return {'max_full_violation':max(0.,worst),'protected_changed_inputs':wrong_p,'new_cuts':added,'cuts':len(self.cuts)}

    def linear(self,protect,slack=False):
        self.slack=slack;self.slack_cap=None;self.initial(protect);start=time.monotonic();trace=[]
        for round in range(MAX_ROUNDS):
            A,b=self.matrix();c=np.r_[np.zeros(self.n),self.weights] if slack else np.zeros(self.n)
            res=lp(c,A,b,f'{self.name}_'+('slack' if slack else 'P' if protect else 'E')+f'_r{round}')
            item={'round':round,'status':int(res.status),'message':str(res.message),'cuts':len(self.cuts),'seconds':time.monotonic()-start}
            if res.status==2:
                certname=self.name+('_P' if protect else '_E')
                cert=certificate(A,b,certname)
                np.savez_compressed(DEST/(certname+'_certificate_constraints.npz'),A=A,b=b,cuts=np.asarray(self.cuts,dtype=np.int64))
                item['certificate']=cert;trace.append(item)
                return None,{'status':'numerically_infeasible' if cert['verified'] else 'unresolved_infeasibility','trace':trace,'certificate':cert}
            if not res.success:
                trace.append(item);return None,{'status':'unresolved_solver_limit','trace':trace}
            lam=-res.ineqlin.marginals
            item['independent_dual_residual']=float(np.max(np.abs(c-A.T@lam)))
            item['independent_dual_gap']=float(c@res.x-b@lam)
            stat=self.scan(res.x,protect);item.update(stat,objective=float(res.fun));trace.append(item)
            emit(stage='head_linear',name=self.name,protect=protect,slack=slack,**item)
            if stat['max_full_violation']<=TOL and not stat['protected_changed_inputs']:
                if slack and (item['independent_dual_residual']>1e-7 or abs(item['independent_dual_gap'])>1e-7):
                    return None,{'status':'unresolved_slack_optimality','trace':trace}
                np.savez_compressed(DEST/(self.name+('_slack' if slack else '_P' if protect else '_E')+'_linear_solution.npz'),
                    solution=res.x,cuts=np.asarray(self.cuts,dtype=np.int64),dual=lam)
                return res.x,{'status':'verified_feasible','trace':trace,'objective':float(res.fun),'full_check':stat}
            if len(self.cuts)>MAX_CUTS or time.monotonic()-start>PROBLEM_SECONDS or not stat['new_cuts']:
                return None,{'status':'unresolved_resource_or_numeric_limit','trace':trace}
        return None,{'status':'unresolved_cut_round_limit','trace':trace}

    def min_parameter_lp(self,start):
        ncore=len(start);starttime=time.monotonic();trace=[]
        for round in range(MAX_ROUNDS):
            D,b=self.matrix();A=np.pad(D,((0,0),(0,self.n)))
            # L1 distance from the unchanged teacher: t_j >= +/- u_j.
            a=np.zeros((2*self.n,ncore+self.n));i=np.arange(self.n)
            a[i,i]=-1;a[i,ncore+i]=1;a[self.n+i,i]=1;a[self.n+i,ncore+i]=1
            A=np.vstack([A,a]);bb=np.r_[b,np.zeros(len(a))];c=np.r_[np.zeros(ncore),np.ones(self.n)]
            res=lp(c,A,bb,f'{self.name}_min_parameter_r{round}')
            if not res.success:return None,{'status':'unresolved_min_parameter','solver_status':int(res.status),'trace':trace}
            u=res.x[:ncore];stat=self.scan(u,True);lam=-res.ineqlin.marginals
            info={**stat,'objective':float(res.fun),'independent_dual_residual':float(np.max(np.abs(c-A.T@lam))),
                  'independent_dual_gap':float(c@res.x-bb@lam),'round':round};trace.append(info)
            emit(stage='head_min_parameter',name=self.name,**info)
            if stat['max_full_violation']<=TOL and not stat['protected_changed_inputs']:
                if info['independent_dual_residual']>1e-7 or abs(info['independent_dual_gap'])>1e-5*max(1.,res.fun):
                    return None,{'status':'unresolved_min_parameter_optimality','trace':trace}
                np.savez_compressed(DEST/(self.name+'_min_parameter_solution.npz'),solution=res.x,dual=lam,cuts=np.asarray(self.cuts,dtype=np.int64))
                return u,{'status':'verified_solution','norm':'L1 in fixed invertibly RMS-conditioned coordinates','trace':trace}
            if len(self.cuts)>MAX_CUTS or time.monotonic()-starttime>PROBLEM_SECONDS or not stat['new_cuts']:
                return None,{'status':'unresolved_min_parameter_limit','trace':trace}
        return None,{'status':'unresolved_min_parameter_round_limit','trace':trace}

    def qp(self,H,g,center,protect,label):
        start=time.monotonic();trace=[];q=g-H@center;warm=center
        for round in range(MAX_ROUNDS):
            D,b=self.matrix();solver=osqp.OSQP();n=ledger_start('quadratic_solver_call',f'{self.name}_{label}_r{round}')
            solver.setup(P=sparse.csc_matrix(np.triu(H)),q=q,A=sparse.csc_matrix(D),l=b,u=np.full(len(b),np.inf),
                verbose=False,eps_abs=1e-9,eps_rel=1e-9,max_iter=100000,time_limit=SOLVE_SECONDS,polishing=True)
            solver.warm_start(x=warm);res=solver.solve(raise_error=False)
            info={'solver_status':str(res.info.status),'primal_residual':float(res.info.prim_res),
                'dual_residual':float(res.info.dual_res),'iterations':int(res.info.iter),'seconds':float(res.info.run_time)}
            ledger_end(n,'completed',**info)
            if res.info.status_val not in [1,2]:return None,{'status':'unresolved_qp','trace':trace+[info]}
            stat=self.scan(res.x,protect);item={**info,**stat};trace.append(item)
            emit(stage='head_qp',name=self.name,label=label,round=round,**item)
            if stat['max_full_violation']<=TOL and not stat['protected_changed_inputs']:
                # A solver's "inaccurate" result is not accepted on primal feasibility alone.
                primal=float(np.maximum(b-D@res.x,0).max());dual=float(np.max(np.abs(H@res.x+q+D.T@res.y)))
                comp=float(np.max(np.abs(res.y*(D@res.x-b))))
                verified=primal<1e-7 and dual<1e-6 and comp<1e-5
                if not verified:return None,{'status':'unresolved_kkt','trace':trace,'independent_primal':primal,'independent_dual':dual,'complementarity':comp}
                return res.x,{'status':'verified_solution','trace':trace,'independent_primal':primal,'independent_dual':dual,'complementarity':comp}
            if len(self.cuts)>MAX_CUTS or time.monotonic()-start>PROBLEM_SECONDS or not stat['new_cuts']:
                return None,{'status':'unresolved_qp_limit','trace':trace}
            warm=res.x
        return None,{'status':'unresolved_qp_round_limit','trace':trace}

    def objective(self,u,used,cc,hessian=False):
        v=self.V[used];z=self.scores(u,used);total=cc.sum(1);den=total.sum();prob=expit(z)
        loss=float((total@np.logaddexp(0,z).sum(1)-(cc*z).sum())/den+ALPHA/2*(u[:self.n]@u[:self.n]))
        g=np.zeros(len(u));g[:self.n]=(v.T@((prob*total[:,None]-cc)/den)).T.ravel()+ALPHA*u[:self.n]
        if not hessian:return loss,g
        H=np.eye(len(u))*1e-10
        for cls in range(3):
            weight=total*prob[:,cls]*(1-prob[:,cls])/den
            a=v.T@(v*weight[:,None])+ALPHA*np.eye(self.d);sl=slice(cls*self.d,(cls+1)*self.d);H[sl,sl]=a
        return loss,g,H

    def train(self,start,used,cc):
        u=start.copy();nvar=len(u)
        closest,rec=self.min_parameter_lp(start)
        if closest is None:return None,{'status':'aborted_min_parameter','reason':rec}
        u=closest;history=[];stall=0;converged=False
        for step in range(REFINE_STEPS):
            f,g,H=self.objective(u,used,cc,True);proposal,record=self.qp(H,g,u,True,f'risk{step}')
            if proposal is None:
                return u,{'status':'stopped_verified_feasible_risk_solver_limit','min_parameter':rec,'history':history,'reason':record,'converged':False}
            direction=proposal-u;decrement=float(g@direction);fraction=1.;found=False
            for _ in range(30):
                candidate=u+fraction*direction;nf,_=self.objective(candidate,used,cc)
                if nf<=f+1e-4*fraction*decrement+1e-12:found=True;break
                fraction/=2
            if not found:return u,{'status':'stopped_risk_line_search','min_parameter':rec,'history':history,'converged':False}
            u=candidate;stat=self.scan(u,True);assert stat['max_full_violation']<=TOL and not stat['protected_changed_inputs']
            history.append({'step':step,'objective_before':f,'objective_after':nf,'directional_decrement':decrement,'fraction':fraction,'full_check':stat})
            emit(stage='head_risk',name=self.name,**history[-1])
            stall=stall+1 if f-nf<1e-10 else 0
            if abs(decrement)<1e-8 or stall>=3:converged=True;break
        return u,{'status':'completed','min_parameter':rec,'history':history,'converged':converged,'steps':len(history)}


def self_check():
    # True infeasibility and a feasible protected correction, including explicit competitor classes.
    v=np.array([[1.,0.,1.],[0.,1.,1.]]);z=np.array([[0.,.2,0.],[0.,.2,0.]])
    p=Problem(v,z,np.array([0]),np.array([.001,.001]),np.array([1]),np.array([2]),np.array([1.]),'synthetic')
    p.initial(True);D,b=p.matrix();u=np.zeros(p.n);u[2*p.d+1]=.5
    assert np.all(D@u>=b) and p.scan(u,True)['max_full_violation']==0
    z2=np.array([[0.,.2,0.]]);p2=Problem(np.ones((1,1)),z2,[0],np.array([.001]),[0],[2],np.array([1.]),'conflict')
    p2.initial(True);D,b=p2.matrix();r=linprog(np.zeros(3),A_ub=-D,b_ub=-b,bounds=(None,None),method='highs')
    assert r.status==2
    p3=Problem(v,z,[],np.zeros(2),[1],[2],np.array([1.]),'gradient')
    u=np.linspace(-.2,.1,p3.n);cc=np.array([[0,2,0],[0,0,3]]);f,g,H=p3.objective(u,np.arange(2),cc,True)
    d=np.arange(len(u),dtype=float);d/=np.linalg.norm(d);e=1e-5
    fd=(p3.objective(u+e*d,np.arange(2),cc)[0]-p3.objective(u-e*d,np.arange(2),cc)[0])/(2*e)
    assert abs(fd-g@d)<1e-7 and np.linalg.eigvalsh(H).min()>0
    return {'synthetic_feasible_protection':True,'synthetic_conflict_detected':True,'objective_gradient_error':float(abs(fd-g@d))}
