"""Fixed ridge factual decoder, never a threat-classifier head."""
import hashlib
import numpy as np
import torch

FIELDS=('action','outcome','transport_protocol','src_role','dst_role','src_port_range','dst_port_range',
        'icmp_message','icmp_unreachable','src_port_fixed','dst_port_fixed','icmp_type','icmp_code')
WIDTHS={'src_port_fixed':17,'dst_port_fixed':17,'icmp_type':9,'icmp_code':9}
MISSING={'src_port_fixed':65536,'dst_port_fixed':65536,'icmp_type':256,'icmp_code':256}
RIDGE=1e-4

def inner_held(root):
    return int(hashlib.sha256(('V150-field-readout:'+str(int(root))).encode()).hexdigest(),16)%5==0

def schema(names):
    entries=[];columns=[]
    for field in FIELDS:
        if field in WIDTHS:
            cols=[names.index(field+':bit'+str(i)) for i in range(WIDTHS[field])];values=None
        else:
            cols=[i for i,n in enumerate(names) if n.startswith(field+'=')]
            values=[names[i].split('=',1)[1] for i in cols]
        assert cols
        start=len(columns);columns.extend(cols)
        entries.append(dict(field=field,start=start,stop=len(columns),values=values))
    assert len(set(columns))==len(columns)
    return columns,entries

def decode(scores,entries):
    result={}
    for e in entries:
        s=scores[:,e['start']:e['stop']];field=e['field']
        if field in WIDTHS:
            result[field]=((s>=.5).astype(np.int64)*(1<<np.arange(s.shape[1]))).sum(1)
        else:
            best=s.argmax(1)
            result[field]=np.array([e['values'][i] if v>=.5 else None for i,v in zip(best,s.max(1))],dtype=object)
    return result

@torch.no_grad()
def solve(x,target,mass,ridge=RIDGE):
    """Original-frequency MSE / total mass + ridge ||standardized slopes||^2."""
    ids=torch.nonzero(mass>0).flatten();xx=x[ids];ww=mass[ids];yy=target[ids];total=ww.sum()
    mean=(xx*ww[:,None]).sum(0)/total
    var=((xx-mean).square()*ww[:,None]).sum(0)/total
    active=var>1e-16
    scale=var[active].sqrt();design=(xx[:,active]-mean[active])/scale
    design=torch.cat([design,torch.ones((len(xx),1),device=x.device,dtype=x.dtype)],1)
    gram=design.T@(design*ww[:,None])/total;rhs=design.T@(yy*ww[:,None])/total
    penalty=torch.eye(len(gram),device=x.device,dtype=x.dtype)*ridge;penalty[-1,-1]=0
    system=gram+penalty;coef=torch.linalg.solve(system,rhs)
    relative_residual=float((system@coef-rhs).norm()/rhs.norm().clamp_min(1e-12))
    assert relative_residual<1e-8 and torch.isfinite(coef).all()
    return dict(mean=mean[active],scale=scale,active=active,coef=coef),dict(
        original_fit_mass=int(total),active_feature_dimensions=int(active.sum()),
        target_dimensions=target.shape[1],normal_equation_relative_residual=relative_residual,
        objective=float(((design@coef-yy).square()*ww[:,None]).sum()/total+ridge*coef[:-1].square().sum()))

@torch.no_grad()
def predict(x,state,chunk=2048):
    outputs=[]
    for begin in range(0,len(x),chunk):
        v=(x[begin:begin+chunk,state['active']]-state['mean'])/state['scale']
        v=torch.cat([v,torch.ones((len(v),1),device=x.device,dtype=x.dtype)],1)
        outputs.append((v@state['coef']).cpu().numpy())
    return np.concatenate(outputs)
