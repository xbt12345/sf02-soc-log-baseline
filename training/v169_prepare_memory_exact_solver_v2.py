"""Remove one 66xP matrix copy, preserving exact SVD inputs and units."""
from experiment_review import ROOT

def main():
    source=ROOT/'training/v169_working_joint_restoration.py';target=ROOT/'training/v169_working_joint_restoration_v2.py';assert not target.exists();s=source.read_text(encoding='utf-8')
    old="    matrix=np.vstack([a,-m,-s]);rhs=np.r_[b-c,dot(m,u)[0],dot(s,u)[0]]\n    scales=np.max(np.abs(matrix),axis=1)\n    if np.any(scales==0):return dict(status='zero_actual_constraint_normal_stop',finite_step_authority=False)\n    scaled=matrix/scales[:,None];lower=rhs/scales"
    new="""    rhs=np.r_[b-c,dot(m,u)[0],dot(s,u)[0]]
    scales=np.r_[np.max(np.abs(a),axis=1),np.max(np.abs(m)),np.max(np.abs(s))]
    if np.any(scales==0):return dict(status='zero_actual_constraint_normal_stop',finite_step_authority=False)
    scaled=np.empty((len(a)+2,len(u)),np.float64)
    np.divide(a,scales[:-2,None],out=scaled[:-2])
    np.divide(-m,scales[-2],out=scaled[-2]);np.divide(-s,scales[-1],out=scaled[-1])
    lower=rhs/scales"""
    assert old in s;s=s.replace(old,new).replace('cutoff=EPS*len(matrix)','cutoff=EPS*(len(a)+2)').replace('64*EPS*len(matrix)','64*EPS*(len(a)+2)')
    old='    for i,row in enumerate(matrix):'
    new="    for i in range(len(a)+2):\n        row=a[i] if i<len(a) else (-m if i==len(a) else -s)"
    assert old in s;s=s.replace(old,new);assert 'matrix' not in s;compile(s,str(target),'exec');target.write_text(s,encoding='utf-8')
    source=ROOT/'training/v169_dynamic_trial_restoration.py';target=ROOT/'training/v169_dynamic_trial_restoration_v2.py';assert not target.exists();s=source.read_text(encoding='utf-8').replace('import v169_working_joint_restoration as solver','import v169_working_joint_restoration_v2 as solver');target.write_text(s,encoding='utf-8');print('Prepared same arithmetic lower-memory solver v2')

if __name__=='__main__':main()
