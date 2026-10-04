"""Fix evidenced Gram rank loss without loosening finite or class protection."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'training/v160_active_margin_direction_v4.py';assert not p.exists()
s=(ROOT/'training/v160_active_margin_direction_v3.py').read_text(encoding='utf-8')
old='basis=np.vstack([m-s,-n]);h=basis@s;matrix=basis@basis.T;matrix=(matrix+matrix.T)*.5\n    variables=len(basis);eig=np.linalg.eigvalsh(matrix);matrix_scale=float(np.max(np.abs(eig))) if eig.size else 0.\n    if eig.min(initial=0)<-CERT_EPS*EPS*variables*matrix_scale:return dict(status=\'non_PSD_gram_unresolved\',direction=np.zeros_like(gm),finite_update_qualified=False)'
new='basis=np.vstack([m-s,-n]);variables=len(basis)\n    # Solve the original tall basis, never squared-condition Gram equations.\n    # Relative rank threshold depends on the <=25 dual variables, not 1M rows.\n    relative_rank_cutoff=EPS*max(1,variables)'
assert old in s;s=s.replace(old,new)
s=s.replace('rhs=-h[free]-matrix[np.ix_(free,bound)]@z[bound]\n            values,_,rank,singular=np.linalg.lstsq(matrix[np.ix_(free,free)],rhs,rcond=None);target[free]=values', 'rhs=-s-basis[bound].T@z[bound]\n            values,_,rank,singular=np.linalg.lstsq(basis[free].T,rhs,rcond=relative_rank_cutoff);target[free]=values')
s=s.replace('gradient=matrix@z+h','residual=s+basis.T@z\n        gradient=basis@residual')
s=s.replace('threshold=CERT_EPS*EPS*variables*np.maximum(np.abs(h)+np.abs(matrix)@np.abs(z),np.finfo(float).tiny)','threshold=CERT_EPS*EPS*variables*np.maximum(np.abs(basis)@(np.abs(s)+np.abs(basis).T@np.abs(z)),np.finfo(float).tiny)')
s=s.replace('condition=condition)', 'condition=condition,relative_rank_cutoff=relative_rank_cutoff,singular_values=singular.tolist() if len(free) else [])')
p.write_text(s,encoding='utf-8')
q=ROOT/'training/v160_active_direction_numeric_qualification_v4.py';assert not q.exists()
s=(ROOT/'training/v160_active_direction_numeric_qualification_v3.py').read_text(encoding='utf-8').replace('v160_active_margin_direction_v3','v160_active_margin_direction_v4').replace('v160_active_direction_numeric_qualification_v3_20261002','v160_active_direction_numeric_qualification_v4_20261002')
s=s.replace("redundant=solve_direction", """small_component=[]
    for width in [3,1060832]:
        for component in [1e-4,1e-8,1e-11]:
            m=np.zeros(width);s=np.zeros(width);a=np.zeros((1,width))
            m[-3:]=[-1,component,0];s[-3:]=[-1,0,2*component];a[0,-3:]=[-1,0,0]
            r=solve_direction(m,s,a);small_component.append(dict(width=width,component=component,result=short(r)))
            assert r['status']=='local_QP_certified_requires_actual_finite_guard',short(r)
            assert np.allclose(r['direction'][-3:],[0,-1,-.5],atol=1e-13,rtol=0)
    redundant=solve_direction""")
s=s.replace('cancellation_residue_not_normalized=True,','cancellation_residue_not_normalized=True,useful_small_orthogonal_component_not_lost_to_Gram_rank_cutoff=True,')
s=s.replace('near_dependent=short(near),primal=', 'near_dependent=short(near),small_useful_component=small_component,primal=')
q.write_text(s,encoding='utf-8')
entry=ROOT/'training/v160_fixed_endpoint_diagnostic_v2.py';assert not entry.exists()
s=(ROOT/'training/v160_fixed_endpoint_diagnostic.py').read_text(encoding='utf-8').replace('v160_active_margin_direction_v3','v160_active_margin_direction_v4').replace("['training/v160_fixed_endpoint_diagnostic.py']", "['training/v160_fixed_endpoint_diagnostic_v2.py']")
entry.write_text(s,encoding='utf-8')
test=ROOT/'training/v160_diagnostic_entry_synthetic_qualification_v2.py';assert not test.exists()
s=(ROOT/'training/v160_diagnostic_entry_synthetic_qualification.py').read_text(encoding='utf-8').replace('import v160_fixed_endpoint_diagnostic as','import v160_fixed_endpoint_diagnostic_v2 as').replace('v160_diagnostic_entry_synthetic_qualification_20261002','v160_diagnostic_entry_synthetic_qualification_v2_20261002').replace("'training/v160_fixed_endpoint_diagnostic.py'","'training/v160_fixed_endpoint_diagnostic_v2.py'").replace('v160_active_margin_direction_v3','v160_active_margin_direction_v4').replace('ctx=dict(fold=0,OOF_rows=rr,deployment_rows=rr)','ctx=dict(fold=0,OOF_rows=rr,deployment_rows=rr,ids=np.arange(3))')
test.write_text(s,encoding='utf-8')
print('Direct basis v4 and new entry/test versions created; zero official calls.')
