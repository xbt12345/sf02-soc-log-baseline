"""Three bounded residual corrections in the same original basis."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'training/v160_active_margin_direction_v5.py';assert not p.exists()
s=(ROOT/'training/v160_active_margin_direction_v4.py').read_text(encoding='utf-8')
old='values,_,rank,singular=np.linalg.lstsq(basis[free].T,rhs,rcond=relative_rank_cutoff);target[free]=values'
new='''values,_,rank,singular=np.linalg.lstsq(basis[free].T,rhs,rcond=relative_rank_cutoff)
            for refinement in range(3):
                remainder=rhs-basis[free].T@values
                correction=np.linalg.lstsq(basis[free].T,remainder,rcond=relative_rank_cutoff)[0]
                values=values+correction
            target[free]=values'''
assert old in s;p.write_text(s.replace(old,new),encoding='utf-8')
q=ROOT/'training/v160_active_direction_numeric_qualification_v5.py';assert not q.exists()
s=(ROOT/'training/v160_active_direction_numeric_qualification_v4.py').read_text(encoding='utf-8').replace('v160_active_margin_direction_v4','v160_active_margin_direction_v5').replace('v160_active_direction_numeric_qualification_v4_20261002','v160_active_direction_numeric_qualification_v5_20261002')
q.write_text(s,encoding='utf-8')
e=ROOT/'training/v160_fixed_endpoint_diagnostic_v3.py';assert not e.exists()
s=(ROOT/'training/v160_fixed_endpoint_diagnostic_v2.py').read_text(encoding='utf-8').replace('v160_active_margin_direction_v4','v160_active_margin_direction_v5').replace('training/v160_fixed_endpoint_diagnostic_v2.py','training/v160_fixed_endpoint_diagnostic_v3.py')
e.write_text(s,encoding='utf-8')
t=ROOT/'training/v160_diagnostic_entry_synthetic_qualification_v3.py';assert not t.exists()
s=(ROOT/'training/v160_diagnostic_entry_synthetic_qualification_v2.py').read_text(encoding='utf-8').replace('v160_fixed_endpoint_diagnostic_v2','v160_fixed_endpoint_diagnostic_v3').replace('v160_active_margin_direction_v4','v160_active_margin_direction_v5').replace('v160_diagnostic_entry_synthetic_qualification_v2_20261002','v160_diagnostic_entry_synthetic_qualification_v3_20261002')
t.write_text(s,encoding='utf-8')
print('Bounded residual-correction sources prepared; no official calls.')
