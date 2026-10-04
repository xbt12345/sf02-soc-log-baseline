"""Separate sign-certificate rejection from absence of class descent."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'training/v160_active_margin_direction_v3.py';assert not p.exists()
s=(ROOT/'training/v160_active_margin_direction_v2.py').read_text(encoding='utf-8')
old="status='local_QP_certified_requires_actual_finite_guard' if passed and cert['resolved_common_descent'] and independent['eligible_for_finite_trial_only'] else ('no_resolved_safe_common_descent' if passed else 'local_QP_certificate_failed_stop')"
new="status=('local_QP_certificate_failed_stop' if not passed else ('no_resolved_safe_common_descent' if not cert['resolved_common_descent'] else ('independent_direction_sign_failed_stop' if not independent['eligible_for_finite_trial_only'] else 'local_QP_certified_requires_actual_finite_guard')))"
assert old in s;p.write_text(s.replace(old,new),encoding='utf-8')
q=ROOT/'training/v160_active_direction_numeric_qualification_v3.py';assert not q.exists()
s=(ROOT/'training/v160_active_direction_numeric_qualification_v2.py').read_text(encoding='utf-8').replace('v160_active_margin_direction_v2','v160_active_margin_direction_v3').replace('v160_active_direction_numeric_qualification_v2_20261002','v160_active_direction_numeric_qualification_v3_20261002')
s=s.replace("'local_QP_certificate_failed_stop','direction_reconstruction_unresolved_stop','no_resolved_safe_common_descent'", "'local_QP_certificate_failed_stop','direction_reconstruction_unresolved_stop','no_resolved_safe_common_descent','independent_direction_sign_failed_stop'")
q.write_text(s,encoding='utf-8')
print('New stop taxonomy prepared, no official calls.')
