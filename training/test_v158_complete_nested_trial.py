"""Actual plan mutations and input-wide projection, no official classifiers."""
import copy,unittest
import numpy as np
from scipy import sparse
from v158_nested_runtime import PLAN,read,validate
from v158_conditions import conditions

class TrialContract(unittest.TestCase):
    def test_actual_positive_complete_sixty_fit_plan(self):
        p=read(PLAN);self.assertEqual(validate(p,False)['total_new_fits_max'],60)

    def test_actual_cost_role_leak_selection_and_source_mutations(self):
        p=read(PLAN)
        for key,value in [('total_new_fits_max',51),('legacy_N1_fits',0),('base_stage_fits',9),
            ('query_labels_used_in_base_supervision_or_guard',True),('outer_labels_used_for_fit_guard_or_selection',True),
            ('base_initialization','reuse_outer_supervised_H1_H2'),('fusion_member_index_input',True),
            ('initialization_prior','best_outer_error_grid'),('canonical_floor_required',False),
            ('execution_entries',['training/v158_nested_base_train.py']),('automatic_repeat',True)]:
            with self.subTest(field=key):
                q=copy.deepcopy(p);q[key]=value
                with self.assertRaises(ValueError):validate(q,False)
        for field in ['full_network_forward_cap','LBFGS_stage_forward_cap','Armijo_stage_forward_cap','canonical_floor','fit_rows']:
            q=copy.deepcopy(p);q['role_budgets'][4][field]+=1
            with self.subTest(role_field=field),self.assertRaises(ValueError):validate(q,False)
        q=copy.deepcopy(p);q['source_sha256']={'training/v158_conditions.py':'0'*64}
        with self.assertRaises(ValueError):validate(q,True)

    def test_text_projection_facts_and_row_independence(self):
        rows=np.array([0,0,1,1,2]);cols=np.array([0,65791,322,66001,66286]);values=np.array([2.,-3.,1.,.25,.75])
        x=sparse.csr_matrix((values,(rows,cols)),shape=(3,66287));z=conditions(x)
        self.assertEqual(z.shape,(3,527));np.testing.assert_array_equal(z[:,32:],x[:,65792:].toarray())
        np.testing.assert_array_equal(conditions(x[[2,0,1]]),z[[2,0,1]])
        for r in range(3):np.testing.assert_array_equal(conditions(x[r]),z[r:r+1])
        self.assertTrue(np.linalg.norm(z[0,:32])>0);self.assertTrue(np.linalg.norm(z[1,:32])>0)
        np.testing.assert_array_equal(z[2,:32],np.zeros(32))

if __name__=='__main__':unittest.main()
