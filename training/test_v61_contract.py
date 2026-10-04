"""Adversarial input tests: label-free context, split isolation, preserved relations."""
import unittest
import numpy as np
from prepare_v61 import normalize, contexts
from v61_common import FIELDS, MISSING, fit_vocab, encode_facts, metrics


def facts(port='443'):
    f=dict.fromkeys(FIELDS,MISSING)
    f.update(action='deny',outcome='blocked',transport_protocol='tcp',src_role='inside',dst_role='outside',src_port_fixed='123',dst_port_fixed=port)
    return f


def model_neighbors(f,n,r):
    return [[(tuple(f[j]),tuple(r[i,k])) for k,j in enumerate(ns) if j>=0] for i,ns in enumerate(n)]


class Contract(unittest.TestCase):
    def test_identity_date_removed_actual_port_retained(self):
        raw='Oct 1 USER-0001-0001 Deny tcp src inside:10.0.0.1/123 dst outside:10.0.0.2/443 by ORG-1738-group "ACL-45" [0x12, 0x34]'
        text,_=normalize(raw,facts())
        renamed=raw.replace('Oct 1','Dec 4').replace('USER-0001-0001','USER-9988-7766').replace('10.0.0.1','10.7.4.5').replace('10.0.0.2','10.7.8.9').replace('ORG-1738','ORG-4534')
        self.assertEqual(text,normalize(renamed,facts())[0]);self.assertIn('/443',text)
        self.assertNotEqual(text,normalize(raw.replace('/443','/22'),facts('22'))[0])

    def fixture(self):
        ff=np.array([list(facts(p).values()) for p in ['443','443','22','22']])
        meta=[dict(namespace='device',source='src',destination=d,source_port='123',destination_port=p) for d,p in [('d1','443'),('d2','443'),('d3','22'),('d4','22')]]
        return ff,meta,['fit','fit','fit','selection']

    def test_equal_facts_different_destination_not_erased(self):
        ff,mm,rr=self.fixture();s,n,r,_=contexts(ff[:2],mm[:2],rr[:2])
        self.assertTrue((s[:,0]>0).all());self.assertTrue((n[:,0]>=0).all())

    def test_symbol_renaming_and_no_label_access(self):
        ff,mm,rr=self.fixture();a=contexts(ff,mm,rr)
        renamed=[{k:'rename-'+v for k,v in m.items()}|{'label':99} for m in mm]
        b=contexts(ff,renamed,rr)
        np.testing.assert_allclose(a[0],b[0]);self.assertEqual(model_neighbors(ff,a[1],a[2]),model_neighbors(ff,b[1],b[2]))

    def test_duplicates_do_not_become_traffic_rate(self):
        ff,mm,rr=self.fixture();a=contexts(ff,mm,rr)
        fx=np.concatenate([ff,ff[:1]]);b=contexts(fx,mm+[mm[0]],rr+['fit'])
        np.testing.assert_allclose(a[0],b[0][:-1]);self.assertEqual(model_neighbors(ff,a[1],a[2]),model_neighbors(fx,b[1],b[2])[:-1])

    def test_roles_do_not_share_context(self):
        ff,mm,rr=self.fixture();a=contexts(ff,mm,rr);b=contexts(ff[:3],mm[:3],rr[:3])
        np.testing.assert_allclose(a[0][:3],b[0]);self.assertTrue((a[1][3]==-1).all())

    def test_unknown_is_distinct_from_missing(self):
        ff=np.array([['x',MISSING],['y',MISSING]])
        v=fit_vocab(ff);e=encode_facts(np.array([['z',MISSING]]),v)
        self.assertEqual(e.tolist(),[[1,0]])

    def test_benign_predictions_remain_ASA_false_negatives(self):
        out=metrics([1,2],[[1,0,0],[0,0,1]])
        self.assertEqual(out['macro_f1_M_S'],.5);self.assertEqual(out['threat_to_benign'],1)


if __name__=='__main__':unittest.main()
