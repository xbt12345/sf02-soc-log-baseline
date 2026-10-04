"""Small synthetic software integration, never SOC performance validation."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import joblib
from scipy import sparse
import run_v37_train as train
import v37_representation as rep
from v37_contract import CONTRACT

class Execution(unittest.TestCase):
    def test_full_model_artifacts_resume_and_corruption_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared=root/'prepared';prepared.mkdir();out=root/'models';out.mkdir()
            y=np.tile(np.arange(3,dtype=np.uint8),4);roles=np.repeat(np.arange(4,dtype=np.int8),3);n=len(y)
            text=['session success','traffic exploit payload','credential rejected']
            rows=[{'b0':text[c],'b0facts':json.dumps({'src_port':50000+int(c)}),
                   'b1':text[c],'facts':json.dumps({'src_port_category':'tcp|'+str(50000+int(c))})} for c in y]
            pq.write_table(pa.Table.from_pylist(rows),prepared/'prepared.parquet')
            args=SimpleNamespace(prepared=str(prepared),output_dir=str(out))
            meta=(y,np.arange(n),np.array(['synthetic']*n,object),np.zeros(n,bool),np.ones(n,bool),np.zeros(n,bool),np.zeros(n,bool))
            proto=pa.table({'known_dev':roles});summary={'prepared_sha256':'fixture','groups_sha256':'fixture','version':rep.VERSION}
            for view in CONTRACT['views']:
                with contextlib.redirect_stdout(io.StringIO()):train.run_one(args,'known_dev',view,meta,proto,summary)
                folder=out/('known_dev_'+view);bundle=joblib.load(folder/'model.joblib');t=pq.read_table(folder/'evaluation.parquet')
                tc,fc,_=train.columns(view)
                x=sparse.hstack([bundle['tfidf'].transform([r[tc] for r in rows[-3:]]),bundle['facts'].transform([r[fc] for r in rows[-3:]])],format='csr')
                p=bundle['model'].predict_proba(x)
                np.testing.assert_allclose(p,np.column_stack([t[k].to_numpy() for k in ['p_benign','p_malicious','p_suspicious']]),atol=1e-12)
                report=json.loads((folder/'report.json').read_text())
                self.assertEqual(report['final_fit']['fit_rows'],6);self.assertEqual(report['final_fit']['sum_weights'],6.)
                self.assertFalse(report['training_quality_accepted'])
                old_time=(folder/'model.joblib').stat().st_mtime_ns
                with contextlib.redirect_stdout(io.StringIO()):train.run_one(args,'known_dev',view,meta,proto,summary)
                self.assertEqual(old_time,(folder/'model.joblib').stat().st_mtime_ns)
                (folder/'report.json').write_text('{}')
                with self.assertRaises(AssertionError):train.run_one(args,'known_dev',view,meta,proto,summary)
    def test_contract_has_no_weighting_or_target_threshold_route(self):
        self.assertEqual(CONTRACT['views'],['B2_CONTROL','B2_REPAIRED'])
        self.assertFalse(CONTRACT['repeat_weighting']);self.assertEqual(CONTRACT['threshold_role'],2)
        self.assertTrue(CONTRACT['no_target_calibration'])
    def test_fit_only_encoder_ignores_target_word_and_port(self):
        texts=['fit word','targetonly'];facts=['{"src_port_category":"tcp|80"}','{"src_port_category":"tcp|65000"}']
        x,inv,te,fe=train.encode(texts,np.array([0,1]),facts,np.array([0,1]),np.array([True,False]),train.new)
        self.assertNotIn('targetonly',te.names());self.assertFalse(any('65000' in str(k) for k in fe.names()))
    def test_coarse_relation_label_mixture_is_not_an_identical_input_conflict(self):
        result=train.review_flags(np.zeros(2,bool),np.ones(2,bool),np.zeros(2,bool),
             np.array([[100,20,30],[100,20,30]]),np.array([[3,0,0],[3,0,1]]))
        self.assertEqual(result.tolist(),[False,True])
if __name__=='__main__':unittest.main()
