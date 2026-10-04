"""Software smoke tests: fits synthetic strings; never reports SOC quality."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

import run_v331_train as train
from v331_common import unique_matrix,new_vectorizer,check_complete,load,metrics
from verify_v331_protocol import verify_isolation

ROOT=Path(__file__).resolve().parents[1]


class ProtocolAndInference(unittest.TestCase):
    def test_group_crossing_is_rejected(self):
        with self.assertRaises(AssertionError):
            verify_isolation(np.array([0,0,1]),np.array([0,3,1],dtype=np.int8))

    def test_group_linked_exclusion_is_compatible(self):
        verify_isolation(np.array([0,0,1]),np.array([-1,3,0],dtype=np.int8))

    def test_unique_transform_restores_original_row_frequency(self):
        data={"texts":["allow normal", "deny packet", "failure auth"],"text_ids":np.array([0,0,2,1,0,1])}
        v=new_vectorizer();v.fit(data["texts"])
        pos=np.array([4,1,5,3,2])
        x,inv,_=unique_matrix(data,pos,v)
        direct=v.transform([data["texts"][data["text_ids"][i]] for i in pos])
        self.assertEqual((x[inv]-direct).nnz,0)
        self.assertEqual(x[inv].shape[0],5)


class CompleteTaskSmoke(unittest.TestCase):
    def test_fit_selection_refit_thresholds_replay_and_resume(self):
        # Every string here is artificial; this tests software roles and artifacts.
        parent=ROOT/"artifacts"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="v331_software_smoke_",dir=str(parent)) as dirname:
            root=Path(dirname).resolve()
            self.assertIn(parent.resolve(),root.parents)
            run=root/"prepared";out=root/"outputs";run.mkdir();out.mkdir()
            (run/"model_stress_fixtures.json").write_text("[]",encoding="utf-8")
            n=300;y=np.arange(n,dtype=np.uint8)%3
            roles=np.repeat(np.array([0,1,2,3],dtype=np.int8),[150,45,45,60])
            texts=["ordinary successful","packet exploit","authentication denied"]
            # Evaluation-only word must never enter the refit vocabulary.
            ids=y.astype(np.int32).copy()
            texts.append("ordinary successful outer_sentinel")
            ids[(roles==3)&(y==0)]=3
            data={"texts":texts,"text_ids":ids,"y":y,"g":np.arange(n,dtype=np.int32),
                  "info":np.ones(n,bool),"format":np.array(["same"]*n,dtype=object),
                  "product":np.array(["synthetic"]*n,dtype=object),
                  "repair_route":np.array(["unchanged"]*n,dtype=object),
                  "original_empty":np.zeros(n,bool),"filtered_empty":np.zeros(n,bool),
                  "unknown_format":np.zeros(n,bool),"authentication_result_unknown":np.zeros(n,bool),
                  "no_observable_auth_facts":np.zeros(n,bool),"asa_template":np.array([None]*n,dtype=object)}
            task={"name":"software_smoke","kind":"domain","enabled":True}
            binding={"scope":"synthetic software test, not SOC quality"}
            report=train.run_task(data,roles,task,run,out,binding)
            self.assertTrue(report["model_trained"])
            folder=check_complete(out/"tasks/software_smoke",binding)
            v=joblib.load(folder/"vectorizer.joblib");model=joblib.load(folder/"model.joblib")
            self.assertNotIn("outer_sentinel",v.vocabulary_)
            import pyarrow.parquet as pq
            prediction=pq.read_table(folder/"outer_predictions.parquet")
            pos=prediction["row_position"].to_numpy()
            prob=model.predict_proba(v.transform([texts[ids[i]] for i in pos]))
            self.assertTrue(np.array_equal(prob.argmax(1),prediction["R_pred"].to_numpy()))
            for i,c in enumerate(["benign","malicious","suspicious"]):
                np.testing.assert_allclose(prob[:,i],prediction["R_p_"+c].to_numpy(),rtol=0,atol=1e-7)
            for threshold in report["risk_thresholds"]:
                self.assertLessEqual(threshold["R_calibration"]["class_alert_rates"]["benign"],threshold["budget"])
            from review_v331_round import review_task
            verified,_=review_task(data,roles,folder,task)
            self.assertTrue(verified["all_checks_passed"])
            self.assertEqual(verified["replayed_rows"],105)
            with patch.object(train,"fit_model",side_effect=AssertionError("Completed task must not fit again")):
                repeat=train.run_task(data,roles,task,run,out,binding)
                self.assertEqual(repeat["model_sha256"],report["model_sha256"])
            # Bad completion bytes must never silently become a cache hit.
            with (folder/"model.joblib").open("ab") as f:f.write(b"damaged")
            with self.assertRaises(ValueError):check_complete(out/"tasks/software_smoke",binding)
            with self.assertRaises(ValueError):check_complete(out/"tasks/software_smoke",{"different":"binding"})


if __name__=="__main__":
    unittest.main(verbosity=2)
