"""Guards that matter: class tradeoff rejection, tie rules, missing classes, row mass."""
import copy
import unittest
import numpy as np
import pandas as pd
import torch

from v116_train import EPOCHS, choose, metrics


class SelectionTests(unittest.TestCase):
    def population(self):
        return pd.DataFrame({'truth':[1,1,2,2], 'root':[1,2,3,4]})

    def records(self, pred):
        score=metrics(self.population(), np.asarray(pred))
        return [{'epoch':e,'tasks':{'K':copy.deepcopy(score),'U':copy.deepcopy(score)}} for e in EPOCHS]

    def test_better_S_cannot_spend_M(self):
        records=self.records([1,1,1,1])
        changed=metrics(self.population(),np.array([2,1,2,2]))
        records[0]['tasks']={'K':changed,'U':changed}
        out=choose(records)
        self.assertNotIn(1,out['eligible_epochs'])
        self.assertEqual(out['selected_epoch'],2)

    def test_both_tasks_must_pass(self):
        records=self.records([1,1,2,1])
        records[0]['tasks']['K']=metrics(self.population(),np.array([1,1,2,2]))
        records[0]['tasks']['U']=metrics(self.population(),np.array([2,1,2,2]))
        self.assertNotIn(1,choose(records)['eligible_epochs'])

    def test_ties_choose_earliest(self):
        self.assertEqual(choose(self.records([1,1,2,2]))['selected_epoch'],1)

    def test_missing_class_cannot_pass(self):
        records=self.records([1,1,2,2])
        records[0]['tasks']['U']['class']['suspicious']['support']=0
        with self.assertRaises(AssertionError):choose(records)

    def test_original_row_frequency_loss_and_gradient(self):
        logits=torch.tensor([[.2,.5,-.3],[.1,-.2,.7]],requires_grad=True)
        counts=torch.tensor([[0.,3.,1.],[0.,0.,2.]])
        weighted=-(torch.log_softmax(logits,-1)*counts).sum()/6
        expanded=torch.nn.functional.cross_entropy(logits[[0,0,0,0,1,1]],torch.tensor([1,1,1,2,2,2]))
        self.assertTrue(torch.allclose(weighted,expanded))
        g1=torch.autograd.grad(weighted,logits,retain_graph=True)[0]
        g2=torch.autograd.grad(expanded,logits)[0]
        self.assertTrue(torch.allclose(g1,g2))


if __name__=='__main__':unittest.main()
