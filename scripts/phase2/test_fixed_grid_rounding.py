import unittest
import torch
from fixed_grid_rounding import coordinate_round


class RoundingTests(unittest.TestCase):
    def test_correlated_inputs_and_isolation(self):
        torch.manual_seed(42)
        w=torch.randn(8,12);s=w.abs().amax(1,keepdim=True)/7
        x=torch.randn(80,12);x[:,1]=x[:,0]*.9+x[:,1]*.1
        before=w.clone();sb=s.clone()
        trials=coordinate_round(w,s,x[:48],x[48:],(1,4,16))
        self.assertTrue(torch.equal(w,before));self.assertTrue(torch.equal(s,sb))
        floor=(w/s).floor().clamp(-8,7).to(torch.int8)
        ceil=(w/s).ceil().clamp(-8,7).to(torch.int8)
        for trial in trials:
            q=trial['codes'];self.assertTrue(((q==floor)|(q==ceil)).all())
            self.assertGreaterEqual(int(q.min()),-8);self.assertLessEqual(int(q.max()),7)
        for a,b in zip(trials,trials[1:]):self.assertLessEqual(b['fit_mse'],a['fit_mse']+1e-7)

    def test_zero_inputs(self):
        w=torch.tensor([[0.,.5,-.5]])
        trials=coordinate_round(w,torch.ones(1,1),torch.zeros(4,3),torch.zeros(2,3),(2,))
        self.assertEqual(trials[-1]['fit_mse'],0)
        self.assertTrue(torch.equal(trials[0]['codes'],trials[-1]['codes']))

    def test_non_weight_reference(self):
        torch.manual_seed(7)
        w=torch.randn(6,10);s=w.abs().amax(1,keepdim=True)/7
        x=torch.randn(64,10);y=x@w.T+torch.randn(64,6)*.1
        trials=coordinate_round(w,s,x[:48],x[48:],(1,4,16),target_fit=y[:48],target_heldout=y[48:])
        for t in trials:
            actual=float(((t['codes'].float()*s)@x[:48].T-y[:48].T).square().mean())
            self.assertAlmostEqual(actual,t['fit_mse'],places=6)
        for a,b in zip(trials,trials[1:]):self.assertLessEqual(b['fit_mse'],a['fit_mse']+1e-6)

    def test_neighbor_compensates_fixed_grid_saturation(self):
        w=torch.tensor([[9.,0.]])
        s=torch.ones(1,1)
        x=torch.tensor([[1.,1.],[2.,2.],[-1.,-1.]])
        parent=torch.tensor([[7,0]],dtype=torch.int8);before=parent.clone()
        ordinary=coordinate_round(w,s,x,x,(4,))
        trials=coordinate_round(w,s,x,x,(1,4),initial_codes=parent,neighbor_search=True)
        self.assertGreater(ordinary[-1]['fit_mse'],0.)
        self.assertEqual(trials[-1]['fit_mse'],0.)
        self.assertTrue(torch.equal(trials[0]['codes'],before))
        self.assertTrue(torch.equal(parent,before))
        self.assertTrue(torch.equal(trials[-1]['codes'],torch.tensor([[7,2]],dtype=torch.int8)))
        for a,b in zip(trials,trials[1:]):self.assertLessEqual(b['fit_mse'],a['fit_mse'])


if __name__=='__main__':unittest.main()
