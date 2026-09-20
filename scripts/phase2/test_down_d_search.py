import io
import unittest
import torch
from down_d_search import direction, fuse_pair, weight_quant, pack, unpack, mlp
from down_codebooks import quantize


class DownDTests(unittest.TestCase):
    def test_pair_equivalence(self):
        for dtype, tolerance in [(torch.float64, 1e-12), (torch.float32, 2e-5)]:
            torch.manual_seed(4)
            h, gate, up, down = [torch.randn(*s, dtype=dtype) for s in [(9,4),(8,4),(8,4),(4,8)]]
            d = torch.tensor([.25,.5,1,2,4,2,1,.5], dtype=dtype)
            old_up, old_down = up.clone(), down.clone()
            u, w = fuse_pair(up, down, d)
            torch.testing.assert_close(mlp(h,gate,u,w),mlp(h,gate,up,down),rtol=tolerance,atol=tolerance)
            self.assertTrue(torch.equal(up,old_up) and torch.equal(down,old_down))
            self.assertNotEqual(u.data_ptr(),up.data_ptr())
            self.assertNotEqual(w.data_ptr(),down.data_ptr())

    def test_identity_and_shape(self):
        u, w = torch.randn(8,4), torch.randn(4,8)
        a,b = fuse_pair(u,w,torch.ones(8))
        self.assertTrue(torch.equal(a,u) and torch.equal(b,w))
        for d in [torch.ones(7),torch.zeros(8),torch.full((8,),float('nan'))]:
            with self.assertRaises(ValueError): fuse_pair(u,w,d)

    def test_direction_center_bounds_nonconstant(self):
        v = direction(torch.logspace(-6,6,8),torch.ones(4,8))
        self.assertGreater(float(v.max()-v.min()),1.)
        for t in [0,.25,.5,.75,1]:
            d=(t*v).exp()
            self.assertTrue((d>=.25-1e-7).all() and (d<=4+1e-7).all())
            self.assertLess(abs(float(d.double().log().mean())),1e-7)
        self.assertTrue(torch.equal(direction(torch.ones(8),torch.ones(4,8)),torch.zeros(8)))

    def test_weight_codes_ties_zero(self):
        w=torch.tensor([[-20.,-8.,-2.5,-1.5,.5,1.5,2.5,20.],[0.]*8])
        q,c,s=weight_quant(w,torch.ones(2,1))
        self.assertEqual(c[0].tolist(),[-8,-8,-2,-2,0,2,2,7])
        q,c,s=weight_quant(w)
        self.assertTrue((s>0).all() and torch.isfinite(q).all())
        self.assertTrue((q[1]==0).all())

    def test_activation_fullrange(self):
        x=torch.tensor([-128.,-127.,-.5,.5,1.5,127.])
        self.assertEqual(quantize(x,127.,'int8').tolist(),[-128.,-127.,0.,0.,2.,127.])
        y=quantize(x,float(x.abs().max()),'int8')
        self.assertTrue(torch.isfinite(y).all())

    def test_pack_reload_isolation(self):
        codes=torch.arange(-8,8,dtype=torch.int8).reshape(2,8)
        payload={'packed':pack(codes),'scale':torch.tensor([[.13],[.27]]),'alpha':2.75}
        memory=io.BytesIO();torch.save(payload,memory);memory.seek(0)
        loaded=torch.load(memory,weights_only=True)
        self.assertTrue(torch.equal(unpack(loaded['packed'],codes.shape),codes))
        self.assertTrue(torch.equal(loaded['scale'],payload['scale']))
        self.assertEqual(loaded['alpha'],2.75)
        u,w=torch.randn(8,4),torch.randn(4,8)
        q1,_,_=weight_quant(fuse_pair(u,w,torch.ones(8))[1])
        fuse_pair(u,w,torch.linspace(.25,4,8))
        q2,_,_=weight_quant(fuse_pair(u,w,torch.ones(8))[1])
        self.assertTrue(torch.equal(q1,q2))

if __name__=='__main__': unittest.main(verbosity=2)
