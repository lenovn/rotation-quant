"""Gradient-free two-choice weight rounding under fixed per-row W4 scales."""
import torch


@torch.no_grad()
def coordinate_round(weight, scale, fit, heldout, milestones=(32,128,512), progress=None,
                     target_fit=None, target_heldout=None, initial_codes=None, neighbor_search=False):
    """Minimize ||(dequant(codes)-weight) X|| on the fixed INT4 grid.

    Inputs are the actual frozen quantizer outputs, not unquantized activations.
    Every accepted coordinate updates all affected gradients analytically.
    Default floor/ceil search is retained. Neighbor search allows successive
    +/-1 code changes, starting from an explicit frozen parent if supplied.
    """
    w=weight.float(); s=scale.float().reshape(-1,1)
    codes=(w/s).round().clamp(-8,7).to(torch.int8)
    if initial_codes is not None:
        assert initial_codes.dtype==torch.int8 and initial_codes.shape==w.shape
        assert ((initial_codes>=-8)&(initial_codes<=7)).all()
        codes=initial_codes.to(w.device).clone()
    low=(w/s).floor().clamp(-8,7).to(torch.int8)
    high=(w/s).ceil().clamp(-8,7).to(torch.int8)
    def dequant(c):return (c.float()*s).to(weight.dtype).float()
    current=dequant(codes)
    h=fit.float().T@fit.float()/fit.shape[0]
    if (target_fit is None)!=(target_heldout is None):raise ValueError('Both reference outputs required')
    if target_fit is None:
        gradient=(current-w)@h
    else:
        assert target_fit.shape==(fit.shape[0],w.shape[0])
        assert target_heldout.shape==(heldout.shape[0],w.shape[0])
        gradient=(current@fit.float().T-target_fit.float().T)@fit.float()/fit.shape[0]
    diagonal=h.diag()
    initial=codes.clone()
    trials=[]
    def snapshot(step):
        q=dequant(codes)
        if target_fit is None:
            fm=float(((q-w)@fit.float().T).square().mean())
            hm=float(((q-w)@heldout.float().T).square().mean())
        else:
            fm=float((q@fit.float().T-target_fit.float().T).square().mean())
            hm=float((q@heldout.float().T-target_heldout.float().T).square().mean())
        if not torch.isfinite(torch.tensor([fm,hm])).all():raise RuntimeError('Non-finite rounding objective')
        trials.append(dict(step=step,fit_mse=fm,heldout_mse=hm,
                           changed_codes=int((codes!=initial).sum()),codes=codes.cpu().clone()))
    snapshot(0)
    for step in range(1,max(milestones)+1):
        if neighbor_search:
            lower=(codes-1).clamp(-8,7)
            upper=(codes+1).clamp(-8,7)
            dl=dequant(lower)-current;du=dequant(upper)-current
            cl=2*dl*gradient+dl.square()*diagonal[None,:]
            cu=2*du*gradient+du.square()*diagonal[None,:]
            other=torch.where(cl<cu,lower,upper)
        else:
            other=torch.where(codes==low,high,low)
        delta=dequant(other)-current
        cost=2*delta*gradient+delta.square()*diagonal[None,:]
        gains=cost.clamp(max=0).sum(dim=0)
        j=int(gains.argmin())
        if float(gains[j])>=-1e-15:
            snapshot(step)
            break
        accept=cost[:,j]<0
        change=torch.where(accept,delta[:,j],0)
        codes[:,j]=torch.where(accept,other[:,j],codes[:,j])
        current[:,j]+=change
        gradient+=change[:,None]*h[j:j+1,:]
        if step in milestones:snapshot(step)
        if progress is not None and step%16==0:progress(step)
    return trials
