"""Weight-only output-channel plots for the current parent layer1 down matrix."""
import argparse
import gc
from pathlib import Path
import shutil
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))
import torch
from experiments.phase3.common import build_training_model, rotated_linears, write_json
from experiments.phase3.postprocess import dequant_record
from experiments.phase3.quantization import unpack_int4
from experiments.phase3.run import source_record

TARGET = 'model.layers.1.mlp.down_proj'


def channel_stats(reference, quantized, codes, scale):
    if reference.ndim != 2 or reference.shape != quantized.shape or scale.shape != (reference.shape[0], 1):
        raise ValueError('Expected [output,input] weights and [output,1] original SW')
    w, q = reference.float(), quantized.float()
    absolute = w.abs()
    energy = w.square().mean(1)
    mse = (q-w).square().mean(1)
    clipped = (w < -8*scale) | (w > 7*scale)
    return dict(max_abs=absolute.amax(1), p99_abs=torch.quantile(absolute, .99, dim=1),
                rms=energy.sqrt(), sw=scale[:,0], relative_rmse=(mse/energy.clamp_min(1e-30)).sqrt(),
                error_mse=mse, zero_code_fraction=(codes==0).float().mean(1),
                clipped_fraction=clipped.float().mean(1))


@torch.no_grad()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output/'plot_down_channels.py')
    write_json(args.output/'settings.json', dict(parent=str(args.parent),reference_state=str(args.reference_state),
        target=TARGET, source=source_record(),
        scope='one down matrix only; no model forward/calibration/optimization/evaluation/export',
        reference='original norm-fused BF16 weights, same saved B100 R via existing rotated_weight; never dequantized W4',
        quantized='original packed INT4 and SW; same existing BF16 dequant_record',
        statistics='rows are output channels; columns are 8192 input weights; relative RMSE=sqrt(mean(error^2)/mean(reference^2)); weight-only, not task saliency'))
    model = build_training_model(args.reference_state).cuda().eval()
    reference = None
    for name, wrapper, rotation2, transpose in rotated_linears(model):
        if name == TARGET:
            reference = wrapper.module.rotated_weight(model.R1.weight, rotation2, transpose).detach().clone()
            break
    if reference is None:
        raise ValueError(TARGET)
    del model, wrapper, rotation2
    gc.collect(); torch.cuda.empty_cache()
    state = torch.load(args.parent, map_location='cpu', weights_only=True)
    record = state['weights'][TARGET]
    del state
    codes = unpack_int4(record['packed'], record['shape']).cuda()
    scale = record['scale'].cuda()
    quantized = dequant_record(record, 'cuda', torch.bfloat16)
    stats = channel_stats(reference, quantized, codes, scale)
    tail = stats['max_abs']/stats['p99_abs'].clamp_min(1e-30)
    stats['max_over_p99'] = tail
    selected = [('median relative RMSE',int(stats['relative_rmse'].argsort()[reference.shape[0]//2])),
                ('largest max/P99 tail',int(tail.argmax())),
                ('largest relative RMSE',int(stats['relative_rmse'].argmax()))]
    public = {k:v.cpu().tolist() for k,v in stats.items()}
    public.update(shape=list(reference.shape), target=TARGET,
        selected_rows=[dict(reason=label,channel=i) for label,i in selected],
        total_relative_rmse=float(((quantized.float()-reference.float()).square().sum()/reference.float().square().sum()).sqrt()),
        clipped_elements=int(((reference.float() < -8*scale)|(reference.float()>7*scale)).sum()),
        quantiles={k:torch.quantile(v.float(),torch.tensor([0.,.5,.99,1.],device=v.device)).cpu().tolist()
                   for k,v in stats.items()})
    write_json(args.output/'channels.json',public)
    examples=[]
    selected_ids=[i for _,i in selected]
    limit=max(10.,float((reference[selected_ids].float()/scale[selected_ids]).abs().max().ceil()))
    for label,i in selected:
        values=reference[i].float()/scale[i,0]
        bins=torch.linspace(-limit,limit,161,device=values.device)
        counts=torch.histc(values,bins=160,min=-limit,max=limit)
        examples.append(dict(channel=i,reason=label,scale=float(scale[i,0]),
            reference_bins=bins.cpu().tolist(),reference_counts=counts.cpu().tolist(),
            reference_outside_plot=int(((values < -limit)|(values > limit)).sum()),
            code_counts=torch.bincount(codes[i].long()+8,minlength=16).cpu().tolist()))
    write_json(args.output/'row_histograms.json',examples)
    del reference,quantized,codes,scale,stats
    torch.cuda.empty_cache()
    plot(args.output,public,examples)
    write_json(args.output/'complete.json',dict(target=TARGET,plots=['channels.png','row_histograms.png']))
    print({k:public[k] for k in ['shape','selected_rows','total_relative_rmse','clipped_elements','quantiles']},flush=True)


def plot(output, data, examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x=np.arange(data['shape'][0])
    fig,axes=plt.subplots(2,2,figsize=(13,7),sharex=True)
    ax=axes[0,0]
    ax.plot(x,data['max_abs'],lw=.5,alpha=.8,label='max |Wref|')
    ax.plot(x,data['p99_abs'],lw=.5,alpha=.8,label='P99 |Wref|')
    ax.set_title('Unquantized weight range within each row'); ax.set_ylabel('Weight magnitude'); ax.legend()
    axes[0,1].plot(x,data['sw'],lw=.5); axes[0,1].set_title('Existing per-output-channel SW');axes[0,1].set_ylabel('Scale (one INT4 step)')
    axes[1,0].plot(x,100*np.array(data['relative_rmse']),lw=.5);axes[1,0].set_title('Quantized-vs-reference relative RMSE');axes[1,0].set_ylabel('RMSE / reference RMS (%)')
    axes[1,1].plot(x,100*np.array(data['zero_code_fraction']),lw=.5);axes[1,1].set_title('Fraction of weights assigned code 0');axes[1,1].set_ylabel('Zero code (%)')
    for ax in axes.flat:
        ax.set_xlabel('Output channel index');ax.grid(alpha=.15)
    fig.suptitle('Layer 1 down_proj: current Phase3 parent, same-R reference\n2048 output channels; 8192 weights per channel; weight-only statistics')
    fig.tight_layout();fig.savefig(output/'channels.png',dpi=170);fig.savefig(output/'channels.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),sharex=True,sharey=True)
    for ax,example in zip(axes,examples):
        edges=np.array(example['reference_bins']);width=edges[1]-edges[0]
        ax.stairs(np.array(example['reference_counts'])/data['shape'][1]/width,edges,color='C0',label='Wref / SW density',fill=True,alpha=.35)
        ax.bar(np.arange(-8,8),np.array(example['code_counts'])/data['shape'][1],width=1,color='C1',alpha=.65,label='Stored q mass / unit bin')
        i=example['channel'];ax.set_title(f"Output channel {i}\n{example['reason']}\nrelative RMSE={100*data['relative_rmse'][i]:.2f}%, max/P99={data['max_over_p99'][i]:.2f}")
        ax.set_xlim(edges[0],edges[-1]);ax.set_xlabel('Weight / original SW');ax.grid(alpha=.15)
    axes[0].set_ylabel('Density / probability per unit-width code bin');axes[0].legend(fontsize=8)
    fig.suptitle('Layer 1 down: row-wise distributions on the original INT4 grid (no requantization)')
    fig.tight_layout();fig.savefig(output/'row_histograms.png',dpi=170);fig.savefig(output/'row_histograms.svg');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--reference-state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
