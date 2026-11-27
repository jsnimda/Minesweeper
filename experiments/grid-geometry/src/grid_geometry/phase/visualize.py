"""Notebook-oriented prediction and history visualizations."""
import numpy as np
import torch

from .decoder import decode_phase
from .model import phase_predictions
from .validation import validate_phase


def plot_history(history):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, ax = plt.subplots()
    for split in ('train', 'validation'):
        ax.plot([h['epoch'] for h in history], [h[split]['metrics']['all']['loss'] for h in history], marker='o', markersize=3, label=split)
    ax.set(xlabel='Epoch', ylabel='Loss'); ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8)); ax.legend()
    return fig


def plot_sample(model, sample, device='cpu'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    model.eval()
    with torch.inference_mode():
        pred = phase_predictions(model(sample.image[None].to(device)))[0].cpu().numpy()
    target = sample.targets.numpy(); mask = sample.phase_mask.numpy()
    decoded = decode_phase(pred)
    validation = validate_phase(pred, decoded.hypotheses[0], decoder_status=decoded.status) if decoded.hypotheses else None
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes[0, 0].imshow(sample.image.permute(1,2,0)); axes[0,0].set_title('Image and geometry')
    g = sample.metadata.get('geometry')
    if g:
        l,t,r,b = g['visible_box']; axes[0,0].add_patch(Rectangle((l,t),r-l,b-t, fill=False, edgecolor='lime', label='ground truth'))
    if decoded.hypotheses:
        h=decoded.hypotheses[0]; axes[0,0].add_patch(Rectangle((h.x.origin,h.y.origin),h.x.end-h.x.origin,h.y.end-h.y.origin,fill=False,edgecolor='red',label='decoded ideal')); axes[0,0].legend()
    panels = [(target[0], 'GT x cosine', -1, 1, 'coolwarm'),
              (pred[0], 'Pred x cosine', -1, 1, 'coolwarm'),
              (target[2], 'GT y cosine', -1, 1, 'coolwarm'),
              (pred[2], 'Pred y cosine', -1, 1, 'coolwarm'),
              (target[4], 'GT occupancy', 0, 1, 'gray'),
              (pred[4], 'Pred occupancy', 0, 1, 'gray'),
              (mask, 'Phase mask', 0, 1, 'gray'),
              (sample.occupancy_mask.numpy(), 'Occupancy mask', 0, 1, 'gray')]
    for ax, (data, title, vmin, vmax, cmap) in zip(axes.flat[1:9], panels):
        view = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f'{title}\nrange: {np.min(data):.3f}–{np.max(data):.3f}')
        fig.colorbar(view, ax=ax, fraction=.046, pad=.04)
    for axis, start in enumerate((0, 2)):
        error = np.abs(np.arctan2(pred[start+1]*target[start]-pred[start]*target[start+1], pred[start]*target[start]+pred[start+1]*target[start+1]))
        valid = mask & (np.hypot(pred[start], pred[start+1]) > 1e-8)
        view = axes.flat[9+axis].imshow(np.ma.masked_where(~valid, error), vmin=0, vmax=np.pi, cmap='magma')
        axes.flat[9+axis].set_title(('x' if axis == 0 else 'y') + ' absolute phase error (rad)')
        fig.colorbar(view, ax=axes.flat[9+axis], fraction=.046, pad=.04)
    axes.flat[11].axis('off'); axes.flat[11].text(0,1,f"{sample.metadata['sample_id']}\ncategory: {sample.metadata.get('category')}\ndecoder: {decoded.status}\nvalidation: {validation.reasons if validation else 'no hypothesis'}",va='top',wrap=True)
    fig.tight_layout(); return fig
