"""Streaming pixel-weighted phase and occupancy metrics."""
import math

import torch
import torch.nn.functional as F

from .model import phase_predictions


class TargetMetrics:
    def __init__(self, occupancy_weight=1.0):
        self.weight = occupancy_weight
        self.groups = {name: dict(samples=0, phase_n=0, occ_n=0, x=0., y=0., bce=0., angle_x=0., angle_y=0., angle_n_x=0, angle_n_y=0, mag_x=0., mag_y=0., tp=0, fp=0, fn=0, tn=0) for name in ('all', 'positive', 'negative')}

    def update(self, logits, targets, phase_mask, occupancy_mask, positive):
        logits, targets = logits.detach().float(), targets.detach().float()
        p = phase_predictions(logits)
        for group, select in (('all', torch.ones_like(positive, dtype=torch.bool)), ('positive', positive.bool()), ('negative', ~positive.bool())):
            d = self.groups[group]
            d['samples'] += int(select.sum())
            if not select.any():
                continue
            pm, om = phase_mask[select].bool(), occupancy_mask[select].bool()
            pred, target, raw = p[select], targets[select], logits[select]
            d['phase_n'] += int(pm.sum()); d['occ_n'] += int(om.sum())
            for axis, start in (('x', 0), ('y', 2)):
                a, b = pred[:, start:start+2], target[:, start:start+2]
                d[axis] += float(((a-b).square().sum(1) * pm).sum()) / 2
                angle = torch.atan2(a[:, 1]*b[:, 0]-a[:, 0]*b[:, 1], (a*b).sum(1)).abs()
                angle_mask = pm & (torch.linalg.vector_norm(a, dim=1) > 1e-8)
                d['angle_'+axis] += float((angle * angle_mask).sum())
                d['angle_n_'+axis] += int(angle_mask.sum())
                d['mag_'+axis] += float((torch.linalg.vector_norm(a, dim=1) * pm).sum())
            d['bce'] += float((F.binary_cross_entropy_with_logits(raw[:, 4], target[:, 4], reduction='none') * om).sum())
            pred_pos, true_pos = pred[:, 4] >= .5, target[:, 4] >= .5
            d['tp'] += int((pred_pos & true_pos & om).sum()); d['fp'] += int((pred_pos & ~true_pos & om).sum()); d['fn'] += int((~pred_pos & true_pos & om).sum()); d['tn'] += int((~pred_pos & ~true_pos & om).sum())

    def compute(self):
        def div(a, b):
            return a / b if b else None
        result = {}
        for name, d in self.groups.items():
            pn, on = d['phase_n'], d['occ_n']
            x, y, occ = div(d['x'], pn), div(d['y'], pn), div(d['bce'], on)
            phase = (x+y)/2 if pn else None
            result[name] = {'samples': d['samples'], 'phase_pixels': pn, 'occupancy_pixels': on,
                'loss': (phase or 0.) + self.weight * occ if occ is not None else None,
                'phase_loss': phase, 'x_loss': x, 'y_loss': y, 'occupancy_loss': occ,
                'phase_mae_radians_x': div(d['angle_x'], d['angle_n_x']), 'phase_mae_radians_y': div(d['angle_y'], d['angle_n_y']),
                'angle_pixels_x': d['angle_n_x'], 'angle_pixels_y': d['angle_n_y'],
                'magnitude_x': div(d['mag_x'], pn), 'magnitude_y': div(d['mag_y'], pn),
                'precision': div(d['tp'], d['tp']+d['fp']), 'recall': div(d['tp'], d['tp']+d['fn']),
                'iou': div(d['tp'], d['tp']+d['fp']+d['fn']),
                'occupancy_confusion': {key: d[key] for key in ('tp', 'fp', 'fn', 'tn')}}
        return result
