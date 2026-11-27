"""Visualize saved phase-grid samples and their supervision."""
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from grid_geometry.dataset.loader import PhaseDataset

def visualize_samples(records, positives, destination, seed=42):
    rng = random.Random(seed)
    selected = {}
    def choose(label, record):
        if record is None:
            print(f"Unavailable: {label}")
        else:
            key = (record["split"], record["sample_id"])
            selected.setdefault(key, {"record": record, "reasons": []})["reasons"].append(label)

    def first(predicate):
        return next((r for r in records if predicate(r)), None)

    def extreme(label, source, key, maximum=False):
        values = [r for r in source if key(r) is not None]
        choose(label, (max if maximum else min)(values, key=key) if values else None)

    for r in rng.sample(records, min(2, len(records))): choose("random", r)
    for category in ("complete", "interior", "side", "corner", "negative", "hard_negative"):
        choose(category, first(lambda r: r.get("category") == category))
    extreme("small pitch", positives, lambda r: r.get("effective_pitch_x"))
    extreme("large pitch", positives, lambda r: r.get("effective_pitch_x"), maximum=True)
    extreme("low grid coverage", positives, lambda r: r.get("grid_coverage_fraction"))
    for state, predicate in (("upsampling", lambda v: v > 1 + 1e-9), ("downsampling", lambda v: v < 1 - 1e-9)):
        choose(state, first(lambda r: any(predicate(v) for v in ((r.get("transform") or {}).get("scale_x"), (r.get("transform") or {}).get("scale_y")) if v is not None)))
    for context in ("procedural", "desktop"):
        choose(context + " fill", first(lambda r: r.get("background_visible") and r.get("context_type") == context))
    choose("intentional padding", first(lambda r: r.get("padding_type") is not None))
    choose("source fitting", first(lambda r: r.get("sampled_cell_size") is not None and r.get("resolved_cell_size") is not None and r["sampled_cell_size"] > r["resolved_cell_size"]))
    choose("standalone budget reduction", first(lambda r: r.get("scene_mode") == "game" and r.get("sampled_cell_size") is not None and r.get("requested_cell_size") is not None and r["sampled_cell_size"] > r["requested_cell_size"]))

    datasets = {}
    for item in selected.values():
        r = item["record"]
        split = r["split"]
        if split not in datasets: datasets[split] = PhaseDataset(destination, split)
        dataset = datasets[split]
        sample = dataset.load_sample(dataset.names.index(r["sample_id"]))
        targets = sample.targets.numpy()
        mask = sample.phase_mask.numpy()
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        axes = axes.ravel()
        axes[0].imshow(sample.image.permute(1, 2, 0), extent=(0, dataset.size[0], dataset.size[1], 0))
        axes[0].set_title(", ".join(item["reasons"]) + " / RGB", fontsize=9)
        geometry = r.get("geometry")
        if geometry:
            w, h = dataset.size
            ox, oy, px, py = geometry["origin_x"], geometry["origin_y"], geometry["pitch_x"], geometry["pitch_y"]
            rows, columns = r["board_dimensions"]
            left, right = max(0, ox), min(w, ox + columns * px)
            top, bottom = max(0, oy), min(h, oy + rows * py)
            if left < right and top < bottom:
                for x in ox + np.arange(columns + 1) * px:
                    if 0 <= x <= w: axes[0].plot([x, x], [top, bottom], color="yellow", lw=.5)
                for y in oy + np.arange(rows + 1) * py:
                    if 0 <= y <= h: axes[0].plot([left, right], [y, y], color="yellow", lw=.5)
            vl, vt, vr, vb = geometry["visible_box"]
            axes[0].add_patch(Rectangle((vl, vt), vr-vl, vb-vt, fill=False, edgecolor="cyan", lw=1.2, linestyle="--"))
            axes[0].set_xlim(0, w); axes[0].set_ylim(h, 0)
        for i, title in enumerate(("X cosine", "X sine", "Y cosine", "Y sine")):
            axes[i+1].imshow(np.ma.masked_where(~mask, targets[i]), cmap="coolwarm", vmin=-1, vmax=1)
            axes[i+1].set_title(title)
        axes[5].imshow(targets[4], cmap="gray", vmin=0, vmax=1); axes[5].set_title("Soft occupancy")
        axes[6].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[6].set_title("Phase supervision mask")
        axes[7].imshow(sample.occupancy_mask, cmap="gray", vmin=0, vmax=1); axes[7].set_title("Occupancy supervision mask")
        for ax in axes: ax.axis("off")
        pitch = (r.get("effective_pitch_x"), r.get("effective_pitch_y"))
        details = f" | pitch=({pitch[0]:.2f}, {pitch[1]:.2f})" if all(v is not None for v in pitch) else ""
        coverage = r.get("grid_coverage_fraction")
        padding = r.get("padding_fraction")
        if coverage is not None: details += f" | coverage={coverage:.1%}"
        if padding is not None: details += f" | padding={padding:.1%}"
        fig.suptitle(f"{split}/{r['sample_id']} | {r.get('category')} | {r.get('style')}{details}", fontsize=10)
        plt.show()
