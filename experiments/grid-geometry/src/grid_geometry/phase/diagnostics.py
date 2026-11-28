"""Select and display evaluated phase-grid diagnostic samples."""
import random
import matplotlib.pyplot as plt
from grid_geometry.dataset.loader import PhaseDataset
from grid_geometry.phase.model import PhaseGridUNet, load_phase_weights
from grid_geometry.phase.visualize import plot_sample
from grid_geometry.training import select_device

def visualize_predictions(reports, metadata_by_split, saved_root, best_state, DEVICE, EVALUATION_LIMIT, EVALUATION_SEED, DIAGNOSTIC_SPLIT):
    records = reports[DIAGNOSTIC_SPLIT]["records"]
    metadata = metadata_by_split[DIAGNOSTIC_SPLIT]
    rng = random.Random(42)
    shuffled = rng.sample(records, len(records))
    selected = {}
    def select(record, reason):
        if record is not None:
            selected.setdefault(record["sample_id"], []).append(reason)
    def first(predicate):
        return next((r for r in shuffled if predicate(r, metadata[r["sample_id"]])), None)
    for category in ("complete", "interior", "side", "corner", "negative", "hard_negative"):
        select(first(lambda r, m, category=category: m.get("category") == category), category)
    positive = [r for r in records if metadata[r["sample_id"]].get("geometry")]
    if positive:
        pitch = lambda r: metadata[r["sample_id"]]["geometry"]["pitch_x"]
        select(min(positive, key=pitch), "smallest horizontal pitch")
        select(max(positive, key=pitch), "largest horizontal pitch")
    for label, predicate in (
        ("upsampled", lambda m: max((m.get("transform") or {}).get("scale_x", 1), (m.get("transform") or {}).get("scale_y", 1)) > 1),
        ("downsampled", lambda m: min((m.get("transform") or {}).get("scale_x", 1), (m.get("transform") or {}).get("scale_y", 1)) < 1),
    ):
        select(first(lambda r, m, predicate=predicate: predicate(m)), label)
    select(first(lambda r, m: r["positive"] and r.get("local_geometry_correct") is not True), "local geometry failure")
    select(first(lambda r, m: r["positive"] and r.get("local_geometry_correct") is True and not r["local_accepted"]), "correct local geometry rejected")
    select(first(lambda r, m: r["positive"] and r.get("local_geometry_correct") is False and r["local_accepted"]), "incorrect local geometry accepted")
    select(first(lambda r, m: not r["positive"] and (r["local_accepted"] or r["global"]["accepted"])), "false acceptance")
    print(f"Visualization split: {DIAGNOSTIC_SPLIT}; failure selection: {len(records)} evaluated samples, limit={EVALUATION_LIMIT}, seed={EVALUATION_SEED}")
    print(f"Selected {len(selected)} distinct samples")
    dataset = PhaseDataset(saved_root, DIAGNOSTIC_SPLIT, require_nonempty=True)
    indices = {name: i for i, name in enumerate(dataset.names)}
    device = select_device(DEVICE)
    model = PhaseGridUNet()
    load_phase_weights(model, best_state)
    model.to(device).eval()
    for sample_id, reasons in selected.items():
        print(f"{sample_id}: {', '.join(reasons)}")
        plot_sample(model, dataset.load_sample(indices[sample_id]), device=device)
        plt.show()
