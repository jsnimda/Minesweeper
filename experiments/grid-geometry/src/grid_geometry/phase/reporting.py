"""Compact recognition and pixel decision summaries for evaluation reports."""


def recognition_counts(records):
    """Return (numerator, denominator) pairs; correctness is required for TPs."""
    positives = [r for r in records if r['positive']]
    complete = [r for r in positives if r['complete']]
    partial = [r for r in positives if not r['complete']]
    negative = [r for r in records if not r['positive']]
    local_tp = sum(bool(r['local_accepted'] and r['local_correct'] is True) for r in positives)
    complete_tp = sum(bool(r['global']['accepted'] and r['global']['correct'] is True) for r in complete)
    return {
        'Local lattice precision': (local_tp, sum(bool(r['local_accepted']) for r in records)),
        'Local lattice recall': (local_tp, len(positives)),
        'Complete board precision': (complete_tp, sum(bool(r['global']['accepted']) for r in records)),
        'Complete board recall': (complete_tp, len(complete)),
        'Partial board false acceptance rate': (sum(bool(r['global']['accepted']) for r in partial), len(partial)),
        'Negative false acceptance rate': (sum(bool(r['global']['accepted']) for r in negative), len(negative)),
    } | ({'Local geometry precision': (sum(bool(r['local_accepted'] and r['local_geometry_correct'] is True) for r in positives), sum(bool(r['local_accepted']) for r in records)),
          'Local geometry recall': (sum(bool(r['local_accepted'] and r['local_geometry_correct'] is True) for r in positives), len(positives))}
         if all('local_geometry_correct' in r for r in positives) else {})


def outcome_matrix(records):
    """Global acceptance wins; otherwise use its input crop's local acceptance.

    These are decisions, irrespective of geometric correctness. The middle
    outcome is a local lattice without global board acceptance; it does not
    assert that the visible board is partial.
    """
    counts = [[0] * 3 for _ in range(3)]
    for r in records:
        row = 2 if not r['positive'] else 0 if r['complete'] else 1
        g = r['global']
        col = 0 if g['accepted'] else 1 if g['local_accepted'] else 2
        counts[row][col] += 1
    return counts


def occupancy_matrix(report):
    c = report['target_metrics']['all']['occupancy_confusion']
    return [[c['tp'], c['fn']], [c['fp'], c['tn']]]


def fraction(pair):
    n, d = pair
    return f'{n / d:.2%} ({n}/{d})' if d else f'Unavailable ({n}/{d})'


def plot_matrices(matrices, row_labels, column_labels, title):
    """Plot ground-truth rows with a fixed, shared percentage scale."""
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), squeeze=False)
    for ax, (split, counts) in zip(axes[0], matrices.items()):
        values = np.array(counts, dtype=int)
        totals = values.sum(axis=1)
        normalized = np.divide(values * 100., totals[:, None],
                               out=np.full(values.shape, np.nan), where=totals[:, None] != 0)
        ax.imshow(normalized, vmin=0, vmax=100, cmap='Blues')
        for row in range(len(row_labels)):
            for col in range(len(column_labels)):
                text = 'Unavailable\n(0)' if not totals[row] else f'{normalized[row, col]:.2f}%\n({values[row, col]:,})'
                ax.text(col, row, text, ha='center', va='center', fontsize=8,
                        color='white' if totals[row] and normalized[row, col] > 55 else 'black')
        ax.set(title=split, xlabel='Prediction', ylabel='Ground truth')
        ax.set_xticks(range(len(column_labels)), column_labels, rotation=25, ha='right')
        ax.set_yticks(range(len(row_labels)), row_labels)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
