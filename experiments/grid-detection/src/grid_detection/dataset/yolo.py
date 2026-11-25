"""Direct conversion of in-memory scene annotations to YOLO labels."""
import json
from pathlib import Path

YOLO_CLASSES = ('game_region', 'grid', 'mine_counter', 'reset_button', 'timer')


def yolo_lines(metadata):
    width, height = metadata['size']
    lines = []
    for board in metadata['boards']:
        for name, box in board['boxes'].items():
            if name not in YOLO_CLASSES:
                continue
            x, y, right, bottom = box
            if not (0 <= x < right <= width+1e-7 and 0 <= y < bottom <= height+1e-7):
                raise ValueError(f'Invalid detection box: {name}: {box}')
            values = ((x+right)/(2*width), (y+bottom)/(2*height),
                      (right-x)/width, (bottom-y)/height)
            lines.append(f'{YOLO_CLASSES.index(name)} ' + ' '.join(f'{v:.8f}' for v in values))
    return lines


def write_yolo_config(root, *, test=False):
    root = Path(root)
    lines = ['path: ' + json.dumps(root.resolve().as_posix()),
             'train: images/train', 'val: images/val']
    if test:
        lines.append('test: images/test')
    lines.append('names:')
    lines += [f'  {index}: {name}' for index, name in enumerate(YOLO_CLASSES)]
    (root / 'data.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')
