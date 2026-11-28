import json

from PIL import Image
import pytest

from grid_geometry.dataset.validation import validate_saved_dataset, validate_training_dataset


@pytest.fixture
def dataset(tmp_path):
    config = dict(experiment='grid-geometry', generator='phase-grid-v1', generator_revision=2,
                  status='complete', input_size=[384, 384], transition_ratio=0.25,
                  train=1, val=1, test=1)
    (tmp_path / 'config.json').write_text(json.dumps(config))
    records = []
    for split in ('train', 'val', 'test'):
        directory = tmp_path / split
        directory.mkdir()
        record = dict(split=split, sample_id='000000', board_id=None, geometry=None,
                      known_box=[0, 0, 384, 384])
        records.append(record)
        Image.new('RGB', (384, 384)).save(directory / '000000.png')
        (directory / '000000.json').write_text(json.dumps(record))
    (tmp_path / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (tmp_path / 'statistics.json').write_text(json.dumps({'accepted': {'overall': {'totals': {'images': 3}}}}))
    return tmp_path


def write_records(root, records):
    (root / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))


def test_valid_and_lightweight_validation(dataset):
    assert validate_saved_dataset(dataset)['status'] == 'complete'
    assert validate_training_dataset(dataset)['split_counts'] == {'train': 1, 'val': 1, 'test': 1}
    (dataset / 'train/000000.png').unlink()
    validate_saved_dataset(dataset)
    with pytest.raises(ValueError, match='unpaired'):
        validate_training_dataset(dataset)


@pytest.mark.parametrize('key,value', [('experiment', 'other'), ('generator', 'other'),
                                      ('generator_revision', 1), ('generator_revision', True),
                                      ('status', 'generating')])
def test_saved_identity(dataset, key, value):
    path = dataset / 'config.json'
    config = json.loads(path.read_text())
    config[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        validate_saved_dataset(dataset)


def test_required_files(dataset):
    (dataset / 'statistics.json').unlink()
    with pytest.raises(FileNotFoundError):
        validate_saved_dataset(dataset)


@pytest.mark.parametrize('damage,match', [('metadata', 'manifest metadata differs'),
                                         ('coverage', 'manifest and sample files differ'),
                                         ('statistics', 'accepted images'),
                                         ('target', 'invalid known_box'),
                                         ('count', 'sample count'),
                                         ('input_size', '384 x 384')])
def test_training_integrity(dataset, damage, match):
    manifest = dataset / 'manifest.jsonl'
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    if damage == 'metadata':
        records[0]['extra'] = 'different'
        write_records(dataset, records)
    elif damage == 'coverage':
        write_records(dataset, records[:-1])
    elif damage == 'statistics':
        (dataset / 'statistics.json').write_text('{"accepted": {"overall": {"totals": {"images": 2}}}}')
    elif damage == 'target':
        records[0]['known_box'] = [0, 0, 500, 384]
        (dataset / 'train/000000.json').write_text(json.dumps(records[0]))
        write_records(dataset, records)
    else:
        path = dataset / 'config.json'
        config = json.loads(path.read_text())
        config['train' if damage == 'count' else 'input_size'] = 2 if damage == 'count' else [32, 32]
        path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        validate_training_dataset(dataset)


def test_cross_split_board_isolation(dataset):
    records = [json.loads(line) for line in (dataset / 'manifest.jsonl').read_text().splitlines()]
    geometry = dict(origin_x=10, origin_y=10, pitch_x=16, pitch_y=16,
                    visible_box=[10, 10, 100, 100])
    for record in records[:2]:
        record.update(board_id='same-board', geometry=geometry)
        (dataset / record['split'] / '000000.json').write_text(json.dumps(record))
    write_records(dataset, records)
    with pytest.raises(ValueError, match='identity overlap'):
        validate_training_dataset(dataset)
