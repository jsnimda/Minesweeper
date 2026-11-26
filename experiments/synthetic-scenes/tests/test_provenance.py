"""Read-only provenance behavior with controlled source and asset fixtures."""
from pathlib import Path
from types import SimpleNamespace
import subprocess

from synthetic_scenes import provenance


def _checkout(tmp_path):
    root = tmp_path / 'repo'
    anchor = root / 'experiments/grid-detection/src/grid_detection/dataset'
    anchor.mkdir(parents=True)
    return root, anchor


def test_source_state_scopes_dirty_files(monkeypatch, tmp_path):
    root, anchor = _checkout(tmp_path)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if '--show-toplevel' in command:
            output = str(root)
        elif 'HEAD' in command:
            output = 'abc123'
        elif '--' in command:
            output = '?? experiments/grid-detection/src/new.py'
        else:
            output = ' M experiments/grid-detection/README.md\n?? experiments/grid-detection/src/new.py'
        return SimpleNamespace(stdout=output)
    monkeypatch.setattr(provenance.subprocess, 'run', run)
    state = provenance.source_state(anchor, 'grid-detection')
    assert state['commit'] == 'abc123'
    assert state['repository_dirty'] and state['generation_source_dirty']
    assert 'experiments/grid-detection/README.md' not in state['generation_source_paths']
    assert all('--untracked-files=all' in command for command in calls if 'status' in command)
    assert all(command[2] == str(root) for command in calls if 'status' in command)


def test_clean_and_unrelated_documentation(monkeypatch, tmp_path):
    root, anchor = _checkout(tmp_path)
    def check_status(document_change):
        def run(command, **kwargs):
            if '--show-toplevel' in command: output = str(root)
            elif 'HEAD' in command: output = '1234'
            elif '--' in command: output = ''
            else: output = ' M README.md' if document_change else ''
            return SimpleNamespace(stdout=output)
        monkeypatch.setattr(provenance.subprocess, 'run', run)
        return provenance.source_state(anchor, 'grid-detection')
    assert check_status(False)['repository_dirty'] is False
    state = check_status(True)
    assert state['repository_dirty'] is True and state['generation_source_dirty'] is False


def test_git_unavailable_and_unrelated_repository(monkeypatch, tmp_path):
    root, anchor = _checkout(tmp_path)
    def unavailable(*args, **kwargs):
        raise FileNotFoundError('git')
    monkeypatch.setattr(provenance.subprocess, 'run', unavailable)
    assert provenance.source_state(anchor, 'grid-detection')['repository_dirty'] is None
    monkeypatch.setattr(provenance.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=str(root)))
    outside = tmp_path / 'installed'; outside.mkdir()
    state = provenance.source_state(outside, 'grid-detection')
    assert state['commit'] is None and state['generation_source_dirty'] is None
    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(128, 'git')
    monkeypatch.setattr(provenance.subprocess, 'run', failed)
    assert provenance.source_state(anchor, 'grid-detection')['commit'] is None


def test_selected_environment_versions(monkeypatch):
    def version(name):
        if name == 'numpy': return '1.2.3'
        raise provenance.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(provenance.metadata, 'version', version)
    snapshot = provenance.environment_state('grid-geometry')
    assert snapshot['dependencies']['numpy'] == '1.2.3'
    assert snapshot['dependencies']['pillow'] is None
    assert snapshot['dependencies']['grid-geometry'] is None


def test_asset_fingerprint_tracks_bytes_names_and_order(tmp_path):
    a = tmp_path / 'a.png'; b = tmp_path / 'b.png'
    a.write_bytes(b'one'); b.write_bytes(b'two')
    inventory = {'artwork': [a, b]}
    first = provenance.asset_inventory(inventory)
    assert first['status'] == 'available' and first['categories']['artwork']['identifiers'] == ['a.png', 'b.png']
    assert provenance.asset_inventory(inventory) == first
    assert provenance.asset_inventory({'artwork': [b, a]})['fingerprint'] != first['fingerprint']
    b.write_bytes(b'changed')
    assert provenance.asset_inventory(inventory)['fingerprint'] != first['fingerprint']
    b.unlink()
    assert provenance.asset_inventory(inventory)['status'] == 'unavailable'
