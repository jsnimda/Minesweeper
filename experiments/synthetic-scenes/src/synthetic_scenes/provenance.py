"""Small, read-only provenance snapshots for dataset generation runs."""
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess


GENERATION_DEPENDENCIES = ('numpy', 'pillow', 'skia-python', 'pydantic', 'tqdm', 'synthetic-scenes')


def source_state(anchor, experiment):
    """Inspect a checkout without relying on the caller's working directory.

    Dirty state includes untracked, non-ignored files. Source scope excludes
    notebooks, documentation, datasets and outputs; paths are repository-relative.
    """
    paths = [f'experiments/{experiment}/src', f'experiments/{experiment}/pyproject.toml',
             f'experiments/{experiment}/uv.lock', 'experiments/synthetic-scenes/src',
             'experiments/synthetic-scenes/pyproject.toml', 'experiments/synthetic-scenes/uv.lock',
             'pyproject.toml', 'uv.lock']
    result = {'commit': None, 'repository_dirty': None, 'generation_source_dirty': None,
              'generation_source_paths': paths}
    try:
        def git(*args, directory=anchor):
            return subprocess.run(['git', '-C', str(directory), *args], check=True,
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        root = Path(git('rev-parse', '--show-toplevel')).resolve()
        # An installed package in an unrelated checkout is not this source tree.
        if not (root / paths[0]).is_dir() or not Path(anchor).resolve().is_relative_to(root / paths[0]):
            return result
        result['commit'] = git('rev-parse', 'HEAD')
        result['repository_dirty'] = bool(git('status', '--porcelain', '--untracked-files=all', directory=root))
        result['generation_source_dirty'] = bool(git('status', '--porcelain', '--untracked-files=all', '--', *paths, directory=root))
    except (OSError, subprocess.SubprocessError, ValueError, UnicodeError):
        pass
    return result


def environment_state(experiment=None):
    versions = {}
    names = GENERATION_DEPENDENCIES + ((experiment,) if experiment else ())
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except (metadata.PackageNotFoundError, OSError, ValueError, TypeError):
            versions[name] = None
    return {'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'os': platform.system(), 'machine': platform.machine(), 'dependencies': versions,
            'logical_cpu_count': os.cpu_count(),
            'available_cpu_count': getattr(os, 'process_cpu_count', lambda: None)()}


def asset_inventory(assets):
    """Hash ordered category-relative names and contents from the actual inventory.

    Fingerprint observes selection order as well as file bytes. Absolute paths and
    content are not exported. Failures leave an explicitly unavailable fingerprint.
    """
    categories = {}
    digest = sha256()
    try:
        for category, paths in assets.items():
            identifiers = []
            for item in paths:
                path = Path(item)
                identifiers.append(path.name)
                file_hash = sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        file_hash.update(block)
                entry = json.dumps([category, path.name, file_hash.hexdigest()], ensure_ascii=True, separators=(',', ':'))
                digest.update(entry.encode('utf-8') + b'\n')
            categories[category] = {'count': len(identifiers), 'identifiers': identifiers}
    except (OSError, ValueError):
        return {'status': 'unavailable', 'algorithm': 'sha256-ordered-names-and-content',
                'categories': None, 'fingerprint': None}
    return {'status': 'available', 'algorithm': 'sha256-ordered-names-and-content',
            'categories': categories, 'fingerprint': digest.hexdigest()}


def run_provenance(anchor, experiment):
    return {'source': source_state(anchor, experiment), 'environment': environment_state(experiment),
            'fonts': {'resolution': 'system font matching; exact resolved fonts not recorded'}}
