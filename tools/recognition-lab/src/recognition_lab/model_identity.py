"""Checkpoint identity captured around a load, never during inspection."""
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class ModelIdentity:
    sha256: str | None
    fingerprint_status: str
    architecture: str | None = None
    parameters: int | None = None


def file_signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def fingerprint_before_load(path):
    try:
        before = file_signature(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if before != file_signature(path):
            return None, None, 'unavailable: checkpoint changed during fingerprinting'
        return digest.hexdigest(), before, 'available'
    except OSError as exc:
        return None, None, f'unavailable: {type(exc).__name__}: {exc}'


def identity_after_load(path, fingerprint, signature, status, model):
    if signature is not None:
        try:
            if file_signature(path) != signature:
                fingerprint, status = None, 'unavailable: checkpoint changed during loading'
        except OSError as exc:
            fingerprint, status = None, f'unavailable after loading: {type(exc).__name__}: {exc}'
    architecture = f'{type(model).__module__}.{type(model).__qualname__}'
    try:
        parameters = sum(parameter.numel() for parameter in model.parameters())
    except Exception:
        parameters = None
    return ModelIdentity(fingerprint, status, architecture, parameters)
