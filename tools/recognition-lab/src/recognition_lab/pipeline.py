"""Reusable, capture-independent inference pipeline."""
from dataclasses import dataclass
from pathlib import Path
from itertools import count
from time import perf_counter
import numpy as np
from PIL import Image
from .results import Detection, GeometryResult, RecognitionResult, crop_box
from .provenance import DetectionProvenance, GeometryProvenance, require_settings
from .model_identity import fingerprint_before_load, identity_after_load

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_ROOT = ROOT.parents[1] / 'experiments' / 'grid-geometry'
_MODEL_IDS = count(1)


@dataclass(frozen=True)
class Settings:
    device: str = 'auto'
    imgsz: int = 1280
    confidence: float = .25
    padding: int = 20
    mode: str = 'whole'
    tile: int = 384
    overlap: int = 96


def device_for(name):
    import torch
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available() or name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError(f'{name} is not available')
    if name not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Unknown device')
    return name


class Pipeline:
    def __init__(self):
        self.yolo = None
        self.yolo_path = None
        self.yolo_id = None
        self.yolo_identity = None
        self.phase = None
        self.phase_path = None
        self.phase_id = None
        self.phase_identity = None

    def load_phase(self, path):
        import torch
        from grid_geometry.phase.model import PhaseGridUNet, load_phase_weights
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'PhaseGridUNet model not found: {path}')
        fingerprint, signature, status = fingerprint_before_load(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = PhaseGridUNet()
        load_phase_weights(model, state)
        identity = identity_after_load(path, fingerprint, signature, status, model)
        self.phase, self.phase_path, self.phase_id, self.phase_identity = model, path, next(_MODEL_IDS), identity

    def unload_phase(self):
        self.phase = self.phase_path = self.phase_id = self.phase_identity = None

    def load_yolo(self, path):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'YOLO model not found: {path}')
        from ultralytics import YOLO
        fingerprint, signature, status = fingerprint_before_load(path)
        model = YOLO(str(path))
        names = model.names
        expected = ['game_region', 'grid', 'mine_counter', 'reset_button', 'timer']
        if [names[i] for i in range(len(names))] != expected:
            raise ValueError(f'Unexpected YOLO class order: {names}')
        identity = identity_after_load(path, fingerprint, signature, status, getattr(model, 'model', model))
        self.yolo, self.yolo_path, self.yolo_id, self.yolo_identity = model, path, next(_MODEL_IDS), identity

    def unload_yolo(self):
        self.yolo = self.yolo_path = self.yolo_id = self.yolo_identity = None

    def detect(self, image, settings):
        if self.yolo is None:
            raise RuntimeError('Load a YOLO model first')
        require_settings(settings, 'detection')
        device = device_for(settings.device)
        provenance = DetectionProvenance(settings.device, device, str(self.yolo_path) if self.yolo_path is not None else None,
                                         self.yolo_id, settings.imgsz, settings.confidence,
                                         checkpoint_sha256=self.yolo_identity.sha256 if self.yolo_identity else None,
                                         fingerprint_status=self.yolo_identity.fingerprint_status if self.yolo_identity else 'unavailable')
        start = perf_counter()
        result = self.yolo.predict(source=image, imgsz=settings.imgsz, conf=settings.confidence,
                                   device=device, verbose=False)[0]
        detections = [Detection(result.names[int(c)], float(p), tuple(map(float, b)))
                      for b, p, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy())]
        return RecognitionResult(detections, timings={'YOLO wall ms': (perf_counter()-start)*1000,
                                                       **{f'YOLO reported {k} ms': v for k, v in result.speed.items()}},
                                 detection_provenance=provenance)

    def _geometry_output(self, result, settings, device):
        """Create stage metadata without copying detection objects or old geometry."""
        tiled = settings.mode == 'tiled'
        provenance = GeometryProvenance(
            settings.device, device, str(self.phase_path) if self.phase_path is not None else None,
            self.phase_id, settings.padding, settings.mode,
            tile=settings.tile if tiled else None, overlap=settings.overlap if tiled else None,
            whole_input_size=(384, 384) if not tiled and any(d.name == 'grid' for d in result.detections) else None,
            checkpoint_sha256=self.phase_identity.sha256 if self.phase_identity else None,
            fingerprint_status=self.phase_identity.fingerprint_status if self.phase_identity else 'unavailable')
        return RecognitionResult(list(result.detections), [], dict(result.timings),
                                 result.detection_provenance, provenance)

    def geometry(self, image, result, settings):
        """Run the phase model on a resized crop, preserving diagnostic proposals."""
        import torch
        from grid_geometry.phase.decoder import DecoderSettings, decode_phase
        from grid_geometry.phase.model import phase_predictions
        from grid_geometry.phase.global_geometry import recognize_global
        from grid_geometry.phase.tiles import TileObservation
        from grid_geometry.phase.validation import validate_phase
        from grid_geometry.preprocessing import ImageTransform, preprocess_image
        if self.phase is None:
            raise RuntimeError('Load a PhaseGridUNet model first')
        require_settings(settings, 'geometry')
        if settings.mode == 'tiled':
            return self.phase_tiles(image, result, settings)
        if settings.mode != 'whole':
            raise ValueError('PhaseGridUNet supports whole or tiled mode')
        device = device_for(settings.device)
        model = self.phase.to(device).eval()
        output = self._geometry_output(result, settings, device)
        started = perf_counter()
        for index, detection in enumerate(result.detections):
            if detection.name != 'grid':
                continue
            begin = perf_counter()
            crop = crop_box(detection.box, image.size, settings.padding)
            patch = image.crop(crop.box)
            canvas, tr = preprocess_image(patch, (384, 384))
            # Retain the exact unbatched CPU model input: RGB CHW float32 / 255.
            data = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1).copy() / 255
            tensor = torch.from_numpy(data[None]).to(device)
            if device == 'cuda': torch.cuda.synchronize()
            if device == 'mps': torch.mps.synchronize()
            forward = perf_counter()
            with torch.inference_mode():
                values = phase_predictions(model(tensor)).cpu().numpy()[0]
            if device == 'cuda': torch.cuda.synchronize()
            if device == 'mps': torch.mps.synchronize()
            forward_ms = (perf_counter() - forward) * 1000
            left, top = tr.offset
            rw, rh = tr.resized_size
            # Exclude artificial padding: it cannot establish a board termination.
            fields = values[:, top:top+rh, left:left+rw]
            transform = ImageTransform(patch.size, tr.resized_size, tr.resized_size, (0, 0))
            t = perf_counter()
            decoded = decode_phase(fields, DecoderSettings(scale_x=tr.scale_x, scale_y=tr.scale_y))
            local = validate_phase(fields, decoded.hypotheses[0], decoder_status=decoded.status) if decoded.hypotheses else None
            observation = TileObservation((0, 0, rw, rh), decoded.hypotheses[0] if decoded.hypotheses else None,
                                          bool(local and local.local_accepted), local.local_reasons if local else decoded.reasons, fields,
                                          input_array=data, input_transform=tr, full_fields=values,
                                          decoding=decoded, local_validation=local,
                                          input_representation='RGB CHW float32 divided by 255', resize_method='bilinear')
            validation = recognize_global([observation], (rh, rw))
            processing_ms = (perf_counter() - t) * 1000
            hypotheses = []
            if validation.geometry is not None:
                xs = validation.geometry.x_edges / tr.scale_x + crop.box[0]
                ys = validation.geometry.y_edges / tr.scale_y + crop.box[1]
                hypotheses.append((xs, ys, decoded.hypotheses[0].score))
            diagnostics = {'mode': settings.mode, 'image dimensions': image.size,
                           'crop dimensions': patch.size, 'model input dimensions': canvas.size,
                           'valid resized region': (left, top, left+rw, top+rh),
                           'local valid': bool(local and local.local_accepted),
                           'local reasons': local.local_reasons if local else ('no hypothesis',),
                           'global lattice': validation.lattice,
                           'terminations (resized crop coordinates)': validation.boundaries,
                           'support': validation.measurements,
                           'decoder reasons': decoded.reasons}
            times = {'phase forward ms': forward_ms, 'decode and validation ms': processing_ms,
                     'total grid geometry ms': (perf_counter()-begin)*1000}
            output.grids.append(GeometryResult(index, crop, fields, transform, decoded, validation, hypotheses, times, diagnostics,
                                               {'global': validation, 'observations': [observation]}))
        output.timings['geometry wall ms'] = (perf_counter()-started)*1000
        return output

    def phase_tiles(self, image, result, settings):
        """Run tiled phase inference and strict global geometry validation."""
        import torch
        from grid_geometry.phase.model import phase_predictions
        from grid_geometry.phase.tiles import observe_tiles
        from grid_geometry.phase.global_geometry import recognize_global
        from grid_geometry.preprocessing import ImageTransform
        if self.phase is None:
            raise RuntimeError('Load a PhaseGridUNet model first')
        require_settings(settings, 'geometry')
        if settings.mode != 'tiled':
            raise ValueError('Tiled inference requires tiled mode')
        device = device_for(settings.device)
        model = self.phase.to(device).eval()
        output = self._geometry_output(result, settings, device)
        start = perf_counter()
        for index, detection in enumerate(result.detections):
            if detection.name != 'grid':
                continue
            crop = crop_box(detection.box, image.size, settings.padding)
            rgb = np.asarray(image.crop(crop.box))
            height, width = rgb.shape[:2]
            forward_ms = [0.0]
            def predict(padded):
                # Keep the same normalized CPU storage that feeds the model.
                input_tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float().div_(255)
                input_array = input_tensor.numpy()
                tensor = input_tensor.unsqueeze(0).to(device)
                if device == 'cuda': torch.cuda.synchronize()
                if device == 'mps': torch.mps.synchronize()
                before = perf_counter()
                with torch.inference_mode():
                    fields = phase_predictions(model(tensor)).cpu().numpy()[0]
                if device == 'cuda': torch.cuda.synchronize()
                if device == 'mps': torch.mps.synchronize()
                forward_ms[0] += (perf_counter() - before) * 1000
                return fields, input_array
            t = perf_counter()
            observations, layout = observe_tiles(rgb, settings.tile, settings.overlap, predict, retain_artifacts=True,
                                                 input_representation='RGB CHW float32 divided by 255')
            validation = recognize_global(observations, (height, width))
            processing_ms = (perf_counter() - t) * 1000 - forward_ms[0]
            hypotheses = []
            if validation.geometry is not None:
                hypotheses.append((validation.geometry.x_edges + crop.box[0], validation.geometry.y_edges + crop.box[1], 0.0))
            diagnostics = {'mode': 'tiled', 'image dimensions': image.size,
                           'crop dimensions': (width, height), 'tile layout (columns, rows)': layout,
                           'tile count': len(observations), 'local valid tiles': sum(o.local_valid for o in observations),
                           'consensus': validation.lattice, 'terminations (crop coordinates)': validation.boundaries,
                           'support': validation.measurements, 'complete-board validity': validation.accepted}
            transform = ImageTransform((width, height), (width, height), (width, height), (0, 0))
            output.grids.append(GeometryResult(index, crop, None, transform,
                                               None, validation, hypotheses, {'phase forward ms': forward_ms[0], 'geometry processing ms': processing_ms}, diagnostics,
                                               {'global': validation, 'observations': observations}))
        output.timings['geometry wall ms'] = (perf_counter()-start)*1000
        return output
