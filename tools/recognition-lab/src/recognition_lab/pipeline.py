"""Reusable, capture-independent inference pipeline."""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from .results import Detection, GeometryResult, RecognitionResult, crop_box
from .tiles import stitch_tiles
from .native import pad_native, unpad_native

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_ROOT = ROOT.parents[1] / 'experiments' / 'grid-geometry'


@dataclass(frozen=True)
class Settings:
    device: str = 'auto'
    imgsz: int = 1280
    confidence: float = .25
    padding: int = 20
    mode: str = 'whole'
    tile: int = 384
    overlap: int = 96
    batch: int = 4
    decoder: str = 'joint'
    config: Path = GEOMETRY_ROOT / 'configs' / 'baseline.toml'
    validation: Path = GEOMETRY_ROOT / 'configs' / 'strict_validation.toml'


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
        self.unet = None
        self.yolo_path = None
        self.unet_path = None

    def load_yolo(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'YOLO model not found: {path}')
        from ultralytics import YOLO
        model = YOLO(str(path))
        names = model.names
        expected = ['game_region', 'grid', 'mine_counter', 'reset_button', 'timer']
        if [names[i] for i in range(len(names))] != expected:
            raise ValueError(f'Unexpected YOLO class order: {names}')
        self.yolo, self.yolo_path = model, path

    def load_unet(self, path, config):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'GridUNet model not found: {path}')
        from grid_geometry.evaluate import load_evaluation_model
        from grid_geometry.training_config import load_config
        model, _ = load_evaluation_model(path, 'best', load_config(Path(config)))
        self.unet, self.unet_path = model, path

    def unload_yolo(self):
        self.yolo = self.yolo_path = None

    def unload_unet(self):
        self.unet = self.unet_path = None

    def detect(self, image, settings):
        if self.yolo is None:
            raise RuntimeError('Load a YOLO model first')
        start = perf_counter()
        result = self.yolo.predict(source=image, imgsz=settings.imgsz, conf=settings.confidence,
                                   device=device_for(settings.device), verbose=False)[0]
        detections = [Detection(result.names[int(c)], float(p), tuple(map(float, b)))
                      for b, p, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy())]
        return RecognitionResult(detections, timings={'YOLO wall ms': (perf_counter()-start)*1000,
                                                       **{f'YOLO reported {k} ms': v for k, v in result.speed.items()}})

    def geometry(self, image, result, settings):
        import torch
        from grid_geometry.preprocessing import preprocess_image, ImageTransform
        from grid_geometry.training_config import load_config
        from grid_geometry.evaluate import decode_predictions
        from grid_geometry.validation import load_validation_settings
        from grid_geometry.validation_metrics import validate_decoding
        if self.unet is None:
            raise RuntimeError('Load a GridUNet model first')
        config = load_config(Path(settings.config))
        validation_settings = load_validation_settings(Path(settings.validation))
        device = device_for(settings.device)
        model = self.unet.to(device).eval()
        output = RecognitionResult(list(result.detections), [], dict(result.timings))
        start_all = perf_counter()
        for index, detection in enumerate(result.detections):
            if detection.name != 'grid':
                continue
            times = {}
            grid_start = perf_counter()
            t = grid_start
            crop = crop_box(detection.box, image.size, settings.padding)
            patch = image.crop(crop.box)
            times['crop ms'] = (perf_counter()-t)*1000
            t = perf_counter()
            infer_time = [0.0]
            def predict(batch):
                prepare_start = perf_counter()
                tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2), dtype=np.float32) / 255).to(device)
                if device == 'cuda': torch.cuda.synchronize()
                if device == 'mps': torch.mps.synchronize()
                times['preprocess ms'] += (perf_counter()-prepare_start)*1000
                before = perf_counter()
                with torch.inference_mode():
                    maps = model(tensor).float().sigmoid()
                if device == 'cuda': torch.cuda.synchronize()
                if device == 'mps': torch.mps.synchronize()
                infer_time[0] += (perf_counter()-before)*1000
                return maps.cpu().numpy()
            if settings.mode == 'whole':
                canvas, transform = preprocess_image(patch, (config.dataset.width, config.dataset.height))
                times['preprocess ms'] = (perf_counter()-t)*1000
                maps = predict(np.asarray(canvas)[None])[0]
                input_size = canvas.size
                left, top = transform.offset
                padding = (left, top, canvas.width - left - transform.resized_size[0], canvas.height - top - transform.resized_size[1])
            elif settings.mode == 'native':
                padded, padding = pad_native(np.asarray(patch))
                input_size = (padded.shape[1], padded.shape[0])
                transform = ImageTransform(patch.size, patch.size, patch.size, (0, 0))
                times['preprocess ms'] = (perf_counter()-t)*1000
                maps = unpad_native(predict(padded[None])[0], padding)
            elif settings.mode == 'tiled':
                rgb = np.asarray(patch)
                input_size = (settings.tile, settings.tile)
                padding = None  # Tiles share a padded canvas; padding is not a single model input margin.
                transform = ImageTransform(patch.size, patch.size, patch.size, (0, 0))
                times['preprocess ms'] = (perf_counter()-t)*1000
                t = perf_counter()
                maps = stitch_tiles(rgb, settings.tile, settings.overlap, settings.batch, predict)
                times['tile preparation and combination wall ms'] = (perf_counter()-t)*1000 - infer_time[0]
            else:
                raise ValueError('Unknown inference mode')
            times['UNet synchronized forward ms'] = infer_time[0]
            t = perf_counter()
            decodings, _ = decode_predictions(maps, settings.decoder)
            decoding = decodings[settings.decoder]
            times['decode ms'] = (perf_counter()-t)*1000
            validation, elapsed = validate_decoding(maps, decoding, validation_settings)
            times['validation ms'] = elapsed
            hypotheses = []
            for h in decoding.hypotheses:
                xs = (h.x_edges - transform.offset[0]) / transform.scale_x + crop.box[0]
                ys = (h.y_edges - transform.offset[1]) / transform.scale_y + crop.box[1]
                hypotheses.append((xs, ys, h.score))
            times['total grid geometry ms'] = (perf_counter()-grid_start)*1000
            diagnostics = {'mode': settings.mode, 'crop dimensions': patch.size,
                           'model input dimensions': input_size, 'padding (left, top, right, bottom)': padding,
                           'heatmap dimensions': (maps.shape[2], maps.shape[1])}
            output.grids.append(GeometryResult(index, crop, maps, transform, decoding, validation, hypotheses, times, diagnostics))
        output.timings['geometry wall ms'] = (perf_counter()-start_all)*1000
        return output
