
"""Export deterministic synthetic desktop screenshots and annotations."""

import json
import re
from pathlib import Path

from tqdm.std import tqdm

from .sampler import sample_scene
from ..synthetic.renderer import render_scene


_PLACEMENT_ERROR = re.compile(
    r"Could not place \d+ protected windows "
    r"after \d+ layout attempts"
)


def generate_dataset(
    output_dir: str | Path,
    count: int,
    start_seed: int = 0,
    *,
    max_attempts: int | None = None,
) -> dict:
    """Generate exactly `count` successfully rendered screenshots.

    Known protected-window placement failures are skipped.
    Unexpected rendering errors are raised immediately.

    Existing output files are never silently overwritten.
    """
    if count < 0:
        raise ValueError("count must be nonnegative")

    if max_attempts is None:
        max_attempts = max(10, count * 10)

    if max_attempts < count:
        raise ValueError("max_attempts must be at least count")

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    manifest_path = output_dir / "manifest.json"

    # Avoid accidentally overwriting a previous dataset.
    for path in (images_dir, annotations_dir):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {path}"
            )

    if manifest_path.exists():
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}"
        )

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    accepted = []
    rejected = []

    with tqdm(
        total=count,
        desc="Generating screenshots",
        unit="img",
    ) as progress:
        for attempt in range(max_attempts):
            if len(accepted) == count:
                break

            seed = start_seed + attempt
            scene = sample_scene(seed)

            try:
                image, annotations, resolved_scene = render_scene(scene)
            except ValueError as exc:
                if not _PLACEMENT_ERROR.fullmatch(str(exc)):
                    raise

                rejected.append({
                    "seed": seed,
                    "reason": str(exc),
                })
                progress.set_postfix(rejected=len(rejected))
                continue

            index = len(accepted)
            filename = f"{index:06d}"

            image_path = images_dir / f"{filename}.png"
            annotation_path = annotations_dir / f"{filename}.json"

            annotation_record = {
                "seed": seed,
                "image": f"images/{filename}.png",
                "width": image.width,
                "height": image.height,
                "annotations": annotations,
                "scene": resolved_scene,
            }

            image.save(image_path, format="PNG")

            with annotation_path.open("w", encoding="utf-8") as file:
                json.dump(
                    annotation_record,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            accepted.append({
                "index": index,
                "seed": seed,
                "image": f"images/{filename}.png",
                "annotation": f"annotations/{filename}.json",
            })

            progress.update(1)
            progress.set_postfix(rejected=len(rejected))

    manifest = {
        "requested_count": count,
        "generated_count": len(accepted),
        "start_seed": start_seed,
        "attempted_count": len(accepted) + len(rejected),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
    }

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if len(accepted) != count:
        raise RuntimeError(
            f"Generated {len(accepted)}/{count} screenshots "
            f"after {max_attempts} attempts. "
            f"Partial dataset saved to {output_dir}"
        )

    return manifest
