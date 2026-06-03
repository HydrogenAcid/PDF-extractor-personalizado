from __future__ import annotations

from pathlib import Path

from PIL import Image

FRAMES_DIR = Path("output/frames/animacion_charles")
OUTPUT_DIR = Path("output/gifs/animacion_charles")
OUTPUT_NAME = "animacion_charles.gif"
ORIGINAL_FRAME_COUNT = 140
FRAME_DURATION_MS = 375


def build_animacion_charles_gif(
    frame_count: int = ORIGINAL_FRAME_COUNT,
    duration_ms: int = FRAME_DURATION_MS,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_paths = [FRAMES_DIR / f"frame_{index:03d}.png" for index in range(1, frame_count + 1)]
    missing = [str(path) for path in frame_paths if not path.exists()]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Faltan frames para construir el GIF: {preview}")

    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    output_path = OUTPUT_DIR / OUTPUT_NAME

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return output_path


if __name__ == "__main__":
    output = build_animacion_charles_gif()
    print(output)
    print(f"frame_count={ORIGINAL_FRAME_COUNT}")
    print(f"duration_ms={FRAME_DURATION_MS}")
