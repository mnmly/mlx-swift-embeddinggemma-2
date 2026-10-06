"""Reference fixtures for parity tests of the Swift port.

Runs the upstream `EmbeddingGemma2Processor` (transformers >= 5.19) for preprocessing and the
mlx-vlm reference model (pinned revision 3d87e884, BF16) for the forward pass, then writes
processor outputs, per-tower features and final embeddings to
Tests/Fixtures/*.safetensors (+ a small JSON index, test media). Generated, not checked in.

    cd Tools && uv run gen_fixtures.py
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from safetensors.numpy import save_file
from transformers import AutoProcessor
from transformers.video_processing_utils import VideoMetadata

from mlx_vlm.embedding_loader import load_embedding_model
from mlx_vlm.utils import get_model_path

MODEL_ID = "mlx-community/embeddinggemma-2-bf16"
OUT = Path(__file__).parent.parent / "Tests/Fixtures"

QUERY = "task: search result | query: "
DOCUMENT = "title: none | text: "
TEXTS = [
    QUERY + "Which planet is known as the Red Planet?",
    DOCUMENT + "Venus is often called Earth's twin because of its similar size and proximity.",
    DOCUMENT + "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    DOCUMENT + "Проект — это форма жизни.",
    DOCUMENT + "デザインは二十世紀の現象である。",
    QUERY + "¿Dónde está la biblioteca más cercana?",
]


def synthetic_image(height: int, width: int, seed: int) -> np.ndarray:
    """Smooth gradients + shapes: deterministic, compressible, and resize-friendly."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    rng = np.random.RandomState(seed)
    img = np.stack(
        [
            128 + 100 * np.sin(xx / (20 + 10 * rng.rand())),
            128 + 100 * np.cos(yy / (25 + 10 * rng.rand())),
            255 * (xx + yy) / (height + width),
        ],
        axis=-1,
    )
    cy, cx, r = height * 0.5, width * 0.4, min(height, width) * 0.25
    img[(yy - cy) ** 2 + (xx - cx) ** 2 < r**2] = [230, 40, 40]
    return np.clip(img, 0, 255).astype(np.uint8)


def synthetic_audio(seconds: float, seed: int) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000
    rng = np.random.RandomState(seed)
    chirp = np.sin(2 * np.pi * (220 + 300 * t) * t)
    return (0.3 * chirp + 0.02 * rng.randn(t.size)).astype(np.float32)


def moving_frame(i: int, height: int = 360, width: int = 640) -> np.ndarray:
    """A red disc sweeping across a smooth gradient; distinct content per second."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    img = np.stack([128 + 90 * np.sin(xx / 37 + i / 7), 128 + 90 * np.cos(yy / 29), 255 * xx / width], axis=-1)
    cx, cy = 60 + i * 26, height / 2 + 80 * np.sin(i / 3)
    img[(yy - cy) ** 2 + (xx - cx) ** 2 < 60**2] = [230, 40, 40]
    return np.clip(img, 0, 255).astype(np.uint8)


def write_clip(path: Path, frames: list, fps: int):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "12"}
        for f in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")))
        container.mux(stream.encode())


def decode_clip(path: Path):
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(stream)])
        fps = float(stream.average_rate)
    meta = VideoMetadata(
        total_num_frames=len(frames), fps=fps, duration=len(frames) / fps,
        height=frames.shape[1], width=frames.shape[2],
    )
    return frames, meta


def to_mx(inputs):
    return {k: mx.array(v) for k, v in inputs.items() if isinstance(v, np.ndarray)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = get_model_path(MODEL_ID)
    proc = AutoProcessor.from_pretrained(path)
    model = load_embedding_model(path)
    index: dict = {"model": MODEL_ID, "cases": {}}

    def run(name, inputs, extra=None):
        out = model(**to_mx(inputs))
        emb = np.array(out.text_embeds.astype(mx.float32))
        tensors = {k: np.ascontiguousarray(v) for k, v in inputs.items() if isinstance(v, np.ndarray)}
        tensors = {k: v.astype(np.int32) if v.dtype == np.int64 else v for k, v in tensors.items()}
        tensors["text_embeds"] = emb
        tensors.update(extra or {})
        save_file(tensors, str(OUT / f"{name}.safetensors"))
        index["cases"][name] = {k: list(v.shape) for k, v in tensors.items()}
        print(f"{name}: " + ", ".join(f"{k}{list(v.shape)}" for k, v in tensors.items()))
        return emb

    # Text (padded batch) + retrieval smoke test.
    text_inputs = proc.tokenizer(TEXTS, padding=True, return_tensors="np")
    emb = run("text", dict(text_inputs))
    scores = emb[:1] @ emb[1:3].T
    print("venus/mars scores:", scores)
    index["texts"] = TEXTS
    index["venus_mars_scores"] = scores[0].tolist()
    singles = np.stack([run_single(model, proc, t) for t in TEXTS])
    print("batched vs single min cosine: %.6f" % (emb * singles).sum(-1).min())

    # Image only (landscape) and text + image (portrait), via the chat template.
    land = synthetic_image(480, 640, 0)
    port = synthetic_image(700, 300, 1)
    Image.fromarray(land).save(OUT / "image_landscape.png")
    Image.fromarray(port).save(OUT / "image_portrait.png")
    img_inputs = proc(images=[[Image.fromarray(land)]], return_tensors="np")
    vis = model.get_image_features(mx.array(img_inputs["pixel_values"]), mx.array(img_inputs["image_position_ids"]))
    run("image", dict(img_inputs), {"image_features": np.array(vis.astype(mx.float32))})

    conv = [[
        {"role": "system", "content": DOCUMENT},
        {"role": "user", "content": [{"type": "text", "text": "A red circle"}, {"type": "image", "image": Image.fromarray(port)}]},
    ]]
    ti = proc.apply_chat_template(conv, tokenize=True, return_dict=True, return_tensors="np")
    run("text_image", dict(ti))

    # Audio only.
    wav = synthetic_audio(2.37, 0)
    au = proc(audio=[wav], return_tensors="np")
    feats, _ = model.audio_tower(mx.array(au["input_features"]), ~mx.array(au["input_features_mask"]).astype(mx.bool_))
    run("audio", dict(au), {
        "waveform": wav,
        "audio_tower_out": np.array(feats.astype(mx.float32)),
        "audio_features": np.array(model.get_audio_features(mx.array(au["input_features"]), mx.array(au["input_features_mask"])).astype(mx.float32)),
    })

    # Video from decoded frames (3 s at 2 fps -> 1 fps sampling keeps frames 0, 2, 4).
    frames = np.stack([synthetic_image(360, 640, 10 + i) for i in range(6)])
    meta = VideoMetadata(total_num_frames=6, fps=2.0, duration=3.0, height=360, width=640)
    vi = proc(videos=[frames], videos_kwargs={"video_metadata": [meta]}, return_tensors="np")
    run("video", dict(vi), {"frames": frames})

    # Real video file: H.264 4 s at 5 fps, decoded with PyAV (frames + metadata as transformers' loader
    # passes them) -> 1 fps sampling keeps frames 0, 5, 10, 15.
    clip_path = OUT / "video_clip.mp4"
    write_clip(clip_path, [moving_frame(i) for i in range(20)], fps=5)
    decoded, meta = decode_clip(clip_path)
    vf = proc(videos=[decoded], videos_kwargs={"video_metadata": [meta]}, return_tensors="np")
    run("video_file", dict(vf), {"sampled_frames": decoded[[0, 5, 10, 15]]})

    # Heterogeneous batch: text, image, audio in one padded batch.
    hb = proc(
        text=[DOCUMENT + "a short sentence", "<|image|>", "<|audio|>"],
        images=[[], [Image.fromarray(land)], []],
        audio=[wav],
        return_tensors="np",
    )
    run("hetero_batch", dict(hb))

    (OUT / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))
    print("wrote", OUT)


def run_single(model, proc, text):
    inputs = proc.tokenizer([text], return_tensors="np")
    return np.array(model(**to_mx(dict(inputs))).text_embeds.astype(mx.float32))[0]


if __name__ == "__main__":
    main()
