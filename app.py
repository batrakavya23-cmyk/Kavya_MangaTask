"""
Gradio demo for Kavya_MangaTask: 3 manga pages -> JSON script with speaker labels.
Put this file in the REPO ROOT (next to src/ and models/).
"""
import json
import sys
import tempfile
from pathlib import Path

import gradio as gr
import joblib
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from inference import load_magi, load_paddle_ocr, predict_sequence  # noqa: E402

MODELS = ROOT / "models"
device = "cuda" if torch.cuda.is_available() else "cpu"


def find(pattern, what):
    hits = sorted(MODELS.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"No {what} ({pattern}) found in {MODELS}. "
            "Download the release artifacts into that folder first."
        )
    return hits[0]


# ---- load everything ONCE at startup -------------------------------------
from ultralytics import YOLO  # noqa: E402

try:
    yolo_path = find("*.pt", "YOLO bubble detector")
except FileNotFoundError:
    from huggingface_hub import hf_hub_download
    MODELS.mkdir(exist_ok=True)
    yolo_path = Path(hf_hub_download(
        repo_id="ogkalu/comic-speech-bubble-detector-yolov8m",
        filename="comic-speech-bubble-detector.pt",
        local_dir=str(MODELS),
    ))

print("Loading models on", device)
yolo = YOLO(str(yolo_path))
ocr = load_paddle_ocr()
magi = load_magi(device)
clf = joblib.load(find("*.joblib", "speaker random forest"))
assert hasattr(clf, "predict_proba")
print("Models ready.")


def run(p1, p2, p3):
    if not (p1 and p2 and p3):
        raise gr.Error("Please upload all 3 pages, in reading order.")
    result = predict_sequence("demo", [p1, p2, p3], yolo, ocr, magi, clf)

    lines = []
    for i, page in enumerate(result["pages"], 1):
        lines.append(f"--- Page {i} ---")
        if not page:
            lines.append("(no dialogue detected)")
        lines += [f"[{d['speaker']}] {d['text']}" for d in page]

    out = Path(tempfile.mkdtemp()) / "output.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return "\n".join(lines), result, str(out)


with gr.Blocks(title="Manga Pages to JSON Script") as demo:
    gr.Markdown(
        "# Manga Pages → JSON Script\n"
        "Upload **3 consecutive pages in reading order**. The pipeline detects speech "
        "bubbles (YOLO), reads text (PaddleOCR), finds characters (MAGI v2) and groups "
        "lines by speaker (Random Forest + clustering).\n\n"
        "Speakers are anonymous (`character1`, `character2`, ...) and text outside "
        "bubbles is `NARRATION`. Runs take a minute or two."
    )
    with gr.Row():
        i1 = gr.Image(type="filepath", label="Page 1")
        i2 = gr.Image(type="filepath", label="Page 2")
        i3 = gr.Image(type="filepath", label="Page 3")
    btn = gr.Button("Convert", variant="primary")
    transcript = gr.Textbox(label="Readable transcript", lines=14)
    js = gr.JSON(label="JSON output")
    dl = gr.File(label="Download JSON")
    btn.click(run, [i1, i2, i3], [transcript, js, dl])

if __name__ == "__main__":
    demo.queue(max_size=4).launch(share=True)
