"""
Standalone inference pipeline extracted from bytemanga.ipynb.

Pipeline (per 3-page sequence):
  1. YOLO bubble detector + full-page PaddleOCR  -> OCR fragments kept only if
     they fall in a detected bubble/text box (cells 205-212)
  2. MAGI v2 (detection + associations)          -> text boxes, characters,
     text<->character links, per-page clusters, "essential text" flags (cell 235)
  3. MAGI crop embeddings for character boxes    -> L2-normalised vectors (cell 236)
  4. Merge OCR fragments into MAGI text boxes    -> integrated rows (cell 237)
  5. Pairwise same-speaker RandomForest + average-linkage clustering
     -> characterN / NARRATION labels (cell 284)
  6. Write submission.jsonl

Requires (trained earlier in the notebook, cell 232):
  models/comic-speech-bubble-detector.pt        (ogkalu/comic-speech-bubble-detector-yolov8m)
  models/speaker_pair_random_forest.joblib

Usage:
  python inference.py --data "/path/to/BYTE ML Task Data" \
                      --models ./models --out submission.jsonl [--split test]
"""
import argparse
import importlib
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

CLUSTER_SIM_THRESHOLD = 0.45   # selected on validation (cell 284)
YOLO_CONF = 0.25
YOLO_IMGSZ = 640
MAGI_ID = "ragavsachdeva/magiv2"


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    return value


def box_area(box):
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def overlap_fraction(inner, outer):
    ix1 = max(float(inner[0]), float(outer[0]))
    iy1 = max(float(inner[1]), float(outer[1]))
    ix2 = min(float(inner[2]), float(outer[2]))
    iy2 = min(float(inner[3]), float(outer[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(box_area(inner), 1.0)


def center_inside(inner, outer):
    cx = (float(inner[0]) + float(inner[2])) / 2
    cy = (float(inner[1]) + float(inner[3])) / 2
    return (float(outer[0]) <= cx <= float(outer[2])
            and float(outer[1]) <= cy <= float(outer[3]))


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_yolo(models_dir):
    from ultralytics import YOLO
    return YOLO(str(Path(models_dir) / "comic-speech-bubble-detector.pt"))


def load_paddle_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang="en",
        device="cpu",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def load_magi(device):
    """MAGI v2 with internal OCR disabled, plus the transformers-5 shims
    that were needed in the notebook (cell 218)."""
    from transformers import AutoConfig, AutoModel
    from transformers.models.conditional_detr.configuration_conditional_detr import (
        ConditionalDetrConfig,
    )
    from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig

    config = AutoConfig.from_pretrained(MAGI_ID, trust_remote_code=True)
    config.disable_ocr = True
    config.disable_detections = False
    config.disable_crop_embeddings = False

    det = config.detection_model_config.to_dict()
    det.pop("model_type", None)
    config.detection_model_config = ConditionalDetrConfig.from_dict(det)

    crop = config.crop_embedding_model_config.to_dict()
    crop.pop("model_type", None)
    config.crop_embedding_model_config = ViTMAEConfig.from_dict(crop)

    module_name = config.__class__.__module__.rsplit(".", 1)[0] + ".modelling_magiv2"
    Magiv2Model = importlib.import_module(module_name).Magiv2Model
    if not hasattr(Magiv2Model, "all_tied_weights_keys"):
        Magiv2Model.all_tied_weights_keys = {}

    magi = AutoModel.from_pretrained(
        MAGI_ID,
        config=config,
        trust_remote_code=True,
        use_safetensors=False,
        low_cpu_mem_usage=True,
    )
    return magi.eval().to(device)


# --------------------------------------------------------------------------
# Step 1: YOLO + full-page PaddleOCR -> OCR fragments
# --------------------------------------------------------------------------
def ocr_page(ocr, image_path):
    outputs = ocr.predict(input=str(image_path))
    if not outputs:
        return []
    data = outputs[0].json
    if isinstance(data, str):
        data = json.loads(data)
    data = data.get("res", data)

    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    polys = data.get("dt_polys", [])

    items = []
    for i in range(min(len(texts), len(scores), len(polys))):
        text = str(texts[i]).strip()
        if not text:
            continue
        xs = [float(p[0]) for p in polys[i]]
        ys = [float(p[1]) for p in polys[i]]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        items.append({
            "text": text,
            "score": round(float(scores[i]), 4),
            "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center": [(x1 + x2) / 2, (y1 + y2) / 2],
        })
    return items


def assign_ocr_to_boxes(detections, ocr_items):
    """Each OCR fragment goes to at most one detector box; fragments outside
    every detected box are dropped (YOLO acts as a filter)."""
    assigned = [[] for _ in detections]
    for item in ocr_items:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        cx, cy = item["center"]
        text_area = max(1.0, (x2 - x1) * (y2 - y1))

        containing, overlaps = [], []
        for j, det in enumerate(detections):
            bx1, by1, bx2, by2 = det["bbox_xyxy"]
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                containing.append((max(1.0, (bx2 - bx1) * (by2 - by1)), j))
                continue
            ix1, iy1 = max(x1, bx1), max(y1, by1)
            ix2, iy2 = min(x2, bx2), min(y2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            ratio = inter / text_area
            if ratio >= 0.20:
                overlaps.append((ratio, j))

        if containing:
            assigned[min(containing)[1]].append(item)      # smallest containing box
        elif overlaps:
            assigned[max(overlaps)[1]].append(item)        # largest overlap
    return assigned


def page_ocr_parts(yolo, ocr, image_path):
    """Returns a flat list of OCR fragments {text, score, bbox} on one page."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]

    result = yolo.predict(source=str(image_path), conf=YOLO_CONF,
                          imgsz=YOLO_IMGSZ, verbose=False)[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            detections.append({"bbox_xyxy": [x1, y1, x2, y2]})

    assigned = assign_ocr_to_boxes(detections, ocr_page(ocr, image_path))
    parts = []
    for group in assigned:
        for p in group:
            parts.append({"text": p["text"], "score": float(p["score"]),
                          "bbox": p["bbox_xyxy"]})
    return parts


# --------------------------------------------------------------------------
# Steps 2-3: MAGI detection/associations + character embeddings
# --------------------------------------------------------------------------
@torch.inference_mode()
def run_magi(magi, images):
    results = magi.predict_detections_and_associations(images)
    assert len(results) == len(images)
    pages = [json_safe(r) for r in results]

    boxes_per_page = [p["characters"] for p in pages]
    embs = magi.predict_crop_embeddings(images, boxes_per_page, batch_size=64)
    for page, boxes, e in zip(pages, boxes_per_page, embs):
        e = F.normalize(e.detach().float(), p=2, dim=1).cpu().numpy()
        assert len(e) == len(boxes)
        page["embeddings"] = e
    return pages


# --------------------------------------------------------------------------
# Step 4: merge OCR fragments into MAGI text boxes
# --------------------------------------------------------------------------
def integrate(sid, magi_pages, ocr_parts_by_page):
    rows = []
    for page_index, page in enumerate(magi_pages, start=1):
        text_boxes = page.get("texts", [])
        characters = page.get("characters", [])
        clusters = page.get("character_cluster_labels", [])
        essential_flags = page.get("is_essential_text", [])

        char_for_text = defaultdict(list)
        for t, c in page.get("text_character_associations", []):
            char_for_text[int(t)].append(int(c))
        tails_for_text = defaultdict(list)
        for t, tl in page.get("text_tail_associations", []):
            tails_for_text[int(t)].append(int(tl))

        assigned = [[] for _ in text_boxes]
        for part in ocr_parts_by_page[page_index]:
            options = []
            for ti, tb in enumerate(text_boxes):
                ov = overlap_fraction(part["bbox"], tb)
                inside = center_inside(part["bbox"], tb)
                if inside or ov >= 0.15:
                    options.append((int(inside), ov, -box_area(tb), ti))
            if options:
                assigned[max(options)[3]].append(part)

        for ti, bbox in enumerate(text_boxes):
            parts = sorted(
                assigned[ti],
                key=lambda p: ((p["bbox"][1] + p["bbox"][3]) / 2,
                               (p["bbox"][0] + p["bbox"][2]) / 2),
            )
            text = " ".join(p["text"].strip() for p in parts if p["text"].strip())
            text = " ".join(text.split())

            idxs = sorted(set(char_for_text.get(ti, [])))
            assoc = [{
                "character_index": ci,
                "bbox_xyxy": characters[ci] if ci < len(characters) else None,
                "page_cluster": clusters[ci] if ci < len(clusters) else None,
            } for ci in idxs]

            rows.append({
                "sequence_id": sid,
                "page_index": page_index,
                "text_region_index": ti,
                "ocr_text": text,
                "ocr_confidence_mean": (sum(p["score"] for p in parts) / len(parts)
                                        if parts else None),
                "magi_character_associations": assoc,
                "magi_tail_indices": tails_for_text.get(ti, []),
                "magi_is_essential_text": (bool(essential_flags[ti])
                                           if ti < len(essential_flags) else None),
            })
    return rows


# --------------------------------------------------------------------------
# Step 5: pairwise features, RandomForest, clustering
# --------------------------------------------------------------------------
def normalize_text(s):
    return re.sub(r"\s+", " ", str(s or "").upper()).strip()


def tokens(s):
    return re.findall(r"\b\w+\b", normalize_text(s))


def text_jaccard(a, b):
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def text_sequence_ratio(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def text_length_ratio(a, b):
    na, nb = len(tokens(a)), len(tokens(b))
    return min(na, nb) / max(na, nb) if max(na, nb) else 1.0


def get_character_ids(row):
    return {int(a["character_index"])
            for a in row.get("magi_character_associations", [])
            if a.get("character_index") is not None}


def get_character_vectors(row, seq_emb):
    page_emb = seq_emb.get(int(row["page_index"]), {})
    return [page_emb[c] for c in sorted(get_character_ids(row)) if c in page_emb]


def pair_features(a, b, seq_emb):
    ta, tb = str(a.get("ocr_text") or ""), str(b.get("ocr_text") or "")
    pa, pb = int(a["page_index"]), int(b["page_index"])
    same_page = pa == pb
    ca, cb = get_character_ids(a), get_character_ids(b)

    cos = [float(np.dot(va, vb))
           for va in get_character_vectors(a, seq_emb)
           for vb in get_character_vectors(b, seq_emb)]

    same_character = float(bool(ca & cb)) if (same_page and ca and cb) else 0.0

    same_page_cluster = 0.0
    if same_page:
        cla = {x.get("page_cluster") for x in a["magi_character_associations"]
               if x.get("page_cluster") is not None}
        clb = {x.get("page_cluster") for x in b["magi_character_associations"]
               if x.get("page_cluster") is not None}
        if cla and clb:
            same_page_cluster = 1.0 if cla & clb else -1.0

    conf_a = float(a.get("ocr_confidence_mean") or 0.0)
    conf_b = float(b.get("ocr_confidence_mean") or 0.0)

    return [
        float(same_page), float(abs(pa - pb)),
        text_jaccard(ta, tb), text_sequence_ratio(ta, tb), text_length_ratio(ta, tb),
        float(bool(ca)), float(bool(cb)),
        float(bool(cos)), max(cos) if cos else 0.0, float(np.mean(cos)) if cos else 0.0,
        same_character, same_page_cluster,
        float(bool(a.get("magi_is_essential_text"))),
        float(bool(b.get("magi_is_essential_text"))),
        conf_a, conf_b, abs(conf_a - conf_b),
    ]


def build_seq_embeddings(magi_pages):
    return {pi: {ci: v for ci, v in enumerate(page["embeddings"])
                 if np.linalg.norm(v) > 0}
            for pi, page in enumerate(magi_pages, start=1)}


def cluster_speakers(clf, candidates, seq_emb):
    if not candidates:
        return {}

    # Text regions with no associated character are narration.
    attributed_idx = [i for i, r in enumerate(candidates) if get_character_ids(r)]
    speakers = {i: "NARRATION" for i, r in enumerate(candidates)
                if not get_character_ids(r)}
    if not attributed_idx:
        return speakers

    attributed = [candidates[i] for i in attributed_idx]
    n = len(attributed)
    if n == 1:
        speakers[attributed_idx[0]] = "character1"
        return speakers

    pairs, feats = [], []
    for i in range(n):
        for j in range(i + 1, n):
            feats.append(pair_features(attributed[i], attributed[j], seq_emb))
            pairs.append((i, j))

    proba = clf.predict_proba(np.asarray(feats, dtype=np.float32))
    pos = proba[:, list(clf.classes_).index(1)]

    sim = np.eye(n, dtype=np.float32)
    for (i, j), p in zip(pairs, pos):
        sim[i, j] = sim[j, i] = float(p)

    dist = np.clip(1.0 - sim, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    labels = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - CLUSTER_SIM_THRESHOLD,
    ).fit_predict(dist)

    first_seen = {}
    for local, c in enumerate(labels):
        first_seen.setdefault(int(c), local)
    name = {c: f"character{k + 1}"
            for k, c in enumerate(sorted(first_seen, key=first_seen.get))}

    for local, glob in enumerate(attributed_idx):
        speakers[glob] = name[int(labels[local])]
    return speakers


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def predict_sequence(sid, image_paths, yolo, ocr, magi, clf):
    # OCR (per page)
    ocr_parts = {i: page_ocr_parts(yolo, ocr, p)
                 for i, p in enumerate(image_paths, start=1)}

    # MAGI (all 3 pages together)
    images = [np.array(Image.open(p).convert("RGB")) for p in image_paths]
    magi_pages = run_magi(magi, images)

    rows = integrate(sid, magi_pages, ocr_parts)
    candidates = sorted(
        (r for r in rows
         if r["ocr_text"].strip() and r["magi_is_essential_text"] is True),
        key=lambda r: (r["page_index"], r["text_region_index"]),
    )

    speakers = cluster_speakers(clf, candidates, build_seq_embeddings(magi_pages))

    pages = [[], [], []]
    for i, r in enumerate(candidates):
        pages[r["page_index"] - 1].append({
            "speaker": speakers.get(i, "NARRATION"),
            "text": r["ocr_text"].strip(),
        })
    return {"sequence_id": sid, "pages": pages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help='"BYTE ML Task Data" folder')
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="submission.jsonl")
    ap.add_argument("--split", default="test", help="split name in sequences.json")
    args = ap.parse_args()

    data, models = Path(args.data), Path(args.models)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = json.loads((data / "sequences.json").read_text(encoding="utf-8"))
    todo = sorted((m for m in metadata
                   if str(m.get("split", "")).lower() == args.split),
                  key=lambda m: m["sequence_id"])

    yolo = load_yolo(models)
    ocr = load_paddle_ocr()
    magi = load_magi(device)
    clf = joblib.load(models / "speaker_pair_random_forest.joblib")
    assert hasattr(clf, "predict_proba")

    with open(args.out, "w", encoding="utf-8") as f:
        for k, meta in enumerate(todo, start=1):
            paths = [data / rel for rel in meta["images"]]
            assert len(paths) == 3 and all(p.exists() for p in paths)
            item = predict_sequence(meta["sequence_id"], paths, yolo, ocr, magi, clf)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"{k}/{len(todo)} {meta['sequence_id']}")

    print("Wrote", args.out)


if __name__ == "__main__":
    main()
