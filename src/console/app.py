"""FedShield hospital demo console — upload desks + isolation proof."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, Response

from src.common.data_ingest import load_csv_xy, partition_to_csv, summarize_partition, write_partition
from src.common.tenants import hospital_ids, load_hospitals
from src.console.isolation import collect_proof, restart_hospital
from src.console.templates import PAGE

app = Flask(__name__)

ROOT = Path(os.getenv("HOSPITALS_ROOT", "data/hospitals")).resolve()
INPUT_DIM = int(os.getenv("INPUT_DIM", "784"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "10"))


def _docker():
    try:
        import docker

        return docker.from_env()
    except Exception:  # noqa: BLE001
        return None


def _dir(hospital_id: str) -> Path:
    if hospital_id not in hospital_ids():
        abort(404)
    path = (ROOT / hospital_id).resolve()
    if ROOT not in path.parents and path != ROOT:
        abort(400)
    path.mkdir(parents=True, exist_ok=True)
    return path


@app.get("/")
def index():
    cards = []
    for hosp in load_hospitals():
        summary = summarize_partition(_dir(hosp["id"]), NUM_CLASSES)
        cards.append({**hosp, "data": summary})
    return render_template_string(
        PAGE,
        hospitals=cards,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
    )


@app.get("/api/hospitals")
def api_hospitals():
    cards = []
    for hosp in load_hospitals():
        cards.append({**hosp, "data": summarize_partition(_dir(hosp["id"]), NUM_CLASSES)})
    return jsonify(cards)


@app.post("/api/upload/<hospital_id>")
def api_upload(hospital_id: str):
    dest = _dir(hospital_id)
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"ok": False, "message": "CSV seçin (özellik sütunları + etiket)"}), 400
    tmp = dest / "_upload.csv"
    upload.save(tmp)
    try:
        features, labels = load_csv_xy(tmp, INPUT_DIM)
        write_partition(dest, features, labels)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "message": str(exc)}), 400
    finally:
        if tmp.exists():
            tmp.unlink()
    restarted = False
    client = _docker()
    if client is not None:
        try:
            restart_hospital(client, hospital_id)
            restarted = True
        except Exception:  # noqa: BLE001
            restarted = False
    return jsonify(
        {
            "ok": True,
            "hospital": hospital_id,
            "restarted": restarted,
            "data": summarize_partition(dest, NUM_CLASSES),
        }
    )


@app.get("/api/sample/<hospital_id>")
def api_sample(hospital_id: str):
    dest = _dir(hospital_id)
    try:
        body = partition_to_csv(dest)
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 404
    return Response(
        body,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={hospital_id}.csv"},
    )


@app.get("/api/proof")
def api_proof():
    return jsonify(collect_proof(_docker()))


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8500")), debug=False)


if __name__ == "__main__":
    main()
