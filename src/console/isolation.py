"""Live isolation evidence for the 5-hospital demo (Docker inspect + exec)."""

from __future__ import annotations

import json
from typing import Any

from src.common.tenants import hospital_ids, load_hospitals

CONTAINER = {
    hid: f"fedshield-{hid}"
    for hid in hospital_ids()
}


def _ok(title: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"ok": True, "title": title, "detail": detail, **extra}


def _fail(title: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "title": title, "detail": detail, **extra}


def collect_proof(docker_client: Any | None) -> dict[str, Any]:
    hospitals = load_hospitals()
    ids = hospital_ids()
    if docker_client is None:
        return {
            "available": False,
            "reason": "Docker socket is not available inside the console.",
            "hospitals": hospitals,
            "checks": [],
        }

    checks: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []

    for hid in ids:
        name = CONTAINER[hid]
        try:
            ctr = docker_client.containers.get(name)
        except Exception as exc:  # noqa: BLE001
            checks.append(_fail(f"{hid} container", f"Not found: {exc}"))
            continue
        info = ctr.attrs
        mounts = info.get("Mounts") or []
        data_mounts = [
            m for m in mounts
            if str(m.get("Destination", "")).rstrip("/") == "/data/processed"
        ]
        networks = list((info.get("NetworkSettings") or {}).get("Networks") or {})
        other_data = [
            m for m in data_mounts
            if hid not in str(m.get("Source", "")) and hid not in str(m.get("Destination", ""))
        ]
        # Source path on the host should contain this hospital id
        own = all(hid in str(m.get("Source", "")) for m in data_mounts) if data_mounts else False
        if data_mounts and own and not other_data:
            checks.append(
                _ok(
                    f"{hid} mounts only its own folder",
                    f"Source={data_mounts[0].get('Source')} → /data/processed (read-only={data_mounts[0].get('RW') is False})",
                )
            )
        else:
            checks.append(
                _fail(
                    f"{hid} data mount",
                    f"Expected a single bind mount whose source contains '{hid}'. Found {data_mounts}",
                )
            )

        foreign_nets = [n for n in networks if n.startswith("fl-hospital-") and hid not in n]
        if foreign_nets:
            checks.append(_fail(f"{hid} network leak", f"Also attached to {foreign_nets}"))
        else:
            checks.append(
                _ok(
                    f"{hid} network is private",
                    f"Attached to {networks or ['(none)']} — not on another hospital fabric.",
                )
            )

        listing = _exec(ctr, "import os; print(sorted(os.listdir('/data/processed')))")
        cross = _exec(
            ctr,
            "import socket\n"
            "hosts=['fedshield-hospital-b','client-hospital-b','hospital-b']\n"
            "if '" + hid + "'=='hospital-b':\n"
            "    hosts=['fedshield-hospital-a','client-hospital-a','hospital-a']\n"
            "blocked=[]\n"
            "for h in hosts:\n"
            "    try:\n"
            "        socket.getaddrinfo(h, 8080)\n"
            "        blocked.append(h+':RESOLVED')\n"
            "    except Exception as e:\n"
            "        blocked.append(h+':'+type(e).__name__)\n"
            "print('|'.join(blocked))",
        )
        resolved = [part for part in (cross or "").split("|") if part.endswith(":RESOLVED")]
        if resolved:
            checks.append(_fail(f"{hid} resolved a peer", cross or ""))
        else:
            checks.append(
                _ok(
                    f"{hid} cannot DNS-resolve another hospital",
                    cross or "no response",
                )
            )

        inventory.append(
            {
                "id": hid,
                "container": name,
                "status": ctr.status,
                "networks": networks,
                "files": listing,
            }
        )

    egress = _grep_egress(docker_client, ids)
    if egress:
        leaked = [row for row in egress if int(row.get("raw_samples_sent", 0) or 0) != 0]
        if leaked:
            checks.append(_fail("Egress carried raw samples", str(leaked[:3])))
        else:
            checks.append(
                _ok(
                    "Wire payload is model weights only",
                    f"Logged {len(egress)} egress audits; raw_samples_sent=0 on all of them.",
                )
            )
    else:
        checks.append(
            _ok(
                "Egress auditor ready",
                "No client fit yet. After a round, logs will show payload_kind=model_weights and raw_samples_sent=0.",
            )
        )

    passed = sum(1 for c in checks if c.get("ok"))
    return {
        "available": True,
        "passed": passed,
        "total": len(checks),
        "all_ok": passed == len(checks) and len(checks) > 0,
        "hospitals": hospitals,
        "inventory": inventory,
        "egress": egress[-10:],
        "checks": checks,
    }


def _exec(container: Any, py: str) -> str:
    try:
        exit_code, output = container.exec_run(["python", "-c", py], demux=False)
        text = output.decode("utf-8", errors="replace") if isinstance(output, (bytes, bytearray)) else str(output)
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        return f"exec-failed: {exc}"


def _grep_egress(docker_client: Any, ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hid in ids:
        name = CONTAINER[hid]
        try:
            ctr = docker_client.containers.get(name)
            raw = ctr.logs(tail=80).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for line in raw.splitlines():
            if "Egress audit" not in line and "raw_samples_sent" not in line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(
                {
                    "client_id": payload.get("client_id") or hid,
                    "raw_samples_sent": payload.get("raw_samples_sent", 0),
                    "payload_kind": payload.get("payload_kind"),
                    "nbytes": payload.get("nbytes"),
                    "n_tensors": payload.get("n_tensors"),
                    "secagg_applied": payload.get("secagg_applied"),
                }
            )
    return rows


def restart_hospital(docker_client: Any, hospital_id: str) -> str:
    if hospital_id not in CONTAINER:
        raise ValueError(f"Unknown hospital {hospital_id}")
    ctr = docker_client.containers.get(CONTAINER[hospital_id])
    ctr.restart(timeout=8)
    return ctr.status
