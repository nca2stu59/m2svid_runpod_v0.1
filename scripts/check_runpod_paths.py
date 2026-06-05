from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from run_pipeline import DEFAULT_M2SVID_PYTHON, DEFAULT_M2SVID_SERVICE, venv_python


def status(label: str, path: Path, required: bool = True) -> str:
    if path.exists():
        mark = "OK"
    else:
        mark = "MISS" if required else "INFO"
    return f"{mark} {label}: {path}"


def bin_status(label: str, name: str) -> tuple[str, bool]:
    found = shutil.which(name)
    if found:
        return f"OK {label}: {found}", True
    return f"MISS {label}: {name}", False


def main() -> int:
    service = DEFAULT_M2SVID_SERVICE
    default_output = (
        Path("/workspace/outputs/m2svid_runpod_v0.1")
        if os.name != "nt" else APP_ROOT / "outputs"
    )
    checks = [
        ("service", service, True),
        ("app/m2svid python", DEFAULT_M2SVID_PYTHON, True),
        ("vda python", venv_python(service, ".venv-vda"), True),
        ("m2svid weights", service / "ckpts" / "m2svid_weights.pt", True),
        ("open_clip weights", service / "ckpts" / "open_clip_pytorch_model.bin", True),
        ("autoshot weights", service / "ckpts" / "autoshot.pth", True),
        ("m2svid config", service / "configs" / "m2svid.yaml", True),
        ("VDA repo", service / "third_party" / "Video-Depth-Anything", True),
        ("vendored autoshot splitter", APP_ROOT / "vendored" / "autoshot" / "autoshot_splitter.py", True),
        ("output root", Path(os.environ.get("M2SVID_OUTPUT_ROOT", default_output)), False),
    ]
    missing_required = 0
    for label, path, required in checks:
        print(status(label, path, required=required))
        if required and not path.exists():
            missing_required += 1
    for label, name in [("ffmpeg", "ffmpeg"), ("ffprobe", "ffprobe")]:
        line, ok = bin_status(label, name)
        print(line)
        if not ok:
            missing_required += 1
    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
