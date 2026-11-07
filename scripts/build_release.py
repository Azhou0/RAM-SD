"""Create a deterministic source-only archive; datasets and runtime artifacts are excluded."""

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
directories = ("src", "configs", "docs", "scripts", "tests", ".github")
names = ("README.md", "LICENSE", "CITATION.cff", "pyproject.toml", "uv.lock", "requirements.lock.txt", ".python-version", ".gitignore", ".env.example")
files = [ROOT / name for name in names if (ROOT / name).is_file()]
files += [p for name in directories for p in (ROOT / name).rglob("*")
          if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".csv", ".pkl", ".npy", ".faiss")]
files += [ROOT / "references" / "source_manifest.json"]
files = sorted(set(files))
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
destination = ROOT / "dist" / "ram-sd-repro.zip"
destination.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for p in files:
        info = zipfile.ZipInfo("ram-sd-repro/" + str(p.relative_to(ROOT)), date_time=(2020, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, p.read_bytes())
    info = zipfile.ZipInfo("ram-sd-repro/RELEASE_MANIFEST.json", date_time=(2020, 1, 1, 0, 0, 0))
    archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
destination.with_suffix(".zip.sha256").write_text(checksum + "  " + destination.name + "\n")
print(destination)
print("sha256=" + checksum)
