"""Write requirements.lock.txt with the exact versions installed in the CURRENT environment.

Run it inside the environment you actually trained in:  python scripts/pin_requirements.py
"""
import re
from importlib import metadata
from pathlib import Path

root = Path(__file__).resolve().parents[1]
names = []
for line in (root / "requirements.txt").read_text().splitlines():
    line = line.split("#")[0].strip()
    if line:
        names.append(re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0])

pinned = []
for name in names:
    try:
        pinned.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"WARNING: {name} is not installed - skipped")

(root / "requirements.lock.txt").write_text("\n".join(pinned) + "\n")
print("Wrote requirements.lock.txt:\n" + "\n".join(pinned))
