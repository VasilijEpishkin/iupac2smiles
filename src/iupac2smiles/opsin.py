"""OPSIN backend: rule-based IUPAC parser (Java CLI, https://github.com/dan2097/opsin).

Params (``converter_params["opsin"]``), all bool:
    allow_radicals               -r  interpret radicals ("ethyl")
    wildcard_radicals            -w  output radicals as * in SMILES
    detailed_failure_analysis    -f  better error messages
    allow_acids_without_acid     -a  interpret "acetic" without "acid"
    allow_uninterpretable_stereo -s  ignore unresolvable stereochemistry

Env: ``OPSIN_JAR_PATH`` - use a local jar instead of downloading one.
"""

from __future__ import annotations

import os
import subprocess

from ._download import fetch

JAR_URL = "https://github.com/dan2097/opsin/releases/download/2.9.0/opsin-cli-2.9.0-jar-with-dependencies.jar"
JAR_SHA256 = "c2e29326c281f87b59a05d934d8589adac6e9d17b95b984931b3e739111b360f"

_FLAGS = {
    "allow_radicals": "-r",
    "wildcard_radicals": "-w",
    "detailed_failure_analysis": "-f",
    "allow_acids_without_acid": "-a",
    "allow_uninterpretable_stereo": "-s",
}


def _jar_path() -> str:
    env = os.environ.get("OPSIN_JAR_PATH")
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(f"OPSIN_JAR_PATH points to a missing file: {env}")
        return env
    return str(fetch(JAR_URL, JAR_SHA256))


def iupac_to_smiles(names: list[str], *, params: dict | None = None) -> list[str]:
    """One JVM call for the whole batch; OPSIN prints one line per input (empty on failure)."""
    if not names:
        return []
    p = params or {}
    cmd = ["java", "-jar", _jar_path(), "-o", "smi", *(flag for key, flag in _FLAGS.items() if p.get(key))]
    # One name per line: a newline inside a name would shift every following result.
    payload = "".join(" ".join(n.split()) + "\n" for n in names).encode()

    proc = subprocess.run(cmd, input=payload, capture_output=True, timeout=60 + 0.5 * len(names))
    if proc.returncode != 0:
        raise RuntimeError(f"OPSIN exited with {proc.returncode}: {proc.stderr.decode(errors='replace')[-500:]}")

    lines = proc.stdout.decode(errors="replace").splitlines()
    if len(lines) != len(names):
        raise RuntimeError(f"OPSIN returned {len(lines)} lines for {len(names)} names")
    return [line.strip() for line in lines]
