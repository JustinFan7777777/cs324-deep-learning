"""Create the complete, reviewable assignment ZIP without changing originals.

The archive uses an explicit allow-list. It contains the intended revised
report and reproducible code/results, while excluding the superseded report,
raw training logs, caches, and course materials.
"""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORT = "CS324_Assignment1_Revised_Report"


def package(destination):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(
            f"Preserving existing archive: {destination}. "
            "Choose a new path with --output to create another package."
        )
    # Fail before writing an archive if the exact evidence or notebook is stale.
    verification = subprocess.run(
        [sys.executable, str(HERE / "verify_submission.py")],
        cwd=ROOT, capture_output=True, text=True,
    )
    if verification.returncode:
        raise RuntimeError("Submission verification failed:\n" + verification.stdout + verification.stderr)
    notebook = json.loads((HERE / "assignment1_revised.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if [cell.get("execution_count") for cell in code_cells] != list(range(1, len(code_cells) + 1)):
        raise ValueError("Execute the revised notebook before packaging.")
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    if any(output.get("output_type") == "error" for output in outputs):
        raise ValueError("The revised notebook contains an execution error.")
    if sum("image/png" in output.get("data", {}) for output in outputs) != 7:
        raise ValueError("Expected seven embedded notebook figures.")
    recorded_outputs = json.dumps(outputs)
    for name in ("perceptron.json", "mlp.json"):
        digest = hashlib.sha256((HERE / "results" / name).read_bytes()).hexdigest()
        if digest not in recorded_outputs:
            raise ValueError(f"Notebook results are stale: {name}")
    files = {
        "report.pdf": HERE / (REPORT + ".pdf"),
        "README.md": ROOT / "README.md",
    }
    for relative in (
        "Part_1/perceptron.py", "Part_1/experiments.py",
        "Part_2/modules.py", "Part_2/mlp_numpy.py", "Part_2/train_mlp_numpy.py",
    ):
        files[relative] = ROOT / relative
    for name in (
        "perceptron_experiments.py", "mlp_experiments.py", "make_notebook.py",
        "build_report.py", "package_submission.py", "verify_submission.py",
        "requirements.txt", "README.md",
        "assignment1_revised.ipynb", REPORT + ".tex",
    ):
        files["report_revision/" + name] = HERE / name
    for folder in ("results", "figures"):
        for path in sorted((HERE / folder).iterdir()):
            if path.suffix in (".json", ".npz", ".png", ".pdf"):
                files[f"report_revision/{folder}/{path.name}"] = path
    manifest = {
        "student_number": "12412307",
        "report": "report.pdf",
        "notebook": "report_revision/assignment1_revised.ipynb",
        "files_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(files.items())
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    # Reopen the archive and validate both ZIP integrity and every listed digest.
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name, expected in manifest["files_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    print(f"{destination}\n{len(files)+1} entries; {destination.stat().st_size:,} bytes; all SHA-256 checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "12412307_assignment1.zip",
        help="New ZIP path; existing archives are preserved",
    )
    package(parser.parse_args().output)
