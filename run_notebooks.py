"""Run all notebooks with correct CWD injected into the kernel."""
import sys
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).parent.resolve()
NOTEBOOKS = sorted(PROJECT_ROOT.glob("notebooks/*.ipynb"))

# Inject os.chdir as a silent startup cell
CHDIR_SOURCE = f'import os; os.chdir(r"{PROJECT_ROOT}")'


def run_notebook(nb_path: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"Running: {nb_path.name}")
    print('='*60)

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Prepend CWD cell (not saved back to file)
    chdir_cell = nbformat.v4.new_code_cell(CHDIR_SOURCE)
    chdir_cell["id"] = "startup-chdir-00"
    nb.cells.insert(0, chdir_cell)

    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )

    errors = []
    try:
        client.execute()
        print(f"  ✓ {nb_path.name} completed")
    except Exception as e:
        print(f"  ✗ {nb_path.name} FAILED: {e}")
        errors.append(str(e))

    # Collect outputs from real cells (skip injected cell)
    outputs_summary = []
    for cell in nb.cells[1:]:
        if cell.cell_type == "code" and cell.get("outputs"):
            for out in cell["outputs"]:
                if out.get("output_type") == "stream":
                    text = "".join(out.get("text", []))
                    if text.strip():
                        outputs_summary.append(text.strip())
                elif out.get("output_type") in ("execute_result", "display_data"):
                    data = out.get("data", {})
                    text = data.get("text/plain", "")
                    if text.strip():
                        outputs_summary.append(text.strip())

    # Save executed notebook (without the injected cell)
    nb.cells.pop(0)
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)

    return {"notebook": nb_path.name, "errors": errors, "outputs": outputs_summary}


if __name__ == "__main__":
    results = []
    for nb_path in NOTEBOOKS:
        result = run_notebook(nb_path)
        results.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for r in results:
        status = "✓" if not r["errors"] else "✗"
        print(f"  {status} {r['notebook']}")
        for line in r["outputs"][-5:]:  # last 5 output lines
            print(f"      {line[:120]}")
        if r["errors"]:
            print(f"      ERROR: {r['errors'][0][:200]}")
