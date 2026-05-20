# run with: uv run .\data\prepare_simulation_data.py

import re
from pathlib import Path
import json

import numpy as np


def parse_simulation_input(file_path: str | Path):
    """
    Parse Fortran namelist input.dat file and extract specific parameters.

    Returns dict with: rlt, rln (for mass=1.0 species), shat, q (from geom)
    """
    content = Path(file_path).read_text()

    # Extract geom section
    geom_match = re.search(r"&geom\s+(.*?)\s+/", content, re.DOTALL)
    if geom_match is None:
        raise ValueError("No &geom section found in the file.")
    geom = geom_match.group(1)

    shat_match = re.search(r"shat\s*=\s*([\d.eE+-]+)", geom)
    q_match = re.search(r"q\s*=\s*([\d.eE+-]+)", geom)
    if not shat_match or not q_match:
        raise ValueError("Missing shat or q in &geom section")
    shat = float(shat_match.group(1))
    q = float(q_match.group(1))

    # Find species section with mass = 1.0
    species_sections = re.findall(r"&species\s+(.*?)\s+/", content, re.DOTALL)
    for species in species_sections:
        if re.search(r"mass\s*=\s*1\.0\b", species):
            rlt_match = re.search(r"rlt\s*=\s*([\d.eE+-]+)", species)
            rln_match = re.search(r"rln\s*=\s*([\d.eE+-]+)", species)
            if not rlt_match or not rln_match:
                raise ValueError("Missing rlt or rln in mass=1.0 species section")
            rlt = float(rlt_match.group(1))
            rln = float(rln_match.group(1))
            break

    return {"shat": shat, "q": q, "rlt": rlt, "rln": rln}


def parse_flux(file_path: str | Path) -> np.ndarray:
    """
    Parse fluxes.dat file and extract energy flux column (index 1).

    File format: 3 columns of space-separated floats
    Returns: Energy flux values (second column) as numpy array
    """
    data = np.loadtxt(file_path)
    return data[:, 1]


def is_stable_simulation(flux: np.ndarray) -> bool:
    """Check if a gyrokinetic simulation is stable or not.

    Args:
        flux (np.ndarray): Energy flux array

    Returns:
        bool: True if the simulation is stable, False otherwise.
    """
    mean_head: float = float(np.mean(flux[:240]))
    mean_tail: float = float(np.mean(flux[-240:]))
    if 1.0 <= mean_head <= np.inf and 1.0 <= mean_tail <= np.inf:
        return False
    return True


if __name__ == "__main__":
    # Base directory containing flux data
    flux_dir = Path(__file__).parent / "flux" / "raw"

    # Find all directories matching batch_X and gyroswin_*
    namespaces = []
    for item in flux_dir.iterdir():
        if item.is_dir() and (
            item.name.startswith("batch_") or item.name.startswith("gyroswin_")
        ):
            namespaces.append(item)

    print(f"Found {len(namespaces)} namespaces to process")

    # Process each namespace
    results = {}
    stable_simulation_iterations = {}

    for namespace_path in sorted(namespaces):
        namespace_name = namespace_path.name
        print(f"Processing {namespace_name}...")

        results[namespace_name] = {}
        stable_simulation_iterations[namespace_name] = []

        # Find all input files in this namespace
        input_files = list(namespace_path.glob("input_*.dat"))

        for input_file in sorted(input_files):
            # Extract iteration number from filename
            iteration = int(input_file.stem.split("_")[1])  # e.g., "input_70.dat" -> 70

            # Find corresponding flux file
            flux_file = namespace_path / f"fluxes_{iteration}.dat"

            if not flux_file.exists():
                print(f"  Warning: Missing flux file for iteration {iteration}")
                continue

            try:
                # Parse both files
                params = parse_simulation_input(input_file)
                flux = parse_flux(flux_file)

                # Check stability
                if is_stable_simulation(flux):
                    stable_simulation_iterations[namespace_name].append(iteration)
                    continue

                # Store in results
                results[namespace_name][iteration] = {
                    "energy_flux": flux.tolist(),  # Convert numpy array to list for JSON
                    "rlt": params["rlt"],
                    "rln": params["rln"],
                    "q": params["q"],
                    "shat": params["shat"],
                }

            except Exception as e:
                print(f"  Error processing iteration {iteration}: {e}")
                continue

        print(f"  Processed {len(results[namespace_name])} iterations")

    # Save to JSON file
    output_file = Path(__file__).parent / "flux_data.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save stable simulation iterations to a separate file
    stable_file = Path(__file__).parent / "stable_simulations.json"
    with open(stable_file, "w") as f:
        json.dump(stable_simulation_iterations, f, indent=2)

    print("\n# Simulation Stability per Namespace:")
    for namespace in sorted(results.keys()):
        stable_count = len(stable_simulation_iterations.get(namespace, []))
        unstable_count = len(results.get(namespace, {}))
        total = stable_count + unstable_count
        print(
            f"  {namespace}: {unstable_count} unstable, {stable_count} stable (total: {total})"
        )

    print(f"\nData saved to {output_file}")
    print(f"Stable simulations saved to {stable_file}")
    print(f"\nTotal namespaces: {len(results)}")
    total_unstable = sum(len(v) for v in results.values())
    total_stable = sum(len(v) for v in stable_simulation_iterations.values())
    print(f"Total unstable iterations: {total_unstable}")
    print(f"Total stable iterations: {total_stable}")
    print(f"Total iterations: {total_unstable + total_stable}")
