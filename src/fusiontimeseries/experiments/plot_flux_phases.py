"""Plot a flux trace with annotations showing the three characteristic phases.

Run with: uv run src/fusiontimeseries/experiments/plot_flux_phases.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    """Create a plot showing the three phases of gyrokinetic flux evolution."""

    # Load flux data
    flux_data_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "flux_data.json"
    )

    print(f"Loading flux data from: {flux_data_path}")
    with open(flux_data_path, "r") as f:
        all_flux_data = json.load(f)

    # Get ID data (gyroswin_id would be good for this)
    id_data = all_flux_data.get("gyroswin_id", {})

    if not id_data:
        print("gyroswin_id not found, trying batch_1...")
        id_data = all_flux_data.get("batch_1", {})

    # Select a trace that shows clear phases
    # Let's look at a few and pick one with good characteristics
    print(f"\nAvailable simulations: {len(id_data)}")

    # Try simulation 3001 from gyroswin_id (often shows clear growth)
    sim_id = "3001"
    if sim_id not in id_data:
        sim_id = list(id_data.keys())[0]
        print(f"Using simulation {sim_id}")

    flux = np.array(id_data[sim_id]["energy_flux"])
    operating_params = {k: id_data[sim_id][k] for k in ["q", "shat", "rlt", "rln"]}

    print(f"Selected simulation {sim_id}")
    print(
        f"Operating parameters: q={operating_params['q']:.3f}, ŝ={operating_params['shat']:.3f}, "
        f"R/L_T={operating_params['rlt']:.3f}, R/L_n={operating_params['rln']:.3f}"
    )
    print(f"Flux length: {len(flux)} timesteps")
    print(f"Flux range: [{flux.min():.2f}, {flux.max():.2f}]")

    # Identify phases by inspection
    # Phase 1: Initial transient (typically first 50-100 timesteps where flux is near zero)
    # Phase 2: Linear growth (exponential growth phase)
    # Phase 3: Nonlinear saturation (flux fluctuates around a mean)

    # For automated detection, look for:
    # - End of phase 1: when flux starts to systematically grow
    # - End of phase 2: when growth rate decreases significantly

    # Simple heuristic:
    # Phase 1 ends when flux exceeds 10% of final mean
    # Phase 2 ends when flux reaches ~80% of final mean

    final_mean = np.mean(flux[-200:])  # Mean of last 200 timesteps

    # Find phase boundaries
    phase1_end = np.where(flux > 0.1 * final_mean)[0]
    if len(phase1_end) > 0:
        phase1_end = phase1_end[0]
    else:
        phase1_end = 50

    phase2_end = np.where(flux > 0.8 * final_mean)[0]
    if len(phase2_end) > 0:
        phase2_end = phase2_end[0]
    else:
        phase2_end = len(flux) // 2

    print("\nPhase boundaries:")
    print(f"Phase 1 (Transient): 0 - {phase1_end}")
    print(f"Phase 2 (Linear Growth): {phase1_end} - {phase2_end}")
    print(f"Phase 3 (Nonlinear Saturation): {phase2_end} - {len(flux)}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    timesteps = np.arange(len(flux))

    # Plot the flux
    ax.plot(timesteps, flux, "b-", linewidth=1.5, label="Energy Flux")

    # Add phase regions with different background colors
    ax.axvspan(0, phase1_end, alpha=0.15, color="gray", label="Phase 1: Transient")
    ax.axvspan(
        phase1_end,
        phase2_end,
        alpha=0.15,
        color="orange",
        label="Phase 2: Linear Growth",
    )
    ax.axvspan(
        phase2_end,
        len(flux),
        alpha=0.15,
        color="green",
        label="Phase 3: Nonlinear Saturation",
    )

    # Add vertical lines at phase boundaries
    ax.axvline(phase1_end, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(phase2_end, color="orange", linestyle="--", linewidth=1.5, alpha=0.7)

    # Add horizontal line for final mean in saturation phase
    ax.axhline(
        final_mean,
        xmin=phase2_end / len(flux),
        xmax=1.0,
        color="green",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label=f"Q̄ (Saturation Mean): {final_mean:.2f}",
    )

    # Add text annotations for phases
    phase1_mid = phase1_end / 2
    phase2_mid = (phase1_end + phase2_end) / 2
    phase3_mid = (phase2_end + len(flux)) / 2

    y_max = flux.max() * 1.1

    ax.text(
        phase1_mid,
        y_max * 0.95,
        "Phase 1:\nTransient",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.text(
        phase2_mid,
        y_max * 0.95,
        "Phase 2:\nLinear Growth",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.text(
        phase3_mid,
        y_max * 0.95,
        "Phase 3:\nNonlinear Saturation",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Labels and title
    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel("Energy Flux (Q)", fontsize=13)
    ax.set_title("Gyrokinetic Flux Evolution: Three Characteristic Phases")

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    # Set y-axis to start from 0 or slightly below to show initial transient
    y_min = min(0, flux.min() * 1.1)
    ax.set_ylim([y_min, y_max])

    plt.tight_layout()

    # Create dedicated folder for this plot
    output_folder = (
        Path(__file__).parent.parent.parent.parent
        / "outputs"
        / f"flux_phases_sim_{sim_id}"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save the figure
    output_path = output_folder / "flux_phases_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF for thesis
    output_path_pdf = output_folder / "flux_phases_plot.pdf"
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"PDF saved to: {output_path_pdf}")

    # Generate markdown description file
    description_path = output_folder / "description.md"

    # Calculate additional statistics
    flux_std = np.std(flux[-200:])
    flux_min = np.min(flux)
    flux_max = np.max(flux)

    with open(description_path, "w", encoding="utf-8") as f:
        f.write("# Gyrokinetic Flux Evolution: Three Characteristic Phases\n\n")
        f.write("## Simulation Information\n\n")
        f.write(f"- **Simulation ID:** {sim_id}\n")
        f.write("- **Dataset:** gyroswin_id\n")
        f.write(f"- **Total timesteps:** {len(flux)}\n\n")

        f.write("## Operating Parameters\n\n")
        f.write(f"- **q (safety factor):** {operating_params['q']:.4f}\n")
        f.write(f"- **ŝ (magnetic shear):** {operating_params['shat']:.4f}\n")
        f.write(f"- **R/L_T (temperature gradient):** {operating_params['rlt']:.4f}\n")
        f.write(f"- **R/L_n (density gradient):** {operating_params['rln']:.4f}\n\n")

        f.write("## Phase Boundaries\n\n")
        f.write("### Phase 1: Initial Transient (Noise-Dominated)\n")
        f.write(f"- **Duration:** Timesteps 0 – {phase1_end}\n")
        f.write(
            "- **Characteristics:** Simulation evolves from initial conditions; flux fluctuates weakly near zero\n"
        )
        f.write(
            "- **Physics:** Unstable eigenmodes have not yet fully emerged; stable components may still be damped\n\n"
        )

        f.write("### Phase 2: Linear Growth\n")
        f.write(f"- **Duration:** Timesteps {phase1_end} – {phase2_end}\n")
        f.write(
            "- **Characteristics:** Unstable modes amplify rapidly; flux departs systematically from noise level\n"
        )
        f.write("- **Physics:** Exponential growth of linear instabilities\n\n")

        f.write("### Phase 3: Nonlinear Saturation (Turbulent Steady State)\n")
        f.write(f"- **Duration:** Timesteps {phase2_end} – {len(flux)}\n")
        f.write(
            "- **Characteristics:** Flux fluctuates around a roughly constant mean\n"
        )
        f.write(
            "- **Physics:** Nonlinear mode coupling dominates; fully developed turbulence\n\n"
        )

        f.write("## Flux Statistics\n\n")
        f.write(f"- **Q̄ (Saturation mean, last 200 timesteps):** {final_mean:.4f}\n")
        f.write(f"- **σ (Standard deviation in saturation):** {flux_std:.4f}\n")
        f.write(f"- **Q_min (Minimum flux):** {flux_min:.4f}\n")
        f.write(f"- **Q_max (Maximum flux):** {flux_max:.4f}\n\n")

        f.write("## References\n\n")
        f.write(
            "The three-phase structure is characteristic of gyrokinetic turbulence simulations:\n\n"
        )
        f.write(
            "- Abel, I. G., et al. (2013). Multiscale gyrokinetics for rotating tokamak plasmas: fluctuations, transport and energy flows. *Reports on Progress in Physics*, 76(11), 116201.\n"
        )
        f.write(
            "- Paischer, H., et al. (2025). GyroSWIN: A global gyrokinetic simulation database.\n\n"
        )

        f.write("## Plot Files\n\n")
        f.write("- **PNG (high resolution):** `flux_phases_plot.png`\n")
        f.write("- **PDF (vector format):** `flux_phases_plot.pdf`\n")

    print(f"Description saved to: {description_path}")

    plt.show()

    return {
        "sim_id": sim_id,
        "operating_params": operating_params,
        "phase_boundaries": {
            "phase1_end": int(phase1_end),
            "phase2_end": int(phase2_end),
        },
        "flux_stats": {
            "saturation_mean": float(final_mean),
            "saturation_std": float(flux_std),
            "min": float(flux_min),
            "max": float(flux_max),
        },
        "output_folder": str(output_folder),
    }


if __name__ == "__main__":
    results = main()
    print("\nResults summary:")
    print(f"  Output folder: {results['output_folder']}")
    print(f"  Saturation mean (Q̄): {results['flux_stats']['saturation_mean']:.2f}")
