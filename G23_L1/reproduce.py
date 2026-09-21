from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "regular_G23"
OUTPUT_DIR = PROJECT_DIR / "generated"

KB_EV_K = 8.617333e-5
NA_CM2 = 3.816e15
E_CHARGE_C = 1.602176634e-19
EMAX_EV = 2.5
N_INTEGRATION_POINTS = 20_001


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_files() -> None:
    manifest = load_json(DATA_DIR / "manifest.json")
    expected = {item["name"]: item["sha256"] for item in manifest["files"]}
    for name in ("L1_data.csv", "L1_params.json", "results_template.json"):
        actual = sha256(DATA_DIR / name)
        if actual != expected[name]:
            raise RuntimeError(f"Raw-file checksum mismatch for {name}")


def fermi(E_eV: np.ndarray, EF_eV: float, T_K: float) -> np.ndarray:
    x = np.clip((E_eV - EF_eV) / (KB_EV_K * T_K), -60.0, 60.0)
    return 1.0 / (np.exp(x) + 1.0)


def n_band(EF_eV: float, T_K: float, c: float) -> float:
    energy = np.linspace(0.0, EMAX_EV, N_INTEGRATION_POINTS)
    integrand = c * energy * fermi(energy, EF_eV, T_K)
    return float(NA_CM2 * np.trapezoid(integrand, energy))


def n_fermi(EF_eV: float, T_K: float, c: float) -> float:
    energy = np.linspace(EF_eV, EMAX_EV, N_INTEGRATION_POINTS)
    integrand = c * energy * fermi(energy, EF_eV, T_K)
    return float(NA_CM2 * np.trapezoid(integrand, energy))


def n_fermi_closed(EF_eV: float, T_K: float, c: float) -> float:
    kT_eV = KB_EV_K * T_K
    return float(NA_CM2 * c * kT_eV * (EF_eV * math.log(2.0) + (math.pi**2 / 12.0) * kT_eV))


def correction_model(EF_eV: float, T_K: float, c: float) -> float:
    return n_band(EF_eV, T_K, c) / n_fermi(EF_eV, T_K, c)


def bisection_crossing(target: float, T_K: float, c: float, low: float = 0.0, high: float = 0.25) -> float:
    f_low = correction_model(low, T_K, c) - target
    f_high = correction_model(high, T_K, c) - target
    if f_low * f_high > 0.0:
        raise RuntimeError(f"C={target:g} is not bracketed by [{low}, {high}] eV")
    for _ in range(70):
        mid = 0.5 * (low + high)
        f_mid = correction_model(mid, T_K, c) - target
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return 0.5 * (low + high)


def read_l1_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = ["EF_eV", "n_band_cm2", "rho_ohm_sq"]
        if reader.fieldnames != expected_columns:
            raise RuntimeError(f"Unexpected L1 columns: {reader.fieldnames}")
        for row in reader:
            rows.append({key: float(row[key]) for key in expected_columns})
    if not rows:
        raise RuntimeError("L1_data.csv is empty")
    return rows


def current_git_commit() -> str:
    override = os.environ.get("PIPELINE_COMMIT")
    if override is not None:
        commit = override.strip().lower()
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise RuntimeError(f"Invalid PIPELINE_COMMIT value: {override}")
        return commit

    git = os.environ.get("GIT_EXE") or shutil.which("git")
    if git is None:
        raise RuntimeError("git is not on PATH; add Git to PATH or set GIT_EXE before running the pipeline")
    commit = subprocess.check_output(
        [git, "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True
    ).strip()
    if len(commit) != 40:
        raise RuntimeError(f"Unexpected git commit hash: {commit}")
    return commit


def write_calculation_table(rows: list[dict[str, float]], path: Path) -> None:
    fieldnames = ["EF_eV", "n_band_cm2_supplied", "n_fermi_cm2", "correction_factor"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(row[key], ".12g") for key in fieldnames})


def make_plot(
    rows: list[dict[str, float]],
    EF_op_eV: float,
    C_op: float,
    crossing_2_eV: float,
    crossing_4_eV: float,
    path: Path,
) -> None:
    ef = np.array([row["EF_eV"] for row in rows])
    correction = np.array([row["correction_factor"] for row in rows])

    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    ax.plot(ef, correction, color="#173f67", linewidth=2.0, marker="o", markersize=5.5,
            label="G23 supplied $n_{band}$ / calculated $n_F$")
    ax.scatter([EF_op_eV], [C_op], color="#c23b3b", marker="*", s=170, zorder=5,
               label="Exact operating point")

    crossing_colors = {2.0: "#4a8f6b", 4.0: "#8b5fa8"}
    for target, crossing in ((2.0, crossing_2_eV), (4.0, crossing_4_eV)):
        color = crossing_colors[target]
        ax.axhline(target, color=color, linestyle="--", linewidth=1.1, alpha=0.85)
        ax.axvline(crossing, color=color, linestyle=":", linewidth=1.1, alpha=0.85)
        ax.scatter([crossing], [target], color=color, s=42, zorder=4)
        ax.annotate(
            f"C={target:g} at {crossing:.4f} eV",
            xy=(crossing, target),
            xytext=(7, 9),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )

    ax.annotate(
        f"$E_{{F,op}}$={EF_op_eV:.5f} eV\n$C_{{op}}$={C_op:.4f}",
        xy=(EF_op_eV, C_op),
        xytext=(18, -48),
        textcoords="offset points",
        fontsize=9.5,
        color="#8d2525",
        arrowprops={"arrowstyle": "->", "color": "#8d2525", "lw": 1.1},
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d7b0b0"},
    )

    ax.set_title("G23 L1 carrier-counting correction", fontsize=14, weight="bold")
    ax.set_xlabel("Fermi level, $E_F$ (eV)")
    ax.set_ylabel("Correction factor, $C(E_F)=n_{band}/n_F$")
    ax.set_xlim(0.02, 0.26)
    ax.set_ylim(1.0, max(6.8, float(correction.max()) + 0.4))
    ax.grid(True, color="#d9dfe5", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    verify_raw_files()
    params = load_json(DATA_DIR / "L1_params.json")
    template = load_json(DATA_DIR / "results_template.json")
    rows = read_l1_csv(DATA_DIR / "L1_data.csv")

    if params["group"] != "G23" or template["group"] != "G23":
        raise RuntimeError("Group-code check failed")
    if params["dataset_set"] != "regular" or template["dataset_set"] != "regular":
        raise RuntimeError("Dataset-set check failed")

    c = float(params["c"])
    T_K = float(params["T"])
    EF_op_eV = float(params["EF_op_eV"])
    rho_op_ohm_sq = float(params["rho_ohm_sq"])

    if c <= 0.0 or T_K <= 0.0 or rho_op_ohm_sq <= 0.0:
        raise RuntimeError("Expected positive c, T and sheet resistivity")

    rho_values = np.array([row["rho_ohm_sq"] for row in rows])
    if not np.allclose(rho_values, rho_op_ohm_sq, rtol=0.0, atol=1e-6):
        raise RuntimeError("CSV sheet resistivity does not match L1_params.json")

    max_band_relative_error = 0.0
    calculation_rows: list[dict[str, float]] = []
    for source_row in rows:
        EF_eV = source_row["EF_eV"]
        supplied_band = source_row["n_band_cm2"]
        recomputed_band = n_band(EF_eV, T_K, c)
        max_band_relative_error = max(
            max_band_relative_error,
            abs(recomputed_band - supplied_band) / supplied_band,
        )
        nF = n_fermi(EF_eV, T_K, c)
        calculation_rows.append(
            {
                "EF_eV": EF_eV,
                "n_band_cm2_supplied": supplied_band,
                "n_fermi_cm2": nF,
                "correction_factor": supplied_band / nF,
            }
        )

    if max_band_relative_error > 2e-8:
        raise RuntimeError(f"Supplied n_band check failed: max relative error={max_band_relative_error:g}")

    sweep_correction = np.array([row["correction_factor"] for row in calculation_rows])
    if not np.all(np.diff(sweep_correction) > 0.0):
        raise RuntimeError("Correction factor is not strictly increasing over the supplied sweep")

    reference_targets = {0.0: 1.00, 0.073: 2.00, 0.10: 2.60, 0.20: 5.10}
    reference_tolerances = {0.0: 0.005, 0.073: 0.03, 0.10: 0.03, 0.20: 0.05}
    reference_checks: dict[str, dict[str, float | bool]] = {}
    for EF_eV, expected in reference_targets.items():
        actual = correction_model(EF_eV, T_K, c)
        passed = abs(actual - expected) <= reference_tolerances[EF_eV]
        reference_checks[f"{EF_eV:.3f}_eV"] = {
            "actual": actual,
            "expected_approx": expected,
            "absolute_tolerance": reference_tolerances[EF_eV],
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(f"Reference ratio check failed at EF={EF_eV} eV: {actual}")

    n_band_op_cm2 = n_band(EF_op_eV, T_K, c)
    n_fermi_op_cm2 = n_fermi(EF_op_eV, T_K, c)
    n_fermi_closed_op_cm2 = n_fermi_closed(EF_op_eV, T_K, c)
    closed_form_relative_error = abs(n_fermi_op_cm2 - n_fermi_closed_op_cm2) / n_fermi_closed_op_cm2
    # The guide prescribes a 20,001-point trapezoidal integral. Its discretisation
    # error relative to the infinite-upper-limit closed form is about 3e-7 here.
    if closed_form_relative_error > 1e-6:
        raise RuntimeError(f"Numerical/closed-form n_F check failed: {closed_form_relative_error:g}")

    correction_factor = n_band_op_cm2 / n_fermi_op_cm2
    mobility_band_cm2_Vs = 1.0 / (E_CHARGE_C * n_band_op_cm2 * rho_op_ohm_sq)
    mobility_corrected_cm2_Vs = 1.0 / (E_CHARGE_C * n_fermi_op_cm2 * rho_op_ohm_sq)
    if not math.isclose(
        mobility_corrected_cm2_Vs / mobility_band_cm2_Vs,
        correction_factor,
        rel_tol=2e-13,
    ):
        raise RuntimeError("Mobility-ratio identity failed")

    crossing_2_eV = bisection_crossing(2.0, T_K, c)
    crossing_4_eV = bisection_crossing(4.0, T_K, c)
    pipeline_commit = current_git_commit()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = template
    results["pipeline_commit"] = pipeline_commit
    results["L1"] = {
        "correction_factor": correction_factor,
        "n_fermi_op_cm2": n_fermi_op_cm2,
        "mobility_corrected_cm2_Vs": mobility_corrected_cm2_Vs,
    }
    results_path = OUTPUT_DIR / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    plot_path = OUTPUT_DIR / "L1_correction_factor.png"
    make_plot(calculation_rows, EF_op_eV, correction_factor, crossing_2_eV, crossing_4_eV, plot_path)
    write_calculation_table(calculation_rows, OUTPUT_DIR / "L1_calculation_table.csv")

    percent_higher = 100.0 * (correction_factor - 1.0)
    percent_band_underestimate = 100.0 * (1.0 - 1.0 / correction_factor)
    brief_note = f"""# G23 L1 brief note

At the exact operating point, $E_{{F,op}}={EF_op_eV:.8f}$ eV, the finite-temperature band-edge count is $n_{{band}}={n_band_op_cm2:.6e}$ cm^-2 and the Fermi-referenced count is $n_F={n_fermi_op_cm2:.6e}$ cm^-2. Their ratio is $C={correction_factor:.6f}$. The $C=2$ and $C=4$ crossings occur at $E_F={crossing_2_eV:.6f}$ eV and $E_F={crossing_4_eV:.6f}$ eV, respectively.

$C$ increases with positive doping because $n_{{band}}$ accumulates all occupied states from the Dirac point to the thermal tail, whereas $n_F$ counts only the occupied states above the current Fermi level. As $E_F$ rises, the filled interval below $E_F$ grows much faster than the thermal population above it.

Using rho_square={rho_op_ohm_sq:.6f} ohm/square gives mu_band={mobility_band_cm2_Vs:.6f} cm^2 V^-1 s^-1 and mu_corrected={mobility_corrected_cm2_Vs:.6f} cm^2 V^-1 s^-1. The corrected mobility is therefore {correction_factor:.4f} times the band-edge estimate ({percent_higher:.2f}% higher); equivalently, the band-edge estimate is {percent_band_underestimate:.2f}% below the corrected value. No additional factor of 10^4 was applied because the carrier density is already in cm^-2.
"""
    (OUTPUT_DIR / "brief_note.md").write_text(brief_note, encoding="utf-8")
    (OUTPUT_DIR / "commit_hash.txt").write_text(pipeline_commit + "\n", encoding="ascii")

    validation = {
        "group": "G23",
        "raw_files_sha256_verified": True,
        "max_supplied_vs_recomputed_n_band_relative_error": max_band_relative_error,
        "n_fermi_numeric_vs_closed_form_relative_error_at_operating_point": closed_form_relative_error,
        "reference_ratio_checks": reference_checks,
        "positive_EF_sweep_strictly_monotonic": True,
        "operating_point": {
            "EF_op_eV": EF_op_eV,
            "n_band_op_cm2": n_band_op_cm2,
            "n_fermi_op_cm2": n_fermi_op_cm2,
            "correction_factor": correction_factor,
            "rho_ohm_sq": rho_op_ohm_sq,
            "mobility_band_cm2_Vs": mobility_band_cm2_Vs,
            "mobility_corrected_cm2_Vs": mobility_corrected_cm2_Vs,
        },
        "crossings_eV": {"C_2": crossing_2_eV, "C_4": crossing_4_eV},
        "pipeline_commit": pipeline_commit,
    }
    with (OUTPUT_DIR / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    checkpoint_zip = PROJECT_DIR / "G23_L1_Checkpoint.zip"
    with zipfile.ZipFile(checkpoint_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in ("results.json", "L1_correction_factor.png", "brief_note.md", "commit_hash.txt"):
            archive.write(OUTPUT_DIR / filename, arcname=filename)

    print(f"G23 exact operating point: EF={EF_op_eV:.12g} eV")
    print(f"n_band={n_band_op_cm2:.12g} cm^-2")
    print(f"n_F={n_fermi_op_cm2:.12g} cm^-2")
    print(f"C={correction_factor:.12g}")
    print(f"mobility_corrected={mobility_corrected_cm2_Vs:.12g} cm^2 V^-1 s^-1")
    print(f"C=2 crossing: {crossing_2_eV:.12g} eV")
    print(f"C=4 crossing: {crossing_4_eV:.12g} eV")
    print(f"pipeline commit: {pipeline_commit}")
    print(f"checkpoint: {checkpoint_zip}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
