#!/usr/bin/env python3
"""
compare_profiles.py

Overlay plots for Schwinger static_profiles from:
- TenPy/TNS: <out>_tns_profiles.npz  (you add this save in tsv2.py)
- NetKet/NQS: <out>_vmc_profiles.npz (from nqs_schwinger_netket.py)

Produces:
- <out>_compare_charge.png
- <out>_compare_efield.png
- <out>_compare_energies.txt
- <out>_compare_energies.png  (simple bar-like plot)

Usage:
  python compare_profiles.py --out profiles \
    --tns profiles_tns_profiles.npz \
    --nqs profiles_vmc_profiles.npz

If you omit --tns/--nqs it will default to <out>_tns_profiles.npz and <out>_vmc_profiles.npz


  python compare_profiles.py --out profiles \
    --tns  ./static_profiles/profiles_for_comp_profiles.npz
    --nqs ./nqs/profiles_vmc_profiles.npz

"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir_for(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    # Convert 0-d object arrays holding dicts back to python dicts
    out = {}
    for k in d.files:
        v = d[k]
        if isinstance(v, np.ndarray) and v.shape == () and v.dtype == object:
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _get_energy_block(data):
    """
    Returns dict with keys:
      E0_vac, E0_str, dE
    for either TN or NQS files.
    """
    if "E0_vac" in data and "E0_str" in data:
        E0_v = float(np.asarray(data["E0_vac"]).item())
        E0_s = float(np.asarray(data["E0_str"]).item())
        dE = float(np.asarray(data.get("dE", E0_s - E0_v)).item())
        return {"E0_vac": E0_v, "E0_str": E0_s, "dE": dE}

    # Both NQS and TNS files store energies in nested dict 
    if "energies" in data and isinstance(data["energies"], dict):
        e = data["energies"]
        print("Energies dict found in npz: {}".format(e))
        # prefer offset-corrected energies if present
        if "Ev_vmc" in e and "Es_vmc" in e:
            E0_v = float(e["Ev_vmc"])
            E0_s = float(e["Es_vmc"])
            return {"E0_vac": E0_v, "E0_str": E0_s, "dE": float(E0_s - E0_v)}
        # fallback
        if "Ev_vmc_raw" in e and "Es_vmc_raw" in e:
            E0_v = float(e["Ev_vmc_raw"])
            E0_s = float(e["Es_vmc_raw"])
            return {"E0_vac": E0_v, "E0_str": E0_s, "dE": float(E0_s - E0_v)}
        
        if "Evac" in e and "Estr" in e:
            E0_v = float(e["Evac"])
            E0_s = float(e["Estr"])
            return {"E0_vac": E0_v, "E0_str": E0_s, "dE": float(E0_s - E0_v)}

    # Another possible NQS layout (older version)
    ## TODO This is dangerous as there may be a collision between E field key name and energy key name
    # if "E_vac" in data and "E_str" in data:
    #     print("E_vac data is  {}".format(data["E_vac"]))
    #     E0_v = float(np.asarray(data["E_vac"]).item())
    #     E0_s = float(np.asarray(data["E_str"]).item())
    #     return {"E0_vac": E0_v, "E0_str": E0_s, "dE": float(E0_s - E0_v)}

    raise KeyError("Could not find energies in npz (expected E0_vac/E0_str or energies dict).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="profiles", help="Output prefix for compare plots.")
    ap.add_argument("--tns", type=str, default=None, help="TenPy saved npz (default: <out>_tns_profiles.npz)")
    ap.add_argument("--nqs", type=str, default=None, help="NetKet saved npz (default: <out>_vmc_profiles.npz)")
    args = ap.parse_args()

    tns_path = args.tns or (args.out + "_tns_profiles.npz")
    nqs_path = args.nqs or (args.out + "_vmc_profiles.npz")

    tns = load_npz(tns_path)
    nqs = load_npz(nqs_path)

    # ---- Fetch profiles ----
    # TenPy file expected keys:
    #   charge_vac, efield_vac, charge_str, efield_str
    Qv_t = np.asarray(tns["Q_vac"], dtype=float)
    Ev_t = np.asarray(tns["E_vac"], dtype=float)
    Qs_t = np.asarray(tns["Q_str"], dtype=float)
    Es_t = np.asarray(tns["E_str"], dtype=float)

    # NetKet file expected keys:
    #   Q_vac, E_vac, Q_str, E_str
    Qv_n = np.asarray(nqs["Q_vac"], dtype=float)
    Ev_n = np.asarray(nqs["E_vac"], dtype=float)
    Qs_n = np.asarray(nqs["Q_str"], dtype=float)
    Es_n = np.asarray(nqs["E_str"], dtype=float)


    # print values for debugging
    print("TNS vacuum charge profile: {}".format(Qv_t))
    print("NQS vacuum charge profile: {}".format(Qv_n))
    print("\n\n")
    print("TNS string charge profile: {}".format(Qs_t))
    print("NQS string charge profile: {}".format(Qs_n))
    print("\n\n")
    print("TNS vacuum E-field profile: {}".format(Ev_t))
    print("NQS vacuum E-field profile: {}".format(Ev_n))
    print("\n\n")
    print("TNS string E-field profile: {}".format(Es_t))
    print("NQS string E-field profile: {}".format(Es_n))

    if Qv_t.shape != Qv_n.shape or Qs_t.shape != Qs_n.shape:
        raise ValueError(f"Charge profile shapes differ: TNS {Qv_t.shape}/{Qs_t.shape} vs NQS {Qv_n.shape}/{Qs_n.shape}")
    if Ev_t.shape != Ev_n.shape or Es_t.shape != Es_n.shape:
        raise ValueError(f"E-field profile shapes differ: TNS {Ev_t.shape}/{Es_t.shape} vs NQS {Ev_n.shape}/{Es_n.shape}")

    L = Qv_t.shape[0]

    # ---- Energies ----
    Et = _get_energy_block(tns)
    En = _get_energy_block(nqs)
    print("TNS energies: {}".format(Et))
    print("NQS energies: {}".format(En))

    # ---- Write energies text ----
    txt_path = args.out + "_compare_energies.txt"
    ensure_dir_for(txt_path)
    with open(txt_path, "w") as f:
        f.write("Comparison energies (offset conventions depend on input files)\n")
        f.write(f"TNS: E0_vac = {Et['E0_vac']:.12f}\n")
        f.write(f"TNS: E0_str = {Et['E0_str']:.12f}\n")
        f.write(f"TNS: dE     = {Et['dE']:.12f}\n\n")
        f.write(f"NQS: E0_vac = {En['E0_vac']:.12f}\n")
        f.write(f"NQS: E0_str = {En['E0_str']:.12f}\n")
        f.write(f"NQS: dE     = {En['dE']:.12f}\n")
    print(f"[saved] {txt_path}")

    # ---- Charge overlay plot ----
    x_sites = np.arange(L)
    fig_path = args.out + "_compare_charge.png"
    ensure_dir_for(fig_path)
    plt.figure()
    plt.plot(x_sites, Qv_t, marker="o", label="TNS vacuum")
    plt.plot(x_sites, Qv_n, marker="o", label="NQS vacuum")
    plt.plot(x_sites, Qs_t, marker="o", label="TNS string")
    plt.plot(x_sites, Qs_n, marker="o", label="NQS string")
    plt.xlabel("site n")
    plt.ylabel(r"$\langle Q_n \rangle$")
    plt.title(f"Charge profile comparison (L={L})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[saved] {fig_path}")

    # ---- E-field overlay plot ----
    x_links = np.arange(L - 1)
    fig_path = args.out + "_compare_efield.png"
    ensure_dir_for(fig_path)
    plt.figure()
    plt.plot(x_links, Ev_t, marker="o", label="TNS vacuum")
    plt.plot(x_links, Ev_n, marker="o", label="NQS vacuum")
    plt.plot(x_links, Es_t, marker="o", label="TNS string")
    plt.plot(x_links, Es_n, marker="o", label="NQS string")
    plt.xlabel("link n")
    plt.ylabel(r"$\langle E_n \rangle$")
    plt.title(f"E-field comparison (L={L})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[saved] {fig_path}")

    # ---- Simple energy comparison plot ----
    fig_path = args.out + "_compare_energies.png"
    ensure_dir_for(fig_path)
    labels = ["E_vac", "E_str", "dE"]
    tvals = [Et["E0_vac"], Et["E0_str"], Et["dE"]]
    nvals = [En["E0_vac"], En["E0_str"], En["dE"]]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure()
    plt.bar(x - width / 2, tvals, width, label="TNS")
    plt.bar(x + width / 2, nvals, width, label="NQS")
    plt.xticks(x, labels)
    plt.ylabel("energy")
    plt.title("Energy comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[saved] {fig_path}")


if __name__ == "__main__":
    main()
