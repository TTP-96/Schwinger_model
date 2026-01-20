 #!/usr/bin/env python3
"""

Schwinger model (1+1D) in Hamiltonian (Kogut–Susskind) formulation with gauge field
integrated out, implemented as a spin-1/2 chain with long-range couplings. The code uses the
TenPy library (https://tenpy.readthedocs.io/en/latest/) for MPS/DMRG/TDVP.

The Hamiltonian in spin language is:
    H = -x Σ (S^+_n S^-_{n+1} + h.c.)
        + m Σ (-1)^n S^z_n
        + (g2/2) Σ_links E_n^2
with Gauss law integrated (open BC, E_{-1}=0):
    Q_n = S^z_n + 0.5*(-1)^n
    E_n = Σ_{k=0}^n (Q_k + qext_k)

This model describes staggered fermions with mass m,
hopping x, and electric coupling g2.
The 0.5*(-1)^n term encodes the staggered background charge of the Dirac sea which ensures 
that the vacuum has zero total charge. Note that the reason it is 0.5*(-1)^n instead of 0.5*(1-(-1)^n) is 
because we are working in spin language here instead of fermion language like in the ED code.

The model includes static external charges via a site-dependent background field (qext).

# Summary of features:
The code provides routines for:
    - DMRG ground state search with static charges
    - Scanning string energy vs. charge separation
    - Real-time quench dynamics with TDVP after introducing static charges

    
# Example usage:

    How to run:
    1. Compute static charge and E-field profiles for vacuum and string states:
    python tsv2.py static_profiles --left 4 --right 12   --L 16 --x 1.0 --m 0.1 --g2 1.0   --q 1.0   --chi 256 --sweeps 10 --out profiles_for_comp    

    2. Compute string energy vs. separation R:
    python tsv2.py string_energy_scan   --L 40 --x 1.0 --m 0.1 --g2 1.0   --q 1.0 --Rmin 2 --Rmax 28 --Rstep 2   --chi 256 --sweeps 10 --out sch_scan

    3. Perform quench dynamics after introducing static charges at t=0:
    python tsv2.py quench      --left 8 --right 24   --L 32 --x 1.0 --m 0.1 --g2 1.0   --q 1.0   --dt 0.01 --tmax 2.0 --tdvp_engine two-site   --chi 256 --out sch_quench  

    Comparing with NetKet NQS results:
    python tsv2.py static_profiles --left 4 --right 12   --L 16 --x 1.0 --m 0.1 --g2 1.0   --q 1.0   --chi 256 --sweeps 10 --out profiles_for_comp    
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
try:
    from tenpy.algorithms.tdvp import TwoSiteTDVPEngine, SingleSiteTDVPEngine
except Exception: # in case of older TenPy versions
    from tenpy.algorithms import tdvp as _tdvp
    TwoSiteTDVPEngine = getattr(_tdvp, "TwoSiteTDVPEngine")
    SingleSiteTDVPEngine = getattr(_tdvp, "SingleSiteTDVPEngine")



# qext-dependent constant energy in the electric-field 

def electric_constant_offset(model_params, include_sz2_constant=False):
    """
    Electric energy: (g2/2) Σ_{n=0}^{L-2} E_n^2 with
      E_n = Σ_{k<=n} (Sz_k + d_k),  d_k = 0.5*(-1)^k + qext_k

    Expanding E_n^2 gives:
      (g2/2) Σ_n [ ... + (Σ_{k<=n} d_k)^2 ]
    The last term is a c-number depending on qext and must be included in total energies
    when comparing different qext configs (string energy vs separation).

    include_sz2_constant:
      If True, also add the Sz^2 constant from (Σ Sz_k)^2 (independent of qext for spin-1/2),
      which cancels in ΔE(R) anyway. Default False.
    """
    L = int(model_params["L"])
    g2 = float(model_params["g2"]) if "g2" in model_params else float(model_params.get("g2", 1.0))

    qext = model_params["qext"] if "qext" in model_params else model_params.get("qext", np.zeros(L))
    qext = np.array(qext, dtype=float)
    if qext.shape != (L,):
        raise ValueError(f"qext must have shape (L,), got {qext.shape}")

    stag = np.array([1.0 if (n % 2 == 0) else -1.0 for n in range(L)], dtype=float)
    d = 0.5 * stag + qext

    # D_n = Σ_{k<=n} d_k for links n=0..L-2
    D_links = np.cumsum(d)[:-1]  # length L-1
    offset = 0.5 * g2 * float(np.sum(D_links**2))
    

    if include_sz2_constant:
        # For spin-1/2: Sz^2 = (1/2)^2 = 1/4.
        # Contribution from (g2/2) Σ_n Σ_{k<=n} Sz_k^2 = (g2/2) Σ_k (L-1-k)*(1/4)
        # = g2 * L*(L-1) / 16, independent of qext.
        # Note Σ_k (L-1-k) comes from counting how many E_n each Sz_k appears in (how many n >= k).  
        # Then sum_{k=0}^{L-1} (L-1-k) = L*(L-1)/2.
        offset += g2 * L * (L - 1) / 16.0

    return offset


# -----------------------------
# Schwinger model (integrated gauge field)
# -----------------------------
class SchwingerIntegratedModel(CouplingMPOModel):
    r"""
    Hamiltonian (common convention; constants dropped from MPO):
      H = -x Σ (S^+_n S^-_{n+1} + h.c.)
          + m Σ (-1)^n S^z_n
          + (g2/2) Σ_links E_n^2

    with Gauss law integrated (open BC, E_{-1}=0):
      Q_n = S^z_n + 0.5*(-1)^n
      E_n = Σ_{k=0}^n (Q_k + qext_k)

    IMPORTANT: the MPO omits additive constants; see electric_constant_offset().
    """

    def init_sites(self, model_params):
        conserve = model_params.get("conserve", "Sz") 
        return SpinHalfSite(conserve=conserve)

    def init_lattice(self, model_params):
        L = int(model_params["L"])
        conserve = model_params.get("conserve", "Sz")
        site = SpinHalfSite(conserve=conserve)
        return Chain(L, site, bc="open", bc_MPS="finite")

    def init_terms(self, model_params):
        L = int(model_params["L"])
        x  = float(model_params["x"])  if "x"  in model_params else float(model_params.get("x", 1.0))
        m  = float(model_params["m"])  if "m"  in model_params else float(model_params.get("m", 0.0))
        g2 = float(model_params["g2"]) if "g2" in model_params else float(model_params.get("g2", 1.0))

        qext = model_params["qext"] if "qext" in model_params else model_params.get("qext", np.zeros(L))
        qext = np.array(qext, dtype=float)
        if qext.shape != (L,):
            raise ValueError(f"qext must have shape (L,), got {qext.shape}")

        stag = np.array([1.0 if (n % 2 == 0) else -1.0 for n in range(L)], dtype=float)
        d = 0.5 * stag + qext

        # kinetic term
        for i in range(L - 1):
            self.add_coupling_term(-x, i, i + 1, "Sp", "Sm", plus_hc=False)
            self.add_coupling_term(-x, i, i + 1, "Sm", "Sp", plus_hc=False)

        # mass term
        for i in range(L):
            self.add_onsite_term(m * stag[i], i, "Sz")

        # electric term (operator-dependent parts)
        A = np.zeros((L, L), dtype=float)
        for i in range(L):
            for j in range(L):
                A[i, j] = float(max(0, (L - 1) - max(i, j)))

        for i in range(L):
            for j in range(i + 1, L):
                Jij = g2 * A[i, j]
                if abs(Jij) > 1e-14:
                    self.add_coupling_term(Jij, i, j, "Sz", "Sz", plus_hc=False)

        h = g2 * (A @ d)
        for i in range(L):
            if abs(h[i]) > 1e-14:
                self.add_onsite_term(float(h[i]), i, "Sz")


# Utilities
def make_static_charges(L, left, right, q=1.0):
    qext = np.zeros(L, dtype=float)
    qext[left]  += q
    qext[right] -= q
    return qext


# Observables
def charge_density_Q(state_mps):
    L = state_mps.L
    sz   = np.array(state_mps.expectation_value("Sz"), dtype=float)
    stag = np.array([1.0 if (n % 2 == 0) else -1.0 for n in range(L)], dtype=float)
    Q = sz + 0.5 * stag
    return Q, sz


def electric_field_profile(state_mps, qext):
    L  = state_mps.L
    sz = np.array(state_mps.expectation_value("Sz"), dtype=float)
    stag = np.array([1.0 if (n % 2 == 0) else -1.0 for n in range(L)], dtype=float)
    d = 0.5 * stag + np.array(qext, dtype=float)
    cum = np.cumsum(sz + d)
    return cum[:-1].copy()


# DMRG ground state 
def dmrg_ground_state(model_params, chi_max=256, dmrg_sweeps=8, mixer=True, verbose=1, init_state="Neel"):
    model = SchwingerIntegratedModel(model_params)
    L     = int(model_params["L"])

    if init_state.lower() == "neel":
        product = ["up" if (i % 2 == 0) else "down" for i in range(L)]
    elif init_state.lower() == "vacuumlike":
        product = ["down" if (i % 2 == 0) else "up" for i in range(L)]
    else:
        product = ["up" for _ in range(L)]

    psi = MPS.from_product_state(model.lat.mps_sites(), product, bc="finite")

    dmrg_params = {
        "mixer": mixer,
        "max_E_err": 1e-9,
        "trunc_params": {"chi_max": int(chi_max), "svd_min": 1e-12},
        "verbose": verbose,
        "N_sweeps_check": 2,
        "N_sweeps": int(dmrg_sweeps),
    }
    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E0_mpo, psi = eng.run()

    # Add back the qext-dependent c-number offset missing from the MPO energy
    E0 = float(E0_mpo) + electric_constant_offset(model_params, include_sz2_constant=False)
    return E0, psi, model


# TDVP time evolution (quench)
def tdvp_quench(psi0, model, dt, tmax, chi_max=256, engine="two-site",
                n_steps_per_meas=1, verbose=1):
    N_steps = int(np.round(tmax / dt))
    if N_steps < 1:
        raise ValueError("tmax must be >= dt")
    tdvp_params = {
        "dt": float(dt),
        "N_steps": 1,
        "trunc_params": {"chi_max": int(chi_max), "svd_min": 1e-12},
        "verbose": verbose,
    }
    Eng = TwoSiteTDVPEngine if engine.lower().startswith("two") else SingleSiteTDVPEngine
    eng = Eng(psi0, model, tdvp_params)
    times  = [0.0]
    states = [psi0.copy()]
    for step in range(1, N_steps + 1):
        eng.run()
        if step % n_steps_per_meas == 0:
            times.append(step * dt)
            states.append(eng.psi.copy())
    return np.array(times), states


# Plot helpers
def plot_profiles(out_prefix, Q, E_links,S, title=""):
    L = len(Q)
    x_sites = np.arange(L)
    x_links = np.arange(L - 1)
    plt.figure()
    plt.plot(x_sites, Q, marker="o")
    plt.xlabel("site n")
    plt.ylabel(r"$\langle Q_n\rangle$")
    plt.title((title + "  charge density") if title else "charge density")
    plt.tight_layout()
    plt.savefig(out_prefix + "_charge.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(x_links, E_links, marker="o")
    plt.xlabel("link n (between n and n+1)")
    plt.ylabel(r"$\langle E_n\rangle$")
    plt.title((title + "  electric field") if title else "electric field")
    plt.tight_layout()
    plt.savefig(out_prefix + "_efield.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(np.arange(L - 1), S, marker="o")
    plt.xlabel("bond b")
    plt.ylabel("Entanglement entropy S(bond)")
    plt.title((title + "  entanglement entropy") if title else "entanglement entropy")
    plt.tight_layout()
    plt.savefig(out_prefix + "_entanglement_entropy.png", dpi=160)
    plt.close() 


def plot_string_energy(out_prefix, Rs, dEs, title="String energy vs separation"):
    plt.figure()
    plt.plot(Rs, dEs, marker="o")
    plt.xlabel("separation R (sites)")
    plt.ylabel(r"$\Delta E(R) = E_{\mathrm{gs}}(R)-E_{\mathrm{vac}}$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_prefix + "_string_energy.png", dpi=160)
    plt.close()


def plot_quench_heatmaps(out_prefix, times, Q_t, E_t, S_t):
    plt.figure()
    plt.imshow(Q_t, aspect="auto", origin="lower")
    plt.colorbar(label=r"$\langle Q_n\rangle$")
    plt.xlabel("site n")
    plt.ylabel("time index")
    plt.title("Quench dynamics: charge density")
    plt.tight_layout()
    plt.savefig(out_prefix + "_quench_charge_heatmap.png", dpi=160)
    plt.close()

    plt.figure()
    plt.imshow(E_t, aspect="auto", origin="lower")
    plt.colorbar(label=r"$\langle E_n\rangle$")
    plt.xlabel("link n")
    plt.ylabel("time index")
    plt.title("Quench dynamics: electric field")
    plt.tight_layout()
    plt.savefig(out_prefix + "_quench_efield_heatmap.png", dpi=160)
    plt.close()

    mid_link = E_t.shape[1] // 2
    plt.figure()
    plt.plot(times, E_t[:, mid_link], marker="o")
    plt.xlabel("t")
    plt.ylabel(r"$\langle E_{\mathrm{mid}}\rangle$")
    plt.title("Quench dynamics: mid-link electric field")
    plt.tight_layout()
    plt.savefig(out_prefix + "_quench_midlink_E.png", dpi=160)
    plt.close()

    plt.figure()
    plt.imshow(S_t, aspect="auto", origin="lower")
    plt.colorbar(label="Entanglement entropy S(bond)")
    plt.xlabel("bond b")
    plt.ylabel("time index")
    plt.title("Quench dynamics: entanglement entropy")
    plt.tight_layout()
    plt.savefig(out_prefix + "_quench_entanglement_entropy_heatmap.png", dpi=160)
    plt.close()


    # mid-link entropy 
    mid_bond = S_t.shape[1] // 2
    plt.figure()
    plt.plot(times, S_t[:, mid_bond], marker="o")
    plt.xlabel("t")
    plt.ylabel("S(mid bond)")
    plt.title("Quench dynamics: mid-bond entanglement entropy")
    plt.tight_layout()
    plt.savefig(out_prefix + "_quench_midbond_entropy.png", dpi=160)
    plt.close() 


# Main routines for CLI
def run_static_profiles(args):
    L     = args.L
    left  = args.left
    right = args.right
    q     = args.q

    vac_params = {"L": L, "x": args.x, "m": args.m, "g2": args.g2, "qext": np.zeros(L)}
    Evac, psivac, _ = dmrg_ground_state(
        vac_params, chi_max=args.chi, dmrg_sweeps=args.sweeps, mixer=True, verbose=args.verbose
    )

    qext = make_static_charges(L, left, right, q=q)
    str_params = {"L": L, "x": args.x, "m": args.m, "g2": args.g2, "qext": qext}
    Estr, psistr, _ = dmrg_ground_state(
        str_params, chi_max=args.chi, dmrg_sweeps=args.sweeps, mixer=True, verbose=args.verbose
    )
        
    Qvac,  sz_vac = charge_density_Q(psivac)
    Evac_links = electric_field_profile(psivac, np.zeros(L))
    Svac=entropy_vac = psivac.entanglement_entropy()
    Qstr,  sz_str = charge_density_Q(psistr)
    Estr_links = electric_field_profile(psistr, qext)
    Sstr=entropy_str = psistr.entanglement_entropy()
    # save profiles to npz to match NetKet output (sigma^z = 2*Sz)
    np.savez(
        args.out + "_profiles.npz",
        Q_vac=Qvac, E_vac=Evac_links,
        Q_str=Qstr, E_str=Estr_links,
        sigmaz_vac=(2.0 * sz_vac), sigmaz_str=(2.0 * sz_str),
        entropy_vac=Svac, entropy_str=Sstr,
        qext_vac=np.zeros(L), qext_str=qext,
        energies=dict(Evac=Evac, Estr=Estr),
        params=dict(L=L, x=args.x, m=args.m, g2=args.g2, left=left, right=right, q=q)
    )


    prefix_v = args.out + "_vac"
    prefix_s = args.out + f"_string_L{left}_R{right}"
    plot_profiles(prefix_v, Qvac,  Evac_links, Svac, title="vacuum")
    plot_profiles(prefix_s, Qstr,  Estr_links, Sstr, title=f"static charges (+{q},-{q}) at {left},{right}")
    
    print(f"[vacuum] E0 = {Evac:.12f}")
    print(f"[string ] E0 = {Estr:.12f}")
    print(f"[string ] ΔE = {Estr - Evac:.12f}")
    print(f"[saved ] {prefix_v}_charge.png, {prefix_v}_efield.png")
    print(f"[saved ] {prefix_s}_charge.png, {prefix_s}_efield.png")


def run_string_energy_scan(args):
    L = args.L
    q = args.q

    vac_params = {"L": L, "x": args.x, "m": args.m, "g2": args.g2, "qext": np.zeros(L)}
    Evac, _, _ = dmrg_ground_state(
        vac_params, chi_max=args.chi, dmrg_sweeps=args.sweeps, mixer=True, verbose=args.verbose
    )

    center = L // 2
    Rs, dEs = [], []
    for R in range(args.Rmin, args.Rmax + 1, args.Rstep):
        left  = max(0, center - R // 2)
        right = min(L - 1, left + R)
        if right >= L:
            continue
        qext = make_static_charges(L, left, right, q=q)
        params = {"L": L, "x": args.x, "m": args.m, "g2": args.g2, "qext": qext}
        E, _, _ = dmrg_ground_state(
            params, chi_max=args.chi, dmrg_sweeps=args.sweeps, mixer=True, verbose=args.verbose
        )
        Rs.append(right - left)
        dEs.append(E - Evac)
        print(f"[scan] R={right-left:3d}  E={E:.10f}  ΔE={E-Evac:.10f}")
    Rs  = np.array(Rs, dtype=int)
    dEs = np.array(dEs, dtype=float)
    plot_string_energy(args.out, Rs, dEs)
    print(f"[saved] {args.out}_string_energy.png")



def run_real_time_quench(args):
    L = args.L
    left, right, q = args.left, args.right, args.q

    vac_params = {"L": L, "x": args.x, "m": args.m, "g2": args.g2, "qext": np.zeros(L)}
    Evac, psivac, _ = dmrg_ground_state(
        vac_params, chi_max=args.chi, dmrg_sweeps=args.sweeps, mixer=True, verbose=args.verbose
    )
    print(f"[vacuum] E0 = {Evac:.12f}")

    qext = make_static_charges(L, left, right, q=q)
    quench_params = {"L": L, "x": args.x, "m": args.m, "g2": args.g2, "qext": qext}
    model_quench = SchwingerIntegratedModel(quench_params)

    times, states = tdvp_quench(
        psivac, model_quench,
        dt=args.dt, tmax=args.tmax,
        chi_max=args.chi,
        engine=args.tdvp_engine,
        n_steps_per_meas=args.meas_every,
        verbose=args.verbose
    )

    Q_t, E_t = [], []
    S_t = []
    for psi in states:
        Q, _ = charge_density_Q(psi)
        E    = electric_field_profile(psi, qext)
        psi.canonical_form()
        S  = psi.entanglement_entropy()
        Q_t.append(Q)
        E_t.append(E)
        S_t.append(S)
    Q_t = np.array(Q_t)
    E_t = np.array(E_t)
    S_t = np.array(S_t)

    out_prefix = args.out + f"_quench_L{left}_R{right}"
    plot_quench_heatmaps(out_prefix, times, Q_t, E_t, S_t)
    plot_profiles(out_prefix + "_t0", Q_t[0], E_t[0], S_t[0], title="quench t=0.0")
    plot_profiles(out_prefix + "_tfinal", Q_t[-1], E_t[-1], S_t[-1], title=f"quench t={times[-1]:.3f}")
    np.savez(
        out_prefix + "_data.npz",
        times=times, Q_t=Q_t, E_t=E_t, S_t=S_t,
        params=dict(L=L, x=args.x, m=args.m, g2=args.g2, left=left, right=right, q=q,
                    dt=args.dt, tmax=args.tmax, tdvp_engine=args.tdvp_engine)
    )
    print(f"[saved] {out_prefix}_quench_charge_heatmap.png")
    print(f"[saved] {out_prefix}_quench_efield_heatmap.png")
    print(f"[saved] {out_prefix}_quench_entanglement_entropy_heatmap.png")
    print(f"[saved] {out_prefix}_quench_midlink_E.png")
    print(f"[saved] {out_prefix}_data.npz")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(pp):
        pp.add_argument("--L", type=int, default=40, help="Lattice size")
        pp.add_argument("--x", type=float, default=1.0, help="Hopping strength")
        pp.add_argument("--m", type=float, default=0.3, help="Fermion mass")
        pp.add_argument("--g2", type=float, default=1.0, help="Electric coupling")
        pp.add_argument("--chi", type=int, default=256, help="Max bond dimension")
        pp.add_argument("--sweeps", type=int, default=10)
        pp.add_argument("--verbose", type=int, default=1)
        pp.add_argument("--out", type=str, default="out_schwinger")

    ps = sub.add_parser("static_profiles")
    add_common(ps)
    ps.add_argument("--left", type=int, default=10, help="Left charge position")
    ps.add_argument("--right", type=int, default=30, help="Right charge position")
    ps.add_argument("--q", type=float, default=1.0, help="Charge magnitude")

    pe = sub.add_parser("string_energy_scan")
    add_common(pe)
    pe.add_argument("--q", type=float, default=1.0)
    pe.add_argument("--Rmin", type=int, default=2, help="Minimum separation")
    pe.add_argument("--Rmax", type=int, default=20, help="Maximum separation")
    pe.add_argument("--Rstep", type=int, default=2, help="Separation step")

    pq = sub.add_parser("quench")
    add_common(pq)
    pq.add_argument("--left", type=int, default=10)
    pq.add_argument("--right", type=int, default=30)
    pq.add_argument("--q", type=float, default=1.0)
    pq.add_argument("--dt", type=float, default=0.05)
    pq.add_argument("--tmax", type=float, default=4.0)
    pq.add_argument("--meas_every", type=int, default=1)
    pq.add_argument("--tdvp_engine", type=str, default="two-site", choices=["two-site", "single-site"])

    args = p.parse_args()
    if args.cmd == "static_profiles":
        print("Running static profiles...")
        run_static_profiles(args)
    elif args.cmd == "string_energy_scan":
        print("Running string energy scan...")
        run_string_energy_scan(args)
    elif args.cmd == "quench":
        print("Running real-time quench dynamics...")
        run_real_time_quench(args)
    else:
        raise RuntimeError("Unknown command")

if __name__ == "__main__":
    main()
