#!/usr/bin/env python3
"""
Pure NetKet NQS VMC for the (spin-chain) Schwinger model 

The NN architecture is a complex-valued RBM. 


including:
- vacuum + string (static charges) energies
- optional ED (Lanczos) baseline
- optional saving + plotting of charge and E-field profiles (use with --save-profiles)

Matches TenPy conventions from TNS script
- static charges: qext[left]+=q, qext[right]-=q
- stagger sign: stag[n]=(+1 if n even else -1)

Example usage:

python nqs_schwinger.py static_profiles \
  --L 16 --x 1.0 --m 0.1 --g2 1.0 --q 1.0 --left 4 --right 12 \
  --do-ed --save-profiles --out profiles \
  --n-iter 80 --n-samples 1008 --n-chains 16 --alpha 2 --lr 0.05 --diag-shift 0.01

"""

import argparse
import os
import numpy as np

import netket as nk
import matplotlib.pyplot as plt



def make_static_charges(L, left, right, q=1.0):
   
    qext = np.zeros(L, dtype=float)
    qext[left] += q
    qext[right] -= q
    return qext


def electric_constant_offset(L, g2, qext, include_sz2_constant=False):
    """
   

    offset = (g2/2) * sum_{n=0}^{L-2} (D_n)^2
    where D_n = sum_{k=0}^{n} d_k, d_k = 0.5*stag_k + qext_k

    include_sz2_constant would add g2*L*(L-1)/16 (TenPy keeps it False in dmrg_ground_state).
    """
    qext = np.asarray(qext, dtype=float)
    if qext.shape != (L,):
        raise ValueError(f"qext must have shape (L,), got {qext.shape}")

    stag = np.array([1.0 if (n % 2 == 0) else -1.0 for n in range(L)], dtype=float)
    d = 0.5 * stag + qext

    D_links = np.cumsum(d)[:-1]  # links 0..L-2
    offset = 0.5 * float(g2) * float(np.sum(D_links ** 2))

    if include_sz2_constant:
        offset += float(g2) * L * (L - 1) / 16.0

    return offset


def schwinger_couplings(L, m, g2, qext):
    """
    Matches init_terms() coupling construction in TenPy model:
      stag = (-1)^n (n starting at 0)
      d = 0.5*stag + qext
      A_ij = max(0, (L-1) - max(i,j))
      Jij = g2*A
      h = g2*(A @ d)
    """
    qext = np.asarray(qext, dtype=float)
    if qext.shape != (L,):
        raise ValueError(f"qext must have shape (L,), got {qext.shape}")

    stag = np.array([1.0 if (n % 2 == 0) else -1.0 for n in range(L)], dtype=float)
    d = 0.5 * stag + qext

    A = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(L):
            A[i, j] = float(max(0.0, (L - 1) - max(i, j)))

    Jij = g2 * A
    h = g2 * (A @ d)
    mass = m * stag
    return stag, d, mass, h, Jij


def build_hamiltonian_netket(L, x, m, g2, qext, total_sz=None):
    """
    Builds NetKet operator for the same MPO terms as TPS Hamiltonian
    (excluding the additive constant which is added later with electric_constant_offset).

    Important basis note:
      NetKet Spin uses sigma^z eigenvalues ±1.
      TenPy model uses Sz = ±1/2.
      So:
        Sz = 0.5*sigmaz
        Sz_i Sz_j = 0.25*sigmaz_i*sigmaz_j
      The ladder operators (sigmap/sigmam) match the spin-1/2 S^± action.
    """
    if total_sz is None:
        hi = nk.hilbert.Spin(s=1 / 2, N=L)
    else:
        hi = nk.hilbert.Spin(s=1 / 2, N=L, total_sz=float(total_sz))

    stag, d, mass, h, Jij = schwinger_couplings(L, m, g2, qext)

    H = nk.operator.LocalOperator(hi)

    # onsite: (mass_i + h_i) * Sz_i = 0.5*(mass_i+h_i)*sigmaz_i
    for i in range(L):
        coeff = 0.5 * float(mass[i] + h[i])
        if abs(coeff) > 1e-14:
            H += coeff * nk.operator.spin.sigmaz(hi, i)

    # long-range ZZ: sum_{i<j} Jij_ij Sz_i Sz_j = 0.25*Jij_ij sigmaz_i sigmaz_j
    for i in range(L):
        for j in range(i + 1, L):
            Jij_ij = float(Jij[i, j])
            if abs(Jij_ij) > 1e-14:
                H += 0.25 * Jij_ij * (
                    nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, j)
                )

    # kinetic: -x * (Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
    for i in range(L - 1):
        H += (-float(x)) * (nk.operator.spin.sigmap(hi, i) @ nk.operator.spin.sigmam(hi, i + 1))
        H += (-float(x)) * (nk.operator.spin.sigmam(hi, i) @  nk.operator.spin.sigmap(hi, i + 1))

    return hi, H


# ----------------------------
# ED helper
# ----------------------------

def ed_ground_state_energy(H):
    """Lanczos ED (smallest eigenvalue)."""
    evals = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)
    return float(np.asarray(evals)[0])


# ----------------------------
# Observable + profile helpers (VMC)
# ----------------------------

def vmc_expect_sigmaz(vstate, L, imag_tol=1e-9):
    """Returns real <sigmaz_i> for i=0..L-1 from VMC state."""
    ops = [nk.operator.spin.sigmaz(vstate.hilbert, i) for i in range(L)]
    vals = np.empty(L, dtype=float)

    for i, op in enumerate(ops):
        stats = vstate.expect(op)
        mu = np.asarray(stats.mean).item()  # scalar, may be complex dtype
        if abs(np.imag(mu)) > imag_tol:
            print(f"[warn] <sigmaz[{i}]> imag part {np.imag(mu):.3e}; keeping real(mu)")
        vals[i] = float(np.real(mu))
    return vals


def profiles_from_sigmaz(sigmaz_exp, qext):
    """
    Reconstruct Q and E profiles analogous to TenPy code:
      Sz = 0.5*sigmaz
      Q_n = <Sz_n> + 0.5*stag_n
      E_n = sum_{k<=n}(<Sz_k> + d_k), links n=0..L-2
    """
    L = len(sigmaz_exp)
    stag = np.array([1.0 if (n % 2 == 0) else -1.0 for n in range(L)], dtype=float)
    Sz_exp = 0.5 * np.asarray(sigmaz_exp, dtype=float)
    Q = Sz_exp + 0.5 * stag

    d = 0.5 * stag + np.asarray(qext, dtype=float)
    cum = np.cumsum(Sz_exp + d)
    E_links = cum[:-1].copy()  # length L-1
    return Q, E_links


# ----------------------------
# Plot helpers
# ----------------------------

def ensure_dir_for(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_charge_profile(Q, title, filepath):
    ensure_dir_for(filepath)
    x = np.arange(len(Q))
    plt.figure()
    plt.plot(x, Q, marker="o")
    plt.xlabel("site n")
    plt.ylabel(r"$\langle Q_n \rangle$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def plot_efield_profile(E_links, title, filepath):
    ensure_dir_for(filepath)
    x = np.arange(len(E_links))  # links 0..L-2
    plt.figure()
    plt.plot(x, E_links, marker="o")
    plt.xlabel("link n")
    plt.ylabel(r"$\langle E_n \rangle$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


# ----------------------------
# Main routine: static_profiles (vacuum + string)
# ----------------------------

def run_static_profiles(args):
    L, x, m, g2 = args.L, args.x, args.m, args.g2
    left, right, q = args.left, args.right, args.q
    total_sz = args.total_sz if args.total_sz_enabled else None

    # Prepare qexts
    qext_vac = np.zeros(L, dtype=float)
    qext_str = make_static_charges(L, left, right, q=q)

    # Build operators (vac + string)
    hi_v, H_v = build_hamiltonian_netket(L, x, m, g2, qext_vac, total_sz=total_sz)
    hi_s, H_s = build_hamiltonian_netket(L, x, m, g2, qext_str, total_sz=total_sz)

    # Add missing constant offset (to match TenPy printed E0)
    off_v = electric_constant_offset(L, g2, qext_vac, include_sz2_constant=False)
    off_s = electric_constant_offset(L, g2, qext_str, include_sz2_constant=False)

    # --- ED (optional) ---
    if args.do_ed:
        Ev_ed_raw = ed_ground_state_energy(H_v)
        Es_ed_raw = ed_ground_state_energy(H_s)
        Ev_ed = Ev_ed_raw + off_v
        Es_ed = Es_ed_raw + off_s
        print(f"[ED vacuum] E0_raw = {Ev_ed_raw:.12f}   E0(+offset) = {Ev_ed:.12f}")
        print(f"[ED string] E0_raw = {Es_ed_raw:.12f}   E0(+offset) = {Es_ed:.12f}")
        print(f"[ED string] ΔE = {Es_ed - Ev_ed:.12f}")

    # --- VMC (NQS) ---
    graph = nk.graph.Chain(L, pbc=False)

    if args.sampler == "exchange":
        sa_v = nk.sampler.MetropolisExchange(hi_v, graph=graph, n_chains=args.n_chains)
        sa_s = nk.sampler.MetropolisExchange(hi_s, graph=graph, n_chains=args.n_chains)
    else:
        sa_v = nk.sampler.MetropolisLocal(hi_v, n_chains=args.n_chains)
        sa_s = nk.sampler.MetropolisLocal(hi_s, n_chains=args.n_chains)

    # NQS: complex RBM
    model = nk.models.RBM(alpha=args.alpha, param_dtype=complex)

    vstate_v = nk.vqs.MCState(sa_v, model, n_samples=args.n_samples, seed=args.seed)
    vstate_s = nk.vqs.MCState(sa_s, model, n_samples=args.n_samples, seed=args.seed + 1)

    opt = nk.optimizer.Adam(learning_rate=args.lr)

    drv_v = nk.driver.VMC_SR(H_v, opt, variational_state=vstate_v, diag_shift=args.diag_shift)
    drv_s = nk.driver.VMC_SR(H_s, opt, variational_state=vstate_s, diag_shift=args.diag_shift)

    print("\n[VMC] optimizing vacuum...")
    log_v = nk.logging.RuntimeLog()
    drv_v.run(args.n_iter, out=log_v)

    print("\n[VMC] optimizing string...")
    log_s = nk.logging.RuntimeLog()
    drv_s.run(args.n_iter, out=log_s)

    

    data_v = log_v.data
    # plot energy log for vacuum
    ev_means = np.asarray(data_v["Energy"]["Mean"])
    iterations = np.arange(len(ev_means))
    plt.figure()
    plt.plot(iterations, ev_means, label='Vacuum Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('VMC Vacuum Energy Optimization Log')
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig(f"{args.out}_vac_energy_log.png")
    plt.close()
    
    data_s = log_s.data
    # plot energy log for string
    es_means = np.asarray(data_s["Energy"]["Mean"])
    iterations = np.arange(len(es_means))   
    plt.figure()
    plt.plot(iterations, es_means, label='String Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('VMC String Energy Optimization Log')
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig(f"{args.out}_str_energy_log.png")
    plt.close()

    # output energy arrays to npz
    np.savez(
        args.out + "_vmc_energy_log.npz",
        ev_means=ev_means,
        es_means=es_means,
    )
    print(f"[saved] {args.out}_vmc_energy_log.npz")

         

    
    
    Ev_vmc_raw = float(np.asarray(log_v.data["Energy"]["Mean"])[-1])
    Es_vmc_raw = float(np.asarray(log_s.data["Energy"]["Mean"])[-1])
    Ev_vmc = Ev_vmc_raw + off_v
    Es_vmc = Es_vmc_raw + off_s

    print("\n[VMC vacuum] E0_raw =", f"{Ev_vmc_raw:.12f}", "  E0(+offset) =", f"{Ev_vmc:.12f}")
    print("[VMC string] E0_raw =", f"{Es_vmc_raw:.12f}", "  E0(+offset) =", f"{Es_vmc:.12f}")
    print("[VMC string] ΔE =", f"{(Es_vmc - Ev_vmc):.12f}")

    # --- Profiles + plots (optional) ---
    if args.save_profiles:
        sig_v = vmc_expect_sigmaz(vstate_v, L)
        sig_s = vmc_expect_sigmaz(vstate_s, L)

        Qv, Ev_links = profiles_from_sigmaz(sig_v, qext_vac)
        Qs, Es_links = profiles_from_sigmaz(sig_s, qext_str)

        np.savez(
            args.out + "_vmc_profiles.npz",
            Q_vac=Qv, E_vac=Ev_links,
            Q_str=Qs, E_str=Es_links,
            sigmaz_vac=sig_v, sigmaz_str=sig_s,
            qext_vac=qext_vac, qext_str=qext_str,
            energies=dict(
                Ev_vmc_raw=Ev_vmc_raw, Es_vmc_raw=Es_vmc_raw,
                Ev_vmc=Ev_vmc, Es_vmc=Es_vmc,
                off_v=off_v, off_s=off_s
            ),
            params=dict(
                L=L, x=x, m=m, g2=g2, left=left, right=right, q=q,
                alpha=args.alpha, n_samples=args.n_samples, n_chains=args.n_chains,
                n_iter=args.n_iter, lr=args.lr, diag_shift=args.diag_shift,
                sampler=args.sampler, total_sz=(total_sz if total_sz is not None else "none"),
            ),
        )
        print(f"[saved] {args.out}_vmc_profiles.npz")

        plot_charge_profile(Qv, f"Vacuum charge profile (L={L})", f"{args.out}_vac_charge.png")
        plot_charge_profile(Qs, f"String charge profile (L={L}, q={q}, [{left},{right}])", f"{args.out}_str_charge.png")
        plot_efield_profile(Ev_links, f"Vacuum E-field (L={L})", f"{args.out}_vac_efield.png")
        plot_efield_profile(Es_links, f"String E-field (L={L}, q={q}, [{left},{right}])", f"{args.out}_str_efield.png")

        print(f"[saved] {args.out}_vac_charge.png")
        print(f"[saved] {args.out}_str_charge.png")
        print(f"[saved] {args.out}_vac_efield.png")
        print(f"[saved] {args.out}_str_efield.png")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("static_profiles", help="Vacuum & string energies with NQS VMC (and optional ED).")

    ps.add_argument("--L", type=int, default=16)
    ps.add_argument("--x", type=float, default=1.0)
    ps.add_argument("--m", type=float, default=0.1)
    ps.add_argument("--g2", type=float, default=1.0)
    ps.add_argument("--left", type=int, default=4)
    ps.add_argument("--right", type=int, default=12)
    ps.add_argument("--q", type=float, default=1.0)

    ps.add_argument("--alpha", type=int, default=2, help="RBM density parameter.")
    ps.add_argument("--n-samples", type=int, default=8192)
    ps.add_argument("--n-chains", type=int, default=16)
    ps.add_argument("--n-iter", type=int, default=400)
    ps.add_argument("--lr", type=float, default=0.05)
    ps.add_argument("--diag-shift", type=float, default=0.01)
    ps.add_argument("--seed", type=int, default=0)

    ps.add_argument("--do-ed", action="store_true", help="Run ED (Lanczos) for vacuum and string.")
    ps.add_argument("--save-profiles", action="store_true", help="Save + plot Q and E profiles from VMC samples.")
    ps.add_argument("--out", type=str, default="profiles")

    ps.add_argument(
        "--sampler",
        type=str,
        choices=["exchange", "local"],
        default="exchange",
        help="exchange conserves total Sz; local can change magnetization.",
    )

    # By default, match TenPy's conserve='Sz' in the Neel sector: total Sz = 0 for even L.
    ps.add_argument("--total-sz", type=float, default=0.0, help="Total Sz sector (NetKet).")
    ps.add_argument(
        "--no-total-sz",
        action="store_true",
        help="Disable total_sz constraint in NetKet Hilbert space.",
    )

    args = p.parse_args()
    args.total_sz_enabled = (not args.no_total_sz)

    if args.cmd == "static_profiles":
        run_static_profiles(args)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
