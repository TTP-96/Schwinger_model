#!/usr/bin/env python3
"""
Exact diagonalization of the Schwinger model (QED in 1+1D)  (open BC, staggered fermions)
States are NumPy arrays of 0/1 occupations. Could speed up by using bit operations

Hamiltonian (dimensionless units):
  H = -w ∑_{n=0}^{N-2} (c†_n c_{n+1} + h.c.) + m ∑_{n=0}^{N-1} (-1)^n (n_n - 1/2) + (g^2/2) ∑_{n=0}^{N-2} L_n^2
 
with Gauss's law (open BC):
  L_n = l0 + ∑_{k=0}^n (n_k - b_k),   b_k = (1 - (-1)^k)/2,   l0 = theta/(2π)

b_k is the background charge on odd sites (0 on even sites, 1 on odd sites).


Requires: numpy, scipy

Example usage:
To check value of chiral condensate vs theory in massless case:

python main_naive.py   --N 12 --m 0.0 --g 1.0 --w 1.0 --q 1.0 --theta 0.0 --k 4 --R_star 8



python main.py   --N 12 --m 0.5 --g 0.1 --w 1.0 --q 1.0 --theta 0.0 --k 4 --R_star 6 # No string breaking, linear potential

python main.py   --N 12 --m 0.5 --g 1.0 --w 1.0 --q 0.8 --theta 0.0 --k 4 --R_star 6 # String breaking observed
"""

from __future__ import annotations
import argparse
import math
from itertools import product

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


def generate_basis(N: int):
    """
    Return (basis, index_map) where:
      - basis is a list of numpy arrays of shape (N,) with occupations 0/1.
      - index_map maps the tuple of occupations (0/1) to the index in the basis list.
    """

    # use itertools to generate all possible occupation configurations
    # product([0, 1], repeat=N) generates all combinations of 0/1 for N sites
    basis = [np.array(config, dtype=np.int8) for config in product([0, 1], repeat=N)]

    index_map = {tuple(s.tolist()): i for i, s in enumerate(basis)}
    #print(f"index_map={index_map}")  # Debug: show index mapping
    
    return basis, index_map


def background_array(N: int) -> np.ndarray:
    """b_n = (1 - (-1)^n)/2 -> 0 on even sites, 1 on odd sites."""
    n = np.arange(N)
    return ((1 - (-1) ** n) // 2).astype(np.int8) # int8 for cache efficiency



def external_charges_vector(N: int, i_left: int, i_right: int, q: float) -> np.ndarray:
    """
    Make a length-N array ext_q with +q at i_left and -q at i_right.
    (Net external charge = 0 to avoid runaway fields with OBC.)
    """
    # print(f"i_right= {i_right}, N={N}")
    # print(f"i_left= {i_left}, N={N}")
     
    ext_q = np.zeros(N, dtype=np.float64)
    ext_q[i_left] += q
    ext_q[i_right] -= q
    return ext_q


def centered_pair_sites(N: int, R: int) -> tuple[int, int]:
    # Place +q and -q symmetrically, leaving R sites between them
    # assert R < N, "R must be less than N"
    assert R >= 0, "R must be non-negative"    
    i_left = (N - 1 - R) // 2    
    i_right = i_left + R
    assert 0 <= i_left < N, "i_left out of bounds. Check that R < N/2."
    assert 0 <= i_right < N, "i_right out of bounds. Check that R < N/2."
    
    return i_left, i_right


def theta_links_from_pair(N: int, i_left: int, i_right: int, q: float, theta0: float = 0.0) -> np.ndarray:
    """
    TODO: Finalize this, still a bit rough.
    Return l0_links[n] = ϑ_n/(2π).
    ϑ_n = theta0 + 2π q on links strictly between i_left and i_right,
           theta0 elsewhere. Length = N-1.
    """
    l0_links = np.full(N - 1, theta0 / (2.0 * math.pi), dtype=np.float64)
    # links n run between sites (n, n+1)
    start = min(i_left, i_right)
    stop  = max(i_left, i_right)  # last site; last included link is stop-1
    if stop > start:
        l0_links[start:stop] += q   # add +q on links between the charges
    return l0_links

def electric_fields(state: np.ndarray,
                    b: np.ndarray,
                    l0: float,
                    l0_links: np.ndarray | None = None,
                    ext_q: np.ndarray | None = None) -> np.ndarray:
    """
    Compute link electric fields L_n (n=0..N-2).

    Gauss's law: L_n = ℓ0_n + sum_{k=0}^n (n_k - b_k + q_ext,k)
      - If l0_links is None: ℓ0_n = l0 (same on all links)
      - Else:    ℓ0_n := l0_links[n] (array of length N-1)

      - ext_q is an array of length N with external static charges at sites (can be fractional).
    """
    N = state.size
    # effective site charge: q_eff = n - b (+ external sources if present)
    q_eff = state.astype(np.float64) - b.astype(np.float64)
    if ext_q is not None:
        q_eff = q_eff + ext_q.astype(np.float64)

    csum = np.cumsum(q_eff)          # length N
    L = csum[:-1]                    # partial sums for links 0..N-2

    if l0_links is None:
        return l0 + L
    else:
        # l0_links must be length N-1
        return np.asarray(l0_links, dtype=np.float64) + L

# def electric_fields(state: np.ndarray, b: np.ndarray, l0: float) -> np.ndarray:
#     """
#     Compute all link electric fields L_n (n=0..N-2) for given occupation 'state'.
#     L_n = l0 + sum_{k=0}^n (n_k - b_k).
#     """
#     # promote to int16 to avoid overflow in large N. Cost negligible compared
#     # to basis generation and Hamiltonian construction.
    
#     q = state.astype(np.int16) - b.astype(np.int16)       # effective charges
#     csum = np.cumsum(q)                                   # size N
#     return l0 + csum[:-1]                                 # N-1 links


def diag_energy(state: np.ndarray,
                m: float, g: float, l0: float, b: np.ndarray,
                l0_links: np.ndarray | None = None,
                ext_q: np.ndarray | None = None) -> float:
    # Mass term
    N = state.size
    mass = m * np.sum(((-1.0) ** np.arange(N)) * (state - 0.5))
    # Electric field
    L = electric_fields(state, b, l0, l0_links=l0_links, ext_q=ext_q)
    efield = 0.5 * g * g * float(np.dot(L, L))
    return float(mass + efield)
# def build_hamiltonian(N: int, m: float, g: float, w: float, theta: float) -> csr_matrix:
#     """
#     Build sparse Hamiltonian in the array occupation basis (size 2^N).
#     Hopping is implemented by swapping entries [1,0] <-> [0,1] at neighbors.
#     """
#     basis, index_map = generate_basis(N)
#     dim = len(basis)
#     b = background_array(N)
#     l0 = theta / (2.0 * math.pi)

#     rows = []
#     cols = []
#     data = []

#     # Diagonal
#     for idx, s in enumerate(basis):
#         e = diag_energy(s, m, g, l0, b)
#         rows.append(idx)
#         cols.append(idx)
#         data.append(e)

#     # Off-diagonal hopping: -w * ∑ (c†_n c_{n+1} + h.c.)  → swap [1,0] ↔ [0,1]
#     for idx, s in enumerate(basis):
#         for n in range(N - 1):
#             left, right = s[n], s[n + 1]
#             if left != right:
#                 # swap occupations at n and n+1
#                 s2 = s.copy()
#                 s2[n], s2[n + 1] = right, left
#                 jdx = index_map[tuple(s2.tolist())]
#                 rows.append(jdx)
#                 cols.append(idx)
#                 data.append(-w)

#     H = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()
#     # Symmetrize 
#     H = 0.5 * (H + H.T)
#     return H


def build_hamiltonian(N: int, m: float, g: float, w: float, theta: float,
                      ext_q: np.ndarray | None = None,
                      l0_links: np.ndarray | None = None) -> csr_matrix:
    """
    Build sparse H in occupation basis.
      - ext_q: length-N array of external static charges at sites (default zeros)
      - l0_links: length-(N-1) array of per-link ℓ0_n = ϑ_n/(2π).
                  If None, we use uniform l0 = theta/(2π).
      - w: hopping amplitude
    """
    basis, index_map = generate_basis(N)
    dim = len(basis)
    b = background_array(N)
    l0 = theta / (2.0 * math.pi)

    rows, cols, data = [], [], []

    # Diagonal
    for idx, s in enumerate(basis):
        e = diag_energy(s, m, g, l0, b, l0_links=l0_links, ext_q=ext_q)
        rows.append(idx); cols.append(idx); data.append(e)

    # Off-diagonal hopping (nearest-neighbor)
    for idx, s in enumerate(basis):
        for n in range(N - 1):
            left, right = s[n], s[n + 1]
            if left != right:
                s2 = s.copy()
                s2[n], s2[n + 1] = right, left
                jdx = index_map[tuple(s2.tolist())]
                rows.append(jdx); cols.append(idx); data.append(-w)

    H = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()
    H = 0.5 * (H + H.T)
    return H


def ground_state(H: csr_matrix, k: int = 2):
    """k lowest eigenpairs via Lanczos algo"""
    vals, vecs = eigsh(H, k=k, which="SA", tol=1e-10, maxiter=200000) 
    order = np.argsort(vals)
    return vals[order], vecs[:, order]



def chiral_condensate_from_occ(occ: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Return (per-site condensate, site-resolved staggered density).
    occ[i] must be ⟨n_i⟩.
    """
    N = occ.size
    stagger = (-1.0) ** np.arange(N)
    s_i = stagger * (occ - 0.5)          # site-resolved contribution
    sigma = s_i.mean()                    # per-site condensate
    return sigma, s_i



def static_potential_vs_R(N: int, m: float, g: float, w: float, theta0: float,
                          q: float, R_list: list[int],
                          use_theta_links: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute V(R) = E0(R) - E0(0) for a probe pair ±q separated by R sites.
    - use_theta_links=False: uses site sources ext_q (+q at left, -q at right)
    - use_theta_links=True:  uses link-dependent l0_links (paper's ϑ_n)
    Returns (R_array, V_array).
    """
    # Reference (no probes): E_ref
    H0 = build_hamiltonian(N, m, g, w, theta0, ext_q=None, l0_links=None)
    E_ref = ground_state(H0, k=1)[0][0]

    V = []
    for R in R_list:
        iL, iR = centered_pair_sites(N, R)
        if use_theta_links:
            l0_links = theta_links_from_pair(N, iL, iR, q, theta0)
            H = build_hamiltonian(N, m, g, w, theta0, ext_q=None, l0_links=l0_links)
        else:
            ext_q = external_charges_vector(N, iL, iR, q)
            H = build_hamiltonian(N, m, g, w, theta0, ext_q=ext_q, l0_links=None)

        E = ground_state(H, k=1)[0][0]
        V.append(E - E_ref)
    return np.asarray(R_list), np.asarray(V, dtype=np.float64)

def gs_observables(N: int, vec: np.ndarray, m: float, g: float, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (for the GS)
    (1) avg of occupation at site i ⟨n_i⟩
    (2) avg effective charge ⟨q_i⟩=⟨n_i - b_i⟩
    (3) exp val of electric link var ⟨L_n⟩ 
    (Simple diagonal estimators over basis.)
    """
    basis, _ = generate_basis(N)
    b = background_array(N)
    l0 = theta / (2.0 * math.pi)
    
    # probs[i] = |⟨s_i|GS⟩|^2, where s_i is the i-th basis state
    probs = np.abs(vec) ** 2


    occ = np.zeros(N, dtype=float)
    qexp = np.zeros(N, dtype=float)
    Lexp = np.zeros(N - 1, dtype=float)

    for idx, s in enumerate(basis):
        p = probs[idx] # weight of GS projected on idx basis state 
        if p == 0.0:
            continue
        occ += p * s
        q = s.astype(float) - b
        qexp += p * q
        Lexp += p * electric_fields(s, b, l0)

    # Compute the chiral condensate <psi_bar psi>
    #cc,cc_i=comp_chiral_condensate(vec, m, g, theta)
    cc,cc_i = chiral_condensate_from_occ(occ)
    print("m=", m)
    if m<=0.1:
        print("Chiral condensate (from occupations) <psi_bar psi> = {:.6f}\t Theory=0.1599*g*cos(theta)={}".format(cc,   -0.1599 * g * math.cos(theta)))
    elif m > 4.0:
        print("Chiral condensate (from occupations) <psi_bar psi> = {:.6f}\t Theory=-1/(4*pi*m)={}".format(cc,   -1.0/(2.0*math.pi*m)))
    # cc,cc_i = comp_chiral_condensate(probs)
    # print occupations
    print("  <n_i> =", np.array_str(occ, precision=10))
    print("Chiral condensate (from occupations) <psi_bar psi> = {:.6f}".format(cc))
    return occ, qexp, Lexp



def flux_and_charge_profiles(N, vec, theta0, q=None, iL=None, iR=None, use_theta_links=False):
    basis, _ = generate_basis(N)
    b = background_array(N)
    probs = np.abs(vec)**2

    if use_theta_links:
        # Only create per-link offsets when q, iL, iR are all specified
        if (q is not None) and (iL is not None) and (iR is not None):
            l0_links = theta_links_from_pair(N, iL, iR, float(q), theta0)
        else:
            l0_links = None  # no probes -> no link shifts
        ext_q = None
        l0 = theta0 / (2*np.pi)
    else:
        l0_links = None
        ext_q = external_charges_vector(N, iL, iR, float(q)) if (q is not None) else None
        l0 = theta0 / (2*np.pi)

    occ = np.zeros(N)
    qeff = np.zeros(N)
    Lavg = np.zeros(N-1)
    for idx, s in enumerate(basis):
        p = probs[idx]
        if p == 0.0:
            continue
        occ += p * s
        eff = s.astype(float) - b.astype(float)
        if ext_q is not None:
            eff += ext_q
        qeff += p * eff
        Lavg += p * electric_fields(s, b, l0, l0_links=l0_links, ext_q=ext_q)
    return occ, qeff, Lavg


def induced_profiles_at_R(N, m, g, w, theta0, q, R,
                          use_theta_links=False):
    # Reference (no probes)
    H0 = build_hamiltonian(N, m, g, w, theta0, ext_q=None, l0_links=None)
    vals0, vecs0 = ground_state(H0, k=1)
    gs0 = vecs0[:, 0]

    # With probes at separation R
    iL, iR = centered_pair_sites(N, R)
    if use_theta_links:
        l0_links = theta_links_from_pair(N, iL, iR, q, theta0)
        H = build_hamiltonian(N, m, g, w, theta0, ext_q=None, l0_links=l0_links)
        vals, vecs = ground_state(H, k=1)
        gs = vecs[:, 0]
        occ0, q0, L0 = flux_and_charge_profiles(N, gs0, theta0,
                                                q=None, iL=iL, iR=iR,
                                                use_theta_links=True)
        occ,  q1, L1 = flux_and_charge_profiles(N, gs, theta0,
                                                q=None, iL=iL, iR=iR,
                                                use_theta_links=True)
    else:
        ext_q = external_charges_vector(N, iL, iR, q)
        H = build_hamiltonian(N, m, g, w, theta0, ext_q=ext_q, l0_links=None)
        vals, vecs = ground_state(H, k=1)
        gs = vecs[:, 0]
        occ0, q0, L0 = flux_and_charge_profiles(N, gs0, theta0,
                                                q=None, iL=iL, iR=iR,
                                                use_theta_links=False)
        occ,  q1, L1 = flux_and_charge_profiles(N, gs, theta0,
                                                q=q, iL=iL, iR=iR,
                                                use_theta_links=False)

    # print GS energies for debugging
    print(f"E0 (no probes) = {vals0[0]:.12f}, E (R={R}) = {vals[0]:.12f}")
    

    dq = q1 - q0   # induced charge
    dL = L1 - L0   # induced flux
    return (iL, iR), dq, dL


def main():
    parser = argparse.ArgumentParser(description="Array-based ED for the lattice Schwinger model (no bit ops).")
    parser.add_argument("--N", type=int, default=8, help="Number of staggered sites (even recommended).")
    parser.add_argument("--m", type=float, default=0.0, help="Fermion mass.")
    parser.add_argument("--g", type=float, default=1.0, help="Gauge coupling.")
    parser.add_argument("--w", type=float, default=1.0, help="Hopping amplitude.")
    parser.add_argument("--theta", type=float, default=0.0, help="Theta angle; l0 = theta/(2π).")
    parser.add_argument("--k", type=int, default=2, help="Number of lowest eigenpairs to compute.")
    parser.add_argument("--q", type=float, default=0.5, help="Static charge at the pair (default 0.5).")
    parser.add_argument("--R_star", type=int, default=4, help="Separation R* for profiles (default 4).")
    parser.add_argument("--plot", action="store_true", help="Plot results.")
    args = parser.parse_args()



    # Parameters

    #m, g, w, theta0 = 0.0, 0.4, 1.0, 0.0
    N, m, g, w, theta, k,q = args.N, args.m, args.g, args.w, args.theta, args.k,args.q, 
    theta0 = theta  # theta0 is the knob for the Hamiltonian
    R_star= args.R_star  # separation for profiles
#    q = 2.0

    R_list = list(range(0, N-1))  # all possible separations

    use_theta_links = False   # set True to use ϑ_n formulation

    print("Computing ground state observables")
    # compute GS observables

    H = build_hamiltonian(N, m, g, w, theta)

    print("Diagonalizing ...")
    evals, evecs = ground_state(H, k=max(1, k))
    for i, E in enumerate(evals):
        print(f"E[{i}] = {E:.12f}")
        
    gs = evecs[:, 0]
    occ, qexp, Lexp = gs_observables(N, gs, m, g, theta)

    
    print("Mass=", m, "g=", g, "w=", w, "theta0=", theta0, "q=", q)
    R, V = static_potential_vs_R(N, m, g, w, theta0, q, R_list, use_theta_links=True)


    # Three separations around observed Rc (example: 3, 4, 5)
    #R_triplet = [3, 4, 5]   
    R_triplet = [6,7,8,9]   
    profiles = [induced_profiles_at_R(N, m, g, w, theta0, q, R,
                                      use_theta_links=use_theta_links)
                for R in R_triplet]


    
    import matplotlib.pyplot as plt
    x_sites = np.arange(N)
    x_links = np.arange(N-1) + 0.5

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    colors=["red", "blue", "green","purple","orange"]


    # Induced charge Δq_n
    for col_count, ((iL, iR), dq, dL) in enumerate(profiles):
        offset = 0.1 * col_count   # shift each curve slightly along x
        markerline, stemlines, baseline = axs[0].stem(
            x_sites + offset, dq, basefmt=' ', label=f'R={iR-iL}'
    )
        plt.setp(markerline, color=colors[col_count], marker='o')
        plt.setp(stemlines, color=colors[col_count])

    axs[0].set_ylabel('Δ⟨q_n⟩')
    axs[0].set_title('Induced charge (with probes minus no probes)')
    axs[0].grid(True); axs[0].legend()

    # Induced flux ΔL_n
    col_count=0
    for (iL, iR), dq, dL in profiles:
        axs[1].plot(x_links, dL, marker='o', label=f'R={iR-iL}', color=colors[col_count])
        col_count+=1
    axs[1].set_xlabel('Link position'); axs[1].set_ylabel('Δ⟨L_n⟩')
    axs[1].set_title('Induced flux (string region vs outside)')
    axs[1].grid(True); axs[1].legend()

    plt.tight_layout(); 
    plt.savefig("induced_profiles.png", dpi=300)
    if args.plot:
        plt.show()
    
    #plot static potential V(R)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(R, V, marker='o', label=f"V(R) for q={q}")
    plt.xlabel("Separation R (sites)")
    plt.ylabel("Static potential V(R)")
    plt.title("Static potential V(R) vs separation R")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("static_potential_VR.png", dpi=300)
    if args.plot:
        plt.show()
    
    
    # print("R   V(R)")
    # for r, v in zip(R, V):
    #     print(f"{r:2d}  {v:.6f}")

    # # ---------- 2) Profiles at one R* ----------
    # # Build the Hamiltonian at R_star and get the GS vector.
    # iL, iR = centered_pair_sites(N, R_star)
    # if use_theta_links:
    #     # link-theta (paper) route
    #     l0_links = theta_links_from_pair(N, iL, iR, q, theta0)
    #     H = build_hamiltonian(N, m, g, w, theta0, ext_q=None, l0_links=l0_links)
    # else:
    #     # site-source (Gauss law) route
    #     ext_q = external_charges_vector(N, iL, iR, q)
    #     H = build_hamiltonian(N, m, g, w, theta0, ext_q=ext_q, l0_links=None)

    # vals, vecs = ground_state(H, k=1)
    # gs_vec = vecs[:, 0]

    # # Compute profiles
    # occ, qeff, Lavg = flux_and_charge_profiles(
    #     N, gs_vec, theta0,
    #     q=(q if not use_theta_links else None),
    #     iL=iL, iR=iR, use_theta_links=use_theta_links
    # )

    # # ---------- Plot profiles ----------
    # x_sites = np.arange(N)
    # x_links = np.arange(N-1) + 0.5  # place links between sites

    # fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # # Effective charge density (includes external charges if site-source route)
    # ax[0].stem(x_sites, qeff, basefmt=' ', use_line_collection=True)
    # ax[0].set_ylabel(r'$\langle q_n \rangle$')
    # ax[0].set_title(f'Profiles at R={R_star} (iL={iL}, iR={iR})')
    # ax[0].grid(True)

    # # Electric field on links
    # ax[1].plot(x_links, Lavg, marker='s')
    # ax[1].set_xlabel('Site / Link position')
    # ax[1].set_ylabel(r'$\langle L_n \rangle$')
    # ax[1].grid(True)

    # plt.tight_layout()
    # plt.show()

    # # (Optional) Print a small table
    # print("\nR    V(R)")
    # for r, v in zip(R, V):
    #     print(f"{r:2d}  {v:.6f}")

if __name__ == "__main__":
    main()
