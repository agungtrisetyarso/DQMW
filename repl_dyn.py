# ============================================================
# DQMW Figure: Reduction to Classical Replicator Dynamics
# Fully self-contained version (no external module needed)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------------------
# Helper functions (replacing verify_unified_generator)
# --------------------------------------------------------------------------------------

def vec(rho):
    """Column-stacking vectorization."""
    return rho.flatten(order='F')

def unvec(v, d):
    """Inverse of vec (column-stacking)."""
    return v.reshape((d, d), order='F')

def gibbs_state(Heff, eta):
    """Compute Gibbs state rho_G = exp(-eta * Heff) / Z."""
    E, V = np.linalg.eigh(Heff)
    weights = np.exp(-eta * E)
    Z = np.sum(weights)
    pG = weights / Z
    rho_G = (V * pG) @ V.conj().T
    return rho_G, E, V, pG

def build_gibbs_relaxation_liouvillian(Heff, eta, gamma0):
    """
    Build the Davies (heat-bath) Liouvillian superoperator for relaxation
    toward the Gibbs state of Heff.
    Uses the standard form with transition rates satisfying detailed balance.
    """
    d = Heff.shape[0]
    E, V = np.linalg.eigh(Heff)
    
    # Build transition rates (satisfy detailed balance)
    G = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                # Heat-bath rate: j -> i
                G[i, j] = gamma0 / (1.0 + np.exp(eta * (E[i] - E[j])))
    
    # Build Liouvillian in the energy eigenbasis
    L = np.zeros((d**2, d**2), dtype=complex)
    
    for i in range(d):
        for j in range(d):
            if i != j:
                # Jump term: L = sqrt(G[i,j]) * |i><j|
                rate = np.sqrt(G[i, j])
                Lij = rate * np.outer(V[:, i], V[:, j].conj())
                L += np.kron(Lij, Lij.conj()) - 0.5 * (
                    np.kron(np.eye(d), Lij.conj().T @ Lij) +
                    np.kron(Lij.T @ Lij.conj(), np.eye(d))
                )
    
    return L

def coherent_superop(Hdrift, d):
    """Coherent part -i[H, .] as superoperator."""
    I = np.eye(d, dtype=complex)
    return -1j * (np.kron(I, Hdrift) - np.kron(Hdrift.T, I))

# --------------------------------------------------------------------------------------
# Classical Pauli/replicator reference
# --------------------------------------------------------------------------------------
def pauli_master_solution(Heff, eta, gamma0, p0, times):
    E, V = np.linalg.eigh(Heff)
    d = len(E)
    G = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                G[i, j] = gamma0 / (1.0 + np.exp(eta * (E[i] - E[j])))
    M = G.copy()
    for i in range(d):
        M[i, i] = -np.sum(G[:, i])

    def rhs(t, p):
        return M @ p
    sol = solve_ivp(rhs, (times[0], times[-1]), p0, t_eval=times,
                    rtol=1e-10, atol=1e-12, method="RK45")
    return sol.y.T, E, V

def random_noncommuting_drift(d, scale, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = 0.5 * (A + A.conj().T)
    return scale * H / np.linalg.norm(H, 2)

def full_rho_populations(Heff, eta, gamma0, rho0, times, Hdrift=None):
    E, V = np.linalg.eigh(Heff)
    d = Heff.shape[0]
    Lsup = build_gibbs_relaxation_liouvillian(Heff, eta, gamma0)
    if Hdrift is not None:
        Lsup = Lsup + coherent_superop(Hdrift, d)
    pops = np.zeros((len(times), d))
    offdiag = np.zeros(len(times))
    v0 = vec(rho0)
    for k, t in enumerate(times):
        rho = unvec(expm(Lsup * t) @ v0, d)
        rho = 0.5 * (rho + rho.conj().T)
        r = V.conj().T @ rho @ V
        pops[k] = np.real(np.diag(r))
        offdiag[k] = np.linalg.norm(r - np.diag(np.diag(r)), "fro")
    return pops, offdiag, E, V

# --------------------------------------------------------------------------------------
# Main Figure Function
# --------------------------------------------------------------------------------------
def make_figure(d=4, eta=0.8, gamma0=12.0, seed=1, outfile="dqmw_fig2_unified.pdf"):
    rng = np.random.default_rng(seed)

    ell = rng.uniform(0.0, 1.0, size=d)
    Heff = np.diag(ell).astype(complex)

    rho_G, E, V, pG = gibbs_state(Heff, eta)

    # Initial state with coherences
    p0 = np.ones(d) / d
    rho0 = np.diag(p0).astype(complex)
    B = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    C = 0.15 * (B - np.diag(np.diag(B)))
    rho0 = rho0 + 0.5 * (C + C.conj().T)
    w, U = np.linalg.eigh(rho0)
    w = np.clip(w, 0, None)
    rho0 = (U * w) @ U.conj().T
    rho0 /= np.trace(rho0).real

    times = np.linspace(0, 1.2, 80)

    # Small coherent drift
    Hdrift = np.zeros((d, d), dtype=complex)
    Hdrift[0, 1] = 1.0
    Hdrift[1, 0] = 1.0

    pops_full, offdiag, E, V = full_rho_populations(
        Heff, eta, gamma0, rho0, times, Hdrift=Hdrift)
    p0_eig = np.real(np.diag(V.conj().T @ rho0 @ V))
    pops_cls, _, _ = pauli_master_solution(Heff, eta, gamma0, p0_eig, times)

    # O(1/gamma0) scaling
    def trace_distance(A, B):
        return 0.5 * np.sum(np.abs(np.linalg.eigvalsh(A - B)))

    gammas = np.array([4, 8, 16, 32, 64, 128, 256, 512], dtype=float)
    td_err, pop_err = [], []
    Lcoh = coherent_superop(Hdrift, d)
    for g in gammas:
        Lsup = build_gibbs_relaxation_liouvillian(Heff, eta, g) + Lcoh
        rho_ss = unvec(expm(Lsup * 15.0) @ vec(rho0), d)
        rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
        rho_ss /= np.trace(rho_ss).real
        td_err.append(trace_distance(rho_ss, rho_G))
        r = V.conj().T @ rho_ss @ V
        pop_err.append(np.linalg.norm(np.real(np.diag(r)) - pG, 1))
    td_err = np.array(td_err)
    pop_err = np.array(pop_err)

    # ---------------- Plot ----------------
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, d))

    for i in range(d):
        ax[0].plot(times, pops_cls[:, i], "-", color=cmap[i], lw=1.6)
        ax[0].plot(times[::4], pops_full[::4, i], "o", color=cmap[i], ms=3, mfc="white")
        ax[0].axhline(pG[i], color=cmap[i], ls=":", lw=1.0, alpha=0.7)
    ax[0].set_xlabel("time $t$")
    ax[0].set_ylabel("population $p_i(t)$")
    ax[0].set_title("(a) Relaxation to Gibbs weights\nfull $\\rho$ (○) vs replicator (—)")
    ax[0].grid(alpha=0.3)

    ax[1].semilogy(times, np.maximum(offdiag, 1e-12), color="tab:red", lw=1.8)
    ax[1].set_xlabel("time $t$")
    ax[1].set_ylabel(r"eigenbasis coherence $\|\rho - \mathrm{diag}_E\rho\|_F$")
    ax[1].set_title("(b) Coherences decay to an $O(1/\gamma_0)$ floor")
    ax[1].grid(alpha=0.3, which="both")

    ax[2].loglog(gammas, td_err, "o-", color="tab:blue", lw=1.6, ms=4,
                 label=r"$\|\rho_{\rm ss}-\rho_{\rm Gibbs}\|_1$")
    ax[2].loglog(gammas, td_err[0] * gammas[0] / gammas, "k--", lw=1.2,
                 label=r"$\propto 1/\gamma_0$ reference")
    ax[2].loglog(gammas, pop_err, "s:", color="tab:green", lw=1.2, ms=3, alpha=0.8,
                 label=r"population error ($\propto 1/\gamma_0^2$)")
    ax[2].set_xlabel(r"dissipation strength $\gamma_0$")
    ax[2].set_ylabel("deviation from Gibbs state")
    ax[2].set_title(r"(c) Steady-state deviation $\sim O(1/\gamma_0)$ (Lemma 1)")
    ax[2].legend(fontsize=7.5, frameon=False)
    ax[2].grid(alpha=0.3, which="both")

    fig.suptitle("Reduction of the unified DQMW generator to classical replicator dynamics\n"
                 "(strong-dissipation limit)", y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    fig.savefig(outfile.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")

    slope_td = np.polyfit(np.log(gammas), np.log(td_err), 1)[0]
    slope_pop = np.polyfit(np.log(gammas), np.log(pop_err), 1)[0]
    print(f"[fig2] wrote {outfile} and .png")
    print(f"[fig2] panel (c) trace-distance slope = {slope_td:.3f}  (expect ~ -1)")
    print(f"[fig2] panel (c) population-error slope = {slope_pop:.3f}  (expect ~ -2)")
    print(f"[fig2] final coherence = {offdiag[-1]:.2e}")
    print(f"[fig2] max |full - classical| populations = {np.max(np.abs(pops_full - pops_cls)):.2e}")
    return fig


if __name__ == "__main__":
    make_figure()
