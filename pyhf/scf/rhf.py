import numpy as np
from pyhf.utils.linalg import symmetric_orthogonalization
from pyhf.utils.constants import (
    SCF_ENERGY_THRESHOLD,
    SCF_DENSITY_THRESHOLD,
    SCF_MAX_ITERATIONS
)


class RHF:
    def __init__(self, molecule, integrals):
        self.molecule = molecule
        self.integrals = integrals

        self.S = integrals['S']  # Overlap
        self.T = integrals['T']  # Kinetic
        self.V = integrals['V']  # Nuclear attraction
        self.ERI = integrals['ERI']  # Electron repulsion

        self.n_basis = self.S.shape[0]

        if molecule.n_electrons % 2 != 0:
            raise ValueError("RHF requires an even number of electrons (closed-shell)")
        self.n_occ = molecule.n_electrons // 2

        self.energy = None
        self.e_electronic = None
        self.orbital_energies = None
        self.mo_coefficients = None
        self.density_matrix = None
        self.fock_matrix = None
        self.converged = False
        self.scf_iterations = 0

        self.H_core = self.T + self.V
        self.X = symmetric_orthogonalization(self.S)

    def run(self, use_diis=False,
            max_iter=SCF_MAX_ITERATIONS,
            e_thresh=SCF_ENERGY_THRESHOLD,
            d_thresh=SCF_DENSITY_THRESHOLD):
        F = self.H_core.copy()
        F_prime = self.X.T @ F @ self.X
        eps, C_prime = np.linalg.eigh(F_prime)
        C = self.X @ C_prime
        P = self._build_density_matrix(C)
        if use_diis:
            from pyhf.scf.diis import DIIS
            diis = DIIS(max_size=6)
        e_old = 0.0
        for iteration in range(1, max_iter + 1):
            F = self._build_fock_matrix(P)
            if use_diis:
                F = diis.update(F, P, self.S)
            F_prime = self.X.T @ F @ self.X
            eps, C_prime = np.linalg.eigh(F_prime)
            C = self.X @ C_prime
            P_new = self._build_density_matrix(C)
            E = self._compute_energy(P_new, F)
            delta_E = abs(E - e_old)
            delta_P = np.max(np.abs(P_new - P))
            if delta_E < e_thresh and delta_P < d_thresh:
                self.converged = True
                break
            P = P_new
            e_old = E

        self.scf_iterations = iteration

        if not self.converged:
            raise RuntimeError(
                f"SCF did not converge in {max_iter} iterations. "
                f"Final ΔE = {delta_E:.2e}, ΔP = {delta_P:.2e}"
            )

        self.orbital_energies = eps
        self.mo_coefficients = C
        self.density_matrix = P_new
        self.fock_matrix = F
        self.e_electronic = E
        self.energy = E + self.molecule.nuclear_repulsion_energy()
        return self.energy

    def _build_density_matrix(self, C):
        C_occ = C[:, :self.n_occ]
        P = 2.0 * C_occ @ C_occ.T
        return P

    def _build_fock_matrix(self, P):
        n = self.n_basis
        F = self.H_core.copy()

        for mu in range(n):
            for nu in range(n):
                for lam in range(n):
                    for sig in range(n):
                        coulomb = self.ERI[mu, nu, lam, sig]
                        exchange = self.ERI[mu, lam, nu, sig]
                        F[mu, nu] += P[lam, sig] * (coulomb - 0.5 * exchange)

        return F

    def _compute_energy(self, P, F):
        E = 0.5 * np.sum(P * (self.H_core + F))
        return E

    def get_molecular_orbitals(self):
        if not self.converged:
            raise RuntimeError("SCF not converged. Run calculation first.")

        occupations = np.zeros(self.n_basis)
        occupations[:self.n_occ] = 2.0

        return {
            'energies': self.orbital_energies,
            'coefficients': self.mo_coefficients,
            'occupations': occupations
        }

__all__ = ['RHF']
