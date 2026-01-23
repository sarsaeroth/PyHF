import numpy as np
from pyhf.integrals.boys import boys_f0


def gaussian_product_center(alpha, A, beta, B):
    return (alpha * A + beta * B) / (alpha + beta)


def overlap_primitive(alpha, A, beta, B):
    gamma = alpha + beta
    diff = A - B
    R2 = np.dot(diff, diff)
    prefactor = (np.pi / gamma) ** 1.5
    exponential = np.exp(-alpha * beta * R2 / gamma)
    return prefactor * exponential


def kinetic_primitive(alpha, A, beta, B):
    gamma = alpha + beta
    diff = A - B
    R2 = np.dot(diff, diff)
    mu = alpha * beta / gamma
    S = overlap_primitive(alpha, A, beta, B)
    T = mu * (3.0 - 2.0 * mu * R2) * S
    return T


def nuclear_attraction_primitive(alpha, A, beta, B, C, Z):
    gamma = alpha + beta
    diff_AB = A - B
    R2_AB = np.dot(diff_AB, diff_AB)
    P = gaussian_product_center(alpha, A, beta, B)
    diff_PC = P - C
    R2_PC = np.dot(diff_PC, diff_PC)
    prefactor = -Z * 2.0 * np.pi / gamma
    exponential = np.exp(-alpha * beta * R2_AB / gamma)
    F0 = boys_f0(gamma * R2_PC)
    return prefactor * exponential * F0


def electron_repulsion_primitive(alpha, A, beta, B, gamma_c, C, delta, D):
    p = alpha + beta
    q = gamma_c + delta
    P = gaussian_product_center(alpha, A, beta, B)
    Q = gaussian_product_center(gamma_c, C, delta, D)
    diff_AB = A - B
    R2_AB = np.dot(diff_AB, diff_AB)
    diff_CD = C - D
    R2_CD = np.dot(diff_CD, diff_CD)
    diff_PQ = P - Q
    R2_PQ = np.dot(diff_PQ, diff_PQ)
    rho = p * q / (p + q)
    prefactor = 2.0 * (np.pi ** 2.5) / (p * q * np.sqrt(p + q))
    exp_AB = np.exp(-alpha * beta * R2_AB / p)
    exp_CD = np.exp(-gamma_c * delta * R2_CD / q)
    F0 = boys_f0(rho * R2_PQ)
    return prefactor * exp_AB * exp_CD * F0


def overlap_contracted(bf1, bf2):
    S = 0.0
    for i, (alpha, c_a) in enumerate(zip(bf1.exponents, bf1.coefficients)):
        N_a = bf1.primitives[i].norm
        for j, (beta, c_b) in enumerate(zip(bf2.exponents, bf2.coefficients)):
            N_b = bf2.primitives[j].norm
            S_prim = overlap_primitive(alpha, bf1.center, beta, bf2.center)
            S += c_a * c_b * N_a * N_b * S_prim
    return S


def kinetic_contracted(bf1, bf2):
    T = 0.0
    for i, (alpha, c_a) in enumerate(zip(bf1.exponents, bf1.coefficients)):
        N_a = bf1.primitives[i].norm
        for j, (beta, c_b) in enumerate(zip(bf2.exponents, bf2.coefficients)):
            N_b = bf2.primitives[j].norm
            T_prim = kinetic_primitive(alpha, bf1.center, beta, bf2.center)
            T += c_a * c_b * N_a * N_b * T_prim
    return T


def nuclear_attraction_contracted(bf1, bf2, nuclei):
    V = 0.0
    for i, (alpha, c_a) in enumerate(zip(bf1.exponents, bf1.coefficients)):
        N_a = bf1.primitives[i].norm
        for j, (beta, c_b) in enumerate(zip(bf2.exponents, bf2.coefficients)):
            N_b = bf2.primitives[j].norm
            for C, Z in nuclei:
                V_prim = nuclear_attraction_primitive(
                    alpha, bf1.center, beta, bf2.center, C, Z
                )
                V += c_a * c_b * N_a * N_b * V_prim
    return V


def electron_repulsion_contracted(bf1, bf2, bf3, bf4):
    ERI = 0.0
    for i, (alpha, c_a) in enumerate(zip(bf1.exponents, bf1.coefficients)):
        N_a = bf1.primitives[i].norm
        for j, (beta, c_b) in enumerate(zip(bf2.exponents, bf2.coefficients)):
            N_b = bf2.primitives[j].norm
            for k, (gamma, c_c) in enumerate(zip(bf3.exponents, bf3.coefficients)):
                N_c = bf3.primitives[k].norm
                for l, (delta, c_d) in enumerate(zip(bf4.exponents, bf4.coefficients)):
                    N_d = bf4.primitives[l].norm
                    ERI_prim = electron_repulsion_primitive(
                        alpha, bf1.center,
                        beta, bf2.center,
                        gamma, bf3.center,
                        delta, bf4.center
                    )
                    ERI += c_a * c_b * c_c * c_d * N_a * N_b * N_c * N_d * ERI_prim
    return ERI


class GaussianIntegrals:
    def __init__(self, molecule, basis_functions):
        self.molecule = molecule
        self.basis_functions = basis_functions
        self.n_basis = len(basis_functions)
        self.nuclei = [
            (atom.position, atom.atomic_number)
            for atom in molecule.atoms
        ]
        self._compute_one_electron()
        self._compute_two_electron()

    def _compute_one_electron(self):
        n = self.n_basis
        self.S = np.zeros((n, n))
        self.T = np.zeros((n, n))
        self.V = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                bf_i = self.basis_functions[i]
                bf_j = self.basis_functions[j]

                self.S[i, j] = overlap_contracted(bf_i, bf_j)
                self.S[j, i] = self.S[i, j]

                self.T[i, j] = kinetic_contracted(bf_i, bf_j)
                self.T[j, i] = self.T[i, j]

                self.V[i, j] = nuclear_attraction_contracted(bf_i, bf_j, self.nuclei)
                self.V[j, i] = self.V[i, j]

    def _compute_two_electron(self):
        n = self.n_basis
        self.ERI = np.zeros((n, n, n, n))

        for i in range(n):
            for j in range(i + 1):
                for k in range(n):
                    for l in range(k + 1):
                        if (i * (i + 1) // 2 + j) >= (k * (k + 1) // 2 + l):
                            bf_i = self.basis_functions[i]
                            bf_j = self.basis_functions[j]
                            bf_k = self.basis_functions[k]
                            bf_l = self.basis_functions[l]

                            val = electron_repulsion_contracted(bf_i, bf_j, bf_k, bf_l)

                            self.ERI[i, j, k, l] = val
                            self.ERI[j, i, k, l] = val
                            self.ERI[i, j, l, k] = val
                            self.ERI[j, i, l, k] = val
                            self.ERI[k, l, i, j] = val
                            self.ERI[l, k, i, j] = val
                            self.ERI[k, l, j, i] = val
                            self.ERI[l, k, j, i] = val

    def as_dict(self):
        return {
            'S': self.S,
            'T': self.T,
            'V': self.V,
            'ERI': self.ERI
        }


def compute_integrals(molecule, basis_name='sto-3g'):
    from pyhf.integrals.basis import build_basis
    basis_functions, basis_to_atom = build_basis(molecule, basis_name)
    integrals = GaussianIntegrals(molecule, basis_functions)
    return integrals.as_dict(), basis_to_atom


__all__ = [
    'overlap_primitive',
    'kinetic_primitive',
    'nuclear_attraction_primitive',
    'electron_repulsion_primitive',
    'overlap_contracted',
    'kinetic_contracted',
    'nuclear_attraction_contracted',
    'electron_repulsion_contracted',
    'GaussianIntegrals',
    'compute_integrals',
]
