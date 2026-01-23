import numpy as np

class DIIS:
    def __init__(self, max_size=6):
        if max_size < 2:
            raise ValueError("DIIS max_size must be at least 2")
        self.max_size = max_size
        self.fock_matrices = []
        self.error_vectors = []

    def update(self, F, P, S):
        FPS = F @ P @ S
        SPF = S @ P @ F
        error = FPS - SPF

        self.fock_matrices.append(F.copy())
        self.error_vectors.append(error.copy())

        if len(self.fock_matrices) > self.max_size:
            self.fock_matrices.pop(0)
            self.error_vectors.pop(0)

        if len(self.fock_matrices) < 2:
            return F

        n = len(self.error_vectors)
        B = np.zeros((n + 1, n + 1))

        for i in range(n):
            for j in range(n):
                B[i, j] = np.sum(self.error_vectors[i] * self.error_vectors[j])

        B[n, :n] = -1.0
        B[:n, n] = -1.0
        B[n, n] = 0.0

        rhs = np.zeros(n + 1)
        rhs[n] = -1.0

        try:
            coeff = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return F

        F_diis = np.zeros_like(F)
        for i in range(n):
            F_diis += coeff[i] * self.fock_matrices[i]

        return F_diis

    def reset(self):
        self.fock_matrices = []
        self.error_vectors = []

    def __len__(self):
        return len(self.fock_matrices)


__all__ = ['DIIS']
