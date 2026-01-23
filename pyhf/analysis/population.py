import numpy as np

class MullikenPopulation:

    @staticmethod
    def compute(rhf, basis_on_atoms=None):
        if not rhf.converged:
            raise RuntimeError("RHF calculation not converged")

        P = rhf.density_matrix
        S = rhf.S
        n_basis = rhf.n_basis
        n_atoms = rhf.molecule.n_atoms

        if basis_on_atoms is None:
            if n_basis != n_atoms:
                raise ValueError("basis_on_atoms must be provided when n_basis != n_atoms")
            basis_on_atoms = list(range(n_atoms))

        if len(basis_on_atoms) != n_basis:
            raise ValueError(f"basis_on_atoms length ({len(basis_on_atoms)}) must equal n_basis ({n_basis})")
        PS = P @ S

        gross_pop = np.zeros(n_atoms)
        for mu in range(n_basis):
            atom_mu = basis_on_atoms[mu]
            gross_pop[atom_mu] += PS[mu, mu]

        charges = np.zeros(n_atoms)
        for i, atom in enumerate(rhf.molecule.atoms):
            charges[i] = atom.atomic_number - gross_pop[i]

        bond_orders = np.zeros((n_atoms, n_atoms))
        for atom_i in range(n_atoms):
            for atom_j in range(atom_i + 1, n_atoms):
                bo = 0.0
                for mu in range(n_basis):
                    if basis_on_atoms[mu] != atom_i:
                        continue
                    for nu in range(n_basis):
                        if basis_on_atoms[nu] != atom_j:
                            continue
                        bo += PS[mu, nu] * PS[nu, mu]
                bond_orders[atom_i, atom_j] = bo
                bond_orders[atom_j, atom_i] = bo

        return {
            'gross_populations': gross_pop,
            'charges': charges,
            'bond_orders': bond_orders
        }

    @staticmethod
    def print_analysis(rhf, basis_on_atoms=None):
        results = MullikenPopulation.compute(rhf, basis_on_atoms)
        print("Mulliken analysis:")
        for i, atom in enumerate(rhf.molecule.atoms):
            print(f"{atom.symbol}{i+1}: pop={results['gross_populations'][i]:.6f} charge={results['charges'][i]:.6f}")
        if rhf.molecule.n_atoms > 1:
            for i in range(rhf.molecule.n_atoms):
                for j in range(i + 1, rhf.molecule.n_atoms):
                    bo = results['bond_orders'][i, j]
                    if abs(bo) > 0.01:
                        print(f"{rhf.molecule.atoms[i].symbol}{i+1}-{rhf.molecule.atoms[j].symbol}{j+1}: {bo:.4f}")


__all__ = ['MullikenPopulation']
