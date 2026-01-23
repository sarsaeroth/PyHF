import numpy as np
from pyhf.molecule.atom import Atom

class Molecule:
    def __init__(self, atoms, charge=0, multiplicity=1):
        if not isinstance(atoms, list):
            raise TypeError("atoms must be a list")
        if not all(isinstance(a, Atom) for a in atoms):
            raise TypeError("All elements in atoms must be Atom objects")
        if len(atoms) == 0:
            raise ValueError("Molecule must have at least one atom")

        self.atoms = atoms
        self.charge = charge
        self.multiplicity = multiplicity
        self.n_atoms = len(atoms)
        total_nuclear_charge = sum(atom.atomic_number for atom in atoms)
        self._n_electrons = total_nuclear_charge - charge

        if self._n_electrons < 0:
            raise ValueError("Invalid charge: results in negative electrons")

        n_unpaired = multiplicity - 1
        if (self._n_electrons - n_unpaired) % 2 != 0:
            raise ValueError(
                f"Incompatible multiplicity {multiplicity} for "
                f"{self._n_electrons} electrons"
            )

    @property
    def n_electrons(self):
        return self._n_electrons

    def nuclear_repulsion_energy(self):
        e_nuc = 0.0
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                atom_i = self.atoms[i]
                atom_j = self.atoms[j]
                distance = atom_i.distance_to(atom_j)
                e_nuc += atom_i.atomic_number * atom_j.atomic_number / distance

        return e_nuc

    def center_of_mass(self):
        total_mass = sum(atom.mass for atom in self.atoms)
        com = np.zeros(3)
        for atom in self.atoms:
            com += atom.mass * atom.position
        return com / total_mass

    def __repr__(self):
        return f"Molecule({len(self.atoms)} atoms, charge={self.charge}, mult={self.multiplicity})"

    def __str__(self):
        lines = [f"Molecule: {self.n_atoms} atoms, {self.n_electrons} electrons"]
        lines.append(f"Charge: {self.charge}, multiplicity: {self.multiplicity}")
        lines.append("\nAtom coordinates (Bohr):")
        lines.append("Atom          X              Y              Z")
        for atom in self.atoms:
            lines.append(str(atom))
        return "\n".join(lines)

    def copy(self):
        atoms_copy = [atom.copy() for atom in self.atoms]
        return Molecule(atoms_copy, self.charge, self.multiplicity)


__all__ = ['Molecule']
