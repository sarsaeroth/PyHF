import numpy as np
from pyhf.utils.constants import HARTREE_TO_EV


def ionization_energy(rhf, orbital_index=None):
    if not rhf.converged:
        raise RuntimeError("RHF calculation not converged")

    n_occ = rhf.n_occ
    if orbital_index is None:
        orbital_index = n_occ - 1

    if orbital_index < 0 or orbital_index >= n_occ:
        raise ValueError(f"orbital_index must be in range [0, {n_occ-1}]")

    return -rhf.orbital_energies[orbital_index]


def electron_affinity(rhf, orbital_index=None):
    if not rhf.converged:
        raise RuntimeError("RHF calculation not converged")

    n_occ = rhf.n_occ
    n_vir = rhf.n_basis - n_occ

    if n_vir == 0:
        raise ValueError("No virtual orbitals available")

    if orbital_index is None:
        orbital_index = 0

    if orbital_index < 0 or orbital_index >= n_vir:
        raise ValueError(f"orbital_index must be in range [0, {n_vir-1}]")

    lumo_index = n_occ + orbital_index
    return -rhf.orbital_energies[lumo_index]


def homo_lumo_gap(rhf):
    if not rhf.converged:
        raise RuntimeError("RHF calculation not converged")

    n_occ = rhf.n_occ
    n_vir = rhf.n_basis - n_occ

    if n_vir == 0:
        raise ValueError("No virtual orbitals available")

    homo_energy = rhf.orbital_energies[n_occ - 1]
    lumo_energy = rhf.orbital_energies[n_occ]
    return lumo_energy - homo_energy


def print_koopmans_analysis(rhf):
    IE = ionization_energy(rhf)
    n_vir = rhf.n_basis - rhf.n_occ
    if n_vir > 0:
        EA = electron_affinity(rhf)
        gap = homo_lumo_gap(rhf)
        print(f"IE(HOMO): {IE:.6f} Ha ({IE*HARTREE_TO_EV:.4f} eV), EA(LUMO): {EA:.6f} Ha ({EA*HARTREE_TO_EV:.4f} eV), Gap: {gap:.6f} Ha")
    else:
        print(f"IE(HOMO): {IE:.6f} Ha ({IE*HARTREE_TO_EV:.4f} eV)")
    for i, eps in enumerate(rhf.orbital_energies):
        occ = "occ" if i < rhf.n_occ else "vir"
        print(f"MO {i+1} ({occ}): {eps:.6f} Ha ({eps*HARTREE_TO_EV:.4f} eV)")


__all__ = [
    'ionization_energy',
    'electron_affinity',
    'homo_lumo_gap',
    'print_koopmans_analysis'
]
