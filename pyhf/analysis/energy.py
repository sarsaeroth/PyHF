import numpy as np


def decompose_energy(rhf):
    if not rhf.converged:
        raise RuntimeError("RHF calculation not converged")

    P = rhf.density_matrix
    T = rhf.T
    V = rhf.V
    F = rhf.fock_matrix
    H_core = rhf.H_core

    e_one = np.sum(P * H_core)
    e_two = 0.5 * np.sum(P * (F - H_core))
    e_elec = e_one + e_two
    e_kin = np.sum(P * T)
    e_nuc_att = np.sum(P * V)
    e_nuc_rep = rhf.molecule.nuclear_repulsion_energy()
    e_total = e_elec + e_nuc_rep

    return {
        'one_electron': e_one,
        'two_electron': e_two,
        'electronic': e_elec,
        'nuclear_repulsion': e_nuc_rep,
        'total': e_total,
        'kinetic': e_kin,
        'nuclear_attraction': e_nuc_att
    }


def print_energy_decomposition(rhf):
    c = decompose_energy(rhf)
    print(f"Kinetic: {c['kinetic']:.10f}, Nuclear attraction: {c['nuclear_attraction']:.10f}")
    print(f"One-electron: {c['one_electron']:.10f}, Two-electron: {c['two_electron']:.10f}")
    print(f"Electronic: {c['electronic']:.10f}, Nuclear repulsion: {c['nuclear_repulsion']:.10f}, Total: {c['total']:.10f}")


def virial_ratio(rhf):
    if not rhf.converged:
        raise RuntimeError("RHF calculation not converged")

    c = decompose_energy(rhf)
    T = c['kinetic']
    V = c['nuclear_attraction'] + c['two_electron'] + c['nuclear_repulsion']
    return -V / T


__all__ = ['decompose_energy', 'print_energy_decomposition', 'virial_ratio']
