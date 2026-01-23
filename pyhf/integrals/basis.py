import numpy as np


class PrimitiveGaussian:
    def __init__(self, alpha, center):
        self.alpha = float(alpha)
        self.center = np.asarray(center, dtype=float)
        self.norm = (2.0 * alpha / np.pi) ** 0.75

    def __call__(self, r):
        r = np.asarray(r)
        r_rel = r - self.center
        r2 = np.sum(r_rel ** 2, axis=-1)
        return self.norm * np.exp(-self.alpha * r2)


class ContractedGaussian:
    def __init__(self, exponents, coefficients, center):
        self.exponents = np.asarray(exponents, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.center = np.asarray(center, dtype=float)
        self.primitives = [
            PrimitiveGaussian(alpha, center)
            for alpha in self.exponents
        ]
        self.n_primitives = len(self.exponents)

    def __call__(self, r):
        result = 0.0
        for coef, prim in zip(self.coefficients, self.primitives):
            result += coef * prim(r)
        return result


STO3G_DATA = {
    'H': {
        '1s': {
            'exponents': [3.42525091, 0.62391373, 0.16885540],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        }
    },
    'He': {
        '1s': {
            'exponents': [6.36242139, 1.15892300, 0.31364979],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        }
    },
    'Li': {
        '1s': {
            'exponents': [16.1195750, 2.9362007, 0.7946505],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        },
        '2s': {
            'exponents': [0.6362897, 0.1478601, 0.0480887],
            'coefficients': [-0.09996723, 0.39951283, 0.70011547]
        }
    },
    'Be': {
        '1s': {
            'exponents': [30.1678710, 5.4951153, 1.4871927],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        },
        '2s': {
            'exponents': [1.3148331, 0.3055389, 0.0993707],
            'coefficients': [-0.09996723, 0.39951283, 0.70011547]
        }
    },
}


def build_basis(molecule, basis_name='sto-3g'):
    if basis_name.lower() != 'sto-3g':
        raise ValueError(f"Basis set '{basis_name}' not implemented")

    basis_functions = []
    basis_to_atom = []

    for atom_idx, atom in enumerate(molecule.atoms):
        symbol = atom.symbol
        center = atom.position

        if symbol not in STO3G_DATA:
            raise ValueError(f"STO-3G basis not available for {symbol}")

        atom_data = STO3G_DATA[symbol]
        for orbital_name in sorted(atom_data.keys()):
            orbital = atom_data[orbital_name]
            bf = ContractedGaussian(
                exponents=orbital['exponents'],
                coefficients=orbital['coefficients'],
                center=center
            )
            basis_functions.append(bf)
            basis_to_atom.append(atom_idx)

    return basis_functions, basis_to_atom


__all__ = [
    'PrimitiveGaussian',
    'ContractedGaussian',
    'STO3G_DATA',
    'build_basis'
]
