import numpy as np
from pyhf.utils.constants import (
    ATOMIC_NUMBERS, ATOMIC_SYMBOLS, ATOMIC_MASSES,
    BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR
)

class Atom:

    def __init__(self, symbol, position, units='bohr'):
        if isinstance(symbol, str):
            if symbol not in ATOMIC_NUMBERS:
                raise ValueError(f"Unknown atomic symbol: {symbol}")
            self.symbol = symbol
            self.atomic_number = ATOMIC_NUMBERS[symbol]
        elif isinstance(symbol, int):
            if symbol not in ATOMIC_SYMBOLS:
                raise ValueError(f"Unknown atomic number: {symbol}")
            self.atomic_number = symbol
            self.symbol = ATOMIC_SYMBOLS[symbol]
        else:
            raise TypeError("symbol must be str or int")

        position = np.asarray(position, dtype=float)
        if position.shape != (3,):
            raise ValueError("position must be a 3D vector")

        if units.lower() == 'angstrom':
            self.position = position * ANGSTROM_TO_BOHR
        elif units.lower() == 'bohr':
            self.position = position
        else:
            raise ValueError("units must be 'bohr' or 'angstrom'")

        self.mass = ATOMIC_MASSES[self.atomic_number]

    def __repr__(self):
        x, y, z = self.position
        return f"Atom('{self.symbol}', [{x:.6f}, {y:.6f}, {z:.6f}])"

    def __str__(self):
        x, y, z = self.position
        return f"{self.symbol:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}"

    def distance_to(self, other):
        return np.linalg.norm(self.position - other.position)

    def position_angstrom(self):
        return self.position * BOHR_TO_ANGSTROM

    def copy(self):
        return Atom(self.symbol, self.position.copy(), units='bohr')

__all__ = ['Atom']
