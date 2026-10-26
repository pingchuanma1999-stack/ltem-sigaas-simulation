"""Tensor-product, cell-centred finite-volume grid for the (x, z) cross-section.

GaAs occupies z < 0 (semiconductor cells), air z > 0.  Lengths are in um.
The grid is nonuniform: fine cells around the pumped spot and near the surface,
coarse cells far away.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .params import Params


def _graded(x_start: float, x_end: float, d_start: float, d_end: float) -> np.ndarray:
    """Cell widths growing geometrically from d_start to d_end filling [x_start, x_end].

    Returns the edges (excluding x_start, including x_end).
    """
    L = x_end - x_start
    if L <= 0:
        return np.array([], dtype=float)
    if d_end <= d_start * 1.0001:
        n = max(1, int(round(L / d_start)))
        return x_start + L * np.arange(1, n + 1) / n
    # choose growth ratio r and number n such that sum d_start r^k = L
    r = 1.15
    widths = []
    d = d_start
    total = 0.0
    while total + d < L:
        widths.append(d)
        total += d
        d = min(d * r, d_end)
    # distribute the remainder proportionally
    widths = np.array(widths)
    if widths.size == 0:
        return np.array([x_end])
    widths *= L / widths.sum()
    return x_start + np.cumsum(widths)


@dataclass
class Grid:
    xe: np.ndarray      # x edges, um, shape (Nx+1,)
    ze: np.ndarray      # z edges, um, shape (Nz+1,), increasing; ze[ks] == 0
    ks: int             # index of the surface edge: semiconductor cells k < ks

    @property
    def xc(self) -> np.ndarray:
        return 0.5 * (self.xe[1:] + self.xe[:-1])

    @property
    def zc(self) -> np.ndarray:
        return 0.5 * (self.ze[1:] + self.ze[:-1])

    @property
    def dx(self) -> np.ndarray:
        return np.diff(self.xe)

    @property
    def dz(self) -> np.ndarray:
        return np.diff(self.ze)

    @property
    def Nx(self) -> int:
        return self.xe.size - 1

    @property
    def Nz(self) -> int:
        return self.ze.size - 1

    @property
    def Nzs(self) -> int:
        """Number of semiconductor cell rows."""
        return self.ks

    # --- index helpers: cell (i, k) -> flat index k*Nx + i (k row-major) -----
    def idx(self, i, k):
        return k * self.Nx + i

    def semi_mask(self) -> np.ndarray:
        """Boolean mask over all cells: True for semiconductor cells."""
        m = np.zeros((self.Nz, self.Nx), dtype=bool)
        m[: self.ks, :] = True
        return m.ravel()

    def volumes(self) -> np.ndarray:
        return np.outer(self.dz, self.dx).ravel()   # um^2 per unit length in y

    def semi_volumes(self) -> np.ndarray:
        return np.outer(self.dz[: self.ks], self.dx).ravel()

    @classmethod
    def build(cls, p: Params) -> "Grid":
        # ---- x: fine |x| <= x_fine, graded to dx_coarse outside -----------
        xf = np.arange(-p.x_fine, p.x_fine + 1e-9, p.dx_fine)
        right = _graded(p.x_fine, p.x_half, p.dx_fine, p.dx_coarse)
        xe = np.concatenate([(-right)[::-1], xf, right])
        # ---- z: semiconductor (negative) ---------------------------------
        zs_fine = -np.arange(0.0, p.z_fine + 1e-9, p.dz_fine)[::-1]           # ... -0.2, -0.1, 0
        zs_coarse = -_graded(p.z_fine, p.depth, p.dz_fine, p.dz_coarse_semi)  # -2.x ... -depth
        ze_semi = np.concatenate([zs_coarse[::-1], zs_fine])                   # from -depth up to 0
        za_fine = np.arange(0.0, p.z_fine + 1e-9, p.dz_fine)[1:]
        za_coarse = _graded(p.z_fine, p.air_height, p.dz_fine, p.dz_coarse_air)
        ze = np.concatenate([ze_semi, za_fine, za_coarse])
        ks = int(np.argmin(np.abs(ze)))
        ze[ks] = 0.0
        return cls(xe=xe, ze=ze, ks=ks)

    def describe(self) -> str:
        return (f"grid: Nx={self.Nx}, Nz={self.Nz} (semi {self.Nzs}, air {self.Nz - self.Nzs}), "
                f"x in [{self.xe[0]:.1f}, {self.xe[-1]:.1f}] um, z in [{self.ze[0]:.1f}, {self.ze[-1]:.1f}] um, "
                f"dx_min={self.dx.min():.3f}, dz_min={self.dz.min():.3f}")
