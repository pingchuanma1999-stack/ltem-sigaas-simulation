"""Finite-volume face bookkeeping for the tensor grid.

Two face sets are needed:
  * the full domain (GaAs + air) for Poisson,
  * the semiconductor rows only for the carrier continuity equations.
Flat cell index = k * Nx + i (k = row from the bottom), so semiconductor cells
share their indices between the two sets.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

from .grid import Grid


@dataclass
class Faces:
    """Interior faces between cells c and nb (nb = c + 1 in x, or c + Nx in z)."""
    c: np.ndarray
    nb: np.ndarray
    A: np.ndarray        # face length (um), unit depth in y
    d1: np.ndarray       # half width of cell c normal to the face (um)
    d2: np.ndarray       # half width of cell nb
    is_z: np.ndarray     # True for z-faces

    @property
    def d(self):
        return self.d1 + self.d2


@dataclass
class BFaces:
    """Boundary faces of a given kind attached to one cell each."""
    cell: np.ndarray
    A: np.ndarray
    dhalf: np.ndarray
    x: np.ndarray        # x position of the face centre (um) — used to assign electrode values


def _interior_faces(g: Grid, nrows: int, skip_z_face_at_row=None, skip_x_mask=None) -> Faces:
    Nx = g.Nx
    dx, dz = g.dx, g.dz[:nrows]
    # x faces
    kk, ii = np.meshgrid(np.arange(nrows), np.arange(Nx - 1), indexing="ij")
    cx = (kk * Nx + ii).ravel()
    nbx = cx + 1
    Ax = np.repeat(dz, Nx - 1)
    d1x = np.tile(dx[:-1] / 2, nrows)
    d2x = np.tile(dx[1:] / 2, nrows)
    # z faces
    kk, ii = np.meshgrid(np.arange(nrows - 1), np.arange(Nx), indexing="ij")
    cz = (kk * Nx + ii).ravel()
    nbz = cz + Nx
    Az = np.tile(dx, nrows - 1)
    d1z = np.repeat(dz[:-1] / 2, Nx)
    d2z = np.repeat(dz[1:] / 2, Nx)
    keep = np.ones(cz.size, dtype=bool)
    if skip_z_face_at_row is not None and skip_x_mask is not None:
        rows = kk.ravel()
        keep &= ~((rows == skip_z_face_at_row) & np.tile(skip_x_mask, nrows - 1))
    c = np.concatenate([cx, cz[keep]])
    nb = np.concatenate([nbx, nbz[keep]])
    A = np.concatenate([Ax, Az[keep]])
    d1 = np.concatenate([d1x, d1z[keep]])
    d2 = np.concatenate([d2x, d2z[keep]])
    is_z = np.concatenate([np.zeros(cx.size, bool), np.ones(int(keep.sum()), bool)])
    return Faces(c, nb, A, d1, d2, is_z)


class Topology:
    """Faces for Poisson (full) and continuity (semiconductor), plus boundary faces."""

    def __init__(self, g: Grid, gap_um: float):
        self.g = g
        Nx, ks = g.Nx, g.ks
        xc = g.xc
        a = gap_um / 2.0
        self.electrode_mask = np.abs(xc) >= a          # per column: electrode on the surface
        self.free_surface_mask = ~self.electrode_mask
        # Poisson over all rows; the GaAs/air interface face at row ks-1 is removed under electrodes
        self.faces_all = _interior_faces(g, g.Nz, skip_z_face_at_row=ks - 1, skip_x_mask=self.electrode_mask)
        # continuity: semiconductor rows only
        self.faces_semi = _interior_faces(g, ks)
        # electrode boundary faces: top of semiconductor cells (row ks-1) and bottom of air cells (row ks)
        cols = np.where(self.electrode_mask)[0]
        self.contact_semi = BFaces(cell=(ks - 1) * Nx + cols, A=g.dx[cols],
                                   dhalf=np.full(cols.size, g.dz[ks - 1] / 2), x=xc[cols])
        self.contact_air = BFaces(cell=ks * Nx + cols, A=g.dx[cols],
                                  dhalf=np.full(cols.size, g.dz[ks] / 2), x=xc[cols])
        fcols = np.where(self.free_surface_mask)[0]
        self.free_surface = BFaces(cell=(ks - 1) * Nx + fcols, A=g.dx[fcols],
                                   dhalf=np.full(fcols.size, g.dz[ks - 1] / 2), x=xc[fcols])
        self.N_all = Nx * g.Nz
        self.N_semi = Nx * ks
        self.vol_all = g.volumes()
        self.vol_semi = g.semi_volumes()
        # permittivity per cell
        self.eps = np.ones(self.N_all)

    def set_eps(self, eps_semi: float, eps_air: float = 1.0):
        self.eps[:] = eps_air
        self.eps[: self.N_semi] = eps_semi

    def electrode_values(self, V_left: float, V_right: float) -> np.ndarray:
        """Potential on each contact face column (scaled units supplied by caller)."""
        return np.where(self.contact_semi.x < 0, V_left, V_right)


def assemble(N: int, rows, cols, vals) -> sp.csr_matrix:
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()


def bernoulli(x: np.ndarray) -> np.ndarray:
    """B(x) = x / (exp(x) - 1), numerically safe."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-5
    big = x > 40.0
    neg = x < -40.0
    mid = ~(small | big | neg)
    out[small] = 1.0 - x[small] / 2.0 + x[small] ** 2 / 12.0
    out[big] = x[big] * np.exp(-x[big])
    out[neg] = -x[neg]
    xm = x[mid]
    out[mid] = xm / np.expm1(xm)
    return out
