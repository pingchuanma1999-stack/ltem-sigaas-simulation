"""Tip scattering and tapping demodulation.

The THz near field E_loc(h, t) at the tip position (no tip present) comes from the
pulse-dynamics stage. The tip is modelled with the finite-dipole model (FDM,
Cvitkovic, Ocelic, Hillenbrand, Opt. Express 15, 8550 (2007); form as in Hauer,
Engelhardt, Taubner, Opt. Express 20, 13173 (2012)):

    alpha_eff(h) ∝ 1 + f0(h) beta / (2 (1 - f1(h) beta)),
    f_i(h) = (g - (R + 2h + W_i)/L) ln(4L/(4h + 2W_i + R)) / ln(4L/R),
    W0 = 1.31 R, W1 = R/2, g = 0.7 exp(0.06 i),   beta = (eps - 1)/(eps + 1).

The scattered field is E_sc(h, t) = alpha_eff(h, t) E_loc(h, t). With the tip tapping,
h(τ) = h_min + A (1 + cos Ωτ), Ω = 2π·68 kHz, the lock-in output at harmonic n is

    S_n(t) = (1/π) ∫_0^{2π} E_sc(h(θ), t) cos(nθ) dθ.

Ω itself drops out (the THz and tapping time scales are separated by 10^7); what
matters is the tapping amplitude A, the minimum height and the harmonic order. The
sample's THz permittivity under the tip includes the photo-plasma (Drude), so beta is
time dependent.
"""
from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-12
QE = 1.602176634e-19
ME = 9.1093837e-31


def beta_thz(n_cm3, f_thz=1.0, eps_inf=12.9, m_eff=0.067, tau_s=2e-13):
    """Quasi-static surface response factor beta = (eps-1)/(eps+1) of photoexcited GaAs at f_thz."""
    w = 2 * np.pi * f_thz * 1e12
    wp2 = n_cm3 * 1e6 * QE ** 2 / (EPS0 * eps_inf * m_eff * ME)
    eps = eps_inf * (1 - wp2 / (w ** 2 + 1j * w / tau_s))
    return (eps - 1) / (eps + 1)


def fdm_alpha(h, beta, R=40e-9, L=300e-9):
    """Finite-dipole effective polarisability (relative, complex) vs height h (m)."""
    g = 0.7 * np.exp(0.06j)
    W0, W1 = 1.31 * R, 0.5 * R
    f0 = (g - (R + 2 * h + W0) / L) * np.log(4 * L / (4 * h + 2 * W0 + R)) / np.log(4 * L / R)
    f1 = (g - (R + 2 * h + W1) / L) * np.log(4 * L / (4 * h + 2 * W1 + R)) / np.log(4 * L / R)
    return 1 + f0 * beta / (2 * (1 - f1 * beta))


def demodulate(t, heights_nm, Ez_tip, n_surface_t, A_nm=60.0, h_min_nm=1.0, harmonics=(1, 2, 3, 4),
               R_nm=40.0, L_nm=300.0, f_thz=1.0, n_theta=256):
    """Return dict harmonic -> S_n(t) (same units as Ez_tip), plus the bare E_loc at h = h_min + A."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    h_tap = h_min_nm + A_nm * (1 + np.cos(theta))          # nm
    out = {n: np.zeros(len(t)) for n in harmonics}
    out["E_loc_mid"] = np.zeros(len(t))
    hs = np.asarray(heights_nm, dtype=float)
    for i in range(len(t)):
        Eh = np.interp(h_tap, hs, Ez_tip[i])               # E_loc at the tapping heights
        beta = beta_thz(max(n_surface_t[i], 0.0), f_thz=f_thz)
        alpha = fdm_alpha(h_tap * 1e-9, beta, R=R_nm * 1e-9, L=L_nm * 1e-9)
        Esc = np.real(alpha) * Eh
        for n in harmonics:
            out[n][i] = np.mean(Esc * np.cos(n * theta)) * 2.0
        out["E_loc_mid"][i] = np.interp(h_min_nm + A_nm, hs, Ez_tip[i])
    return out


def lowpass(t, y, f_cut_thz=2.2):
    """Zero-phase Gaussian low-pass (detector bandwidth) on a uniformly sampled trace."""
    dt = t[1] - t[0]
    sigma_t = 1.0 / (2 * np.pi * f_cut_thz * 1e12) / np.sqrt(2)     # Gaussian with -3 dB near f_cut
    nk = int(4 * sigma_t / dt) + 1
    k = np.arange(-nk, nk + 1) * dt
    w = np.exp(-k ** 2 / (2 * sigma_t ** 2)); w /= w.sum()
    return np.convolve(y, w, mode="same")


def amplitude_metrics(t, y, baseline=False):
    """Peak |y| and RMS over the trace. The simulated traces start from zero carriers, so no baseline is
    removed by default (the earlier default subtracted the mean of the first 5 % of the window, which for a
    pulse centred at 0.15 ps is the rising edge itself; corrected 2026-10-27)."""
    y = np.asarray(y, dtype=float)
    if baseline:
        y = y - y[: max(3, len(y) // 20)].mean()
    return float(np.max(np.abs(y))), float(np.sqrt(np.mean(y ** 2)))


# ---------------------------------------------------------------------------
# Pizzuto-style FDM (the MATLAB script shared on 2026-10-26): original Cvitkovic 2007
# form of f0/f1, L = 600 nm, g = 0.8 exp(0.06 i), tapping 0..400 nm (A = 200 nm, z = 0),
# single frequency 0.84 THz, harmonic coefficients of alpha_eff(h(t)) by cosine-series
# decomposition, |a_n| = sqrt(Re^2 + Im^2).
#   alpha_thz  = (1 + beta) * t1 * t2 * (2 + t3/t4)                     (reflection s-SNOM)
#   alpha_ltem = t_pump * (1 + beta) * (1 - beta) * t1 * t2 * (2 + t3/t4)  (her LTEM FDM)
# with t3 = beta (g - (a + h)/l) ln(4l/(4h + 3a)),  t4 = ln(4l/a) - beta (g - (3a + 4h)/(4l)) ln(2l/(2h + a)).
# ---------------------------------------------------------------------------
from scipy.signal import hilbert as _hilbert


def fdm_alpha_pizzuto(h, beta, a=40e-9, l=600e-9, g=0.8 * np.exp(0.06j)):
    """alpha_eff(h) up to the constant t1*t2 (returned without it), original FDM form used in the MATLAB script."""
    t3 = beta * (g - (a + h) / l) * np.log(4 * l / (4 * h + 3 * a))
    t4 = np.log(4 * l / a) - beta * (g - (3 * a + 4 * h) / (4 * l)) * np.log(2 * l / (2 * h + a))
    return 2 + t3 / t4


def demodulate_pizzuto(t, heights_nm, Ez_tip, n_surface_t, A_nm=200.0, h_min_nm=0.0, harmonics=(1, 2, 3, 4),
                       a_nm=40.0, l_nm=600.0, g=0.8 * np.exp(0.06j), f_thz=0.84, collection_factor=True,
                       ltem_transmission_factor=False, eps_inf=12.9, tau_s=1e-13, n_theta=256):
    """Harmonic traces S_n(t) with the Pizzuto FDM conventions.

    The local near field E_loc(h, t) (real, from the carrier simulation; it already contains the
    static dielectric transmission 2/(eps_r+1)) is turned into its analytic signal in t so that the
    complex polarisability acts as a phase shift at the working frequency:
        E_sc(h, t) = Re[ alpha(h, t) * (E_loc + i H[E_loc]) ].
    collection_factor multiplies by (1 + beta) (tip field reaching the detector directly and via the
    sample); ltem_transmission_factor multiplies by (1 - beta) as in the LTEM FDM of the script (not
    used by default here, because the carrier simulation already provides the emitted field below the tip).
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    h_tap = (h_min_nm + A_nm * (1 + np.cos(theta))) * 1e-9
    hs = np.asarray(heights_nm, dtype=float) * 1e-9
    # analytic signal of E_loc along t for every recorded height
    Ea = _hilbert(Ez_tip, axis=0)                       # (nt, nh) complex
    out = {n: np.zeros(len(t)) for n in harmonics}
    out["E_loc_mid"] = np.zeros(len(t))
    for i in range(len(t)):
        Eh = np.interp(h_tap, hs, Ea[i].real) + 1j * np.interp(h_tap, hs, Ea[i].imag)
        beta = beta_thz(max(n_surface_t[i], 0.0), f_thz=f_thz, eps_inf=eps_inf, tau_s=tau_s)
        alpha = fdm_alpha_pizzuto(h_tap, beta, a=a_nm * 1e-9, l=l_nm * 1e-9, g=g)
        if collection_factor:
            alpha = alpha * (1 + beta)
        if ltem_transmission_factor:
            alpha = alpha * (1 - beta)
        Esc = np.real(alpha * Eh)
        for n in harmonics:
            out[n][i] = np.mean(Esc * np.cos(n * theta)) * 2.0
        out["E_loc_mid"][i] = np.interp(h_min_nm * 1e-9 + A_nm * 1e-9, hs, Ez_tip[i])
    return out


# ---------------------------------------------------------------------------
# Detection conventions (2026-10-27). The near field at the tip is quasi-static (∝ the instantaneous
# dipole of the charge distribution), the tip's induced dipole/charge follows it instantaneously, and
# the field radiated to the detector is ∝ the SECOND time derivative of that dipole — the same order
# as the far-field reference dJ/dt = d²P/dt². The earlier dS/dt convention was one derivative short.
# The detector is a steep low-pass (Butterworth-like |H|² = 1/(1+(f/fc)^(2m))), because a soft Gaussian
# lets the 20 THz plasma oscillations of the model leak into a second-derivative quantity.
# ---------------------------------------------------------------------------
def lowpass_steep(t, y, f_cut_thz=2.2, order=8):
    """Causal Butterworth low-pass (order 8 by default) as a stand-in for the detector response; no baseline
    removal (simulated traces start at zero). Causal, so the pulse keeps its arrival time; the RMS is the same as
    for the zero-phase version (same |H|)."""
    from scipy.signal import butter, sosfilt
    y0 = np.asarray(y, dtype=float)
    fs = 1.0 / (t[1] - t[0])
    sos = butter(order, f_cut_thz * 1e12 / (fs / 2), btype="low", output="sos")
    return sosfilt(sos, y0)


def detected_waveform(t, S, order=2, f_cut_thz=2.2, filter_order=8):
    """Detected THz waveform from a demodulated quasi-static signal S(t): order-th time derivative, steep low-pass."""
    y = np.asarray(S, dtype=float)
    for _ in range(order):
        y = np.gradient(y, t)
    return lowpass_steep(t, y, f_cut_thz=f_cut_thz, order=filter_order)


def demodulate_positions(heights_nm, Q, A_nm=180.0, h_min_nm=10.0, harmonics=(1, 2, 3, 4), n_theta=256):
    """Lock-in harmonics from per-tip-position data Q[k, i] (height k, time i), as in the per-position
    simulation approach: interpolate the sampled heights onto the tapping trajectory h(θ) = h_min + A(1+cos θ)
    and project on cos(nθ). Returns dict n -> S_n(t)."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    h_tap = h_min_nm + A_nm * (1 + np.cos(theta))
    hs = np.asarray(heights_nm, dtype=float)
    order = np.argsort(hs)
    hs, Q = hs[order], np.asarray(Q)[order]
    out = {}
    # interpolate in height for every time sample (linear in h; clamp outside the sampled range)
    Qt = np.empty((n_theta, Q.shape[1]))
    for i in range(Q.shape[1]):
        Qt[:, i] = np.interp(h_tap, hs, Q[:, i])
    for n in harmonics:
        out[n] = (2.0 / n_theta) * np.sum(Qt * np.cos(n * theta)[:, None], axis=0)
    out[0] = Qt.mean(axis=0)
    return out


def band_metric(t, S, deriv=2, band_thz=(0.3, 2.0)):
    """Band-limited amplitude of the detected waveform: sqrt of the spectral power of d^deriv S/dt^deriv inside
    band_thz. This is what the RMS of a detector-band-limited waveform measures, and it is insensitive to the
    exact detector roll-off (unlike the RMS of a filtered time trace, which for a second-derivative quantity
    is dominated by leakage of the 5-15 THz surge). The trace is extended with its final value and the plateau
    removed before the FFT, so the periodic extension has no step."""
    S = np.asarray(S, dtype=float)
    S = np.concatenate([S, np.full(len(S), S[-1])]) - S[-1]
    f = np.fft.rfftfreq(len(S), t[1] - t[0])
    Y = np.fft.rfft(S) * (2j * np.pi * f) ** deriv
    m = (f >= band_thz[0] * 1e12) & (f <= band_thz[1] * 1e12)
    return float(np.sqrt(np.sum(np.abs(Y[m]) ** 2)))
