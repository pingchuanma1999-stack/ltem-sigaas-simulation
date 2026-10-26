"""Print the key numbers of a saved solution.  Usage: python scripts/inspect.py results/baseline/solution.npz"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim import Simulation, Solution
from ltemsim.tip import tip_scan, surface_profiles


def main(path):
    sol = Solution.load(path)
    sim = Simulation(sol.params, verbose=False)
    p = sol.params
    g = sim.grid
    print(f"bias {p.V_bias} V, anode at {p.anode_side}; V(-x)={sol.V_left:+.1f} V, V(+x)={sol.V_right:+.1f} V")
    print(f"P_avg {p.P_avg_mW * sol.gen_factor:.2f} mW, w {p.spot_w_um} um; current {sol.I_left_mA:.3f} mA "
          f"(unity gain {p.unity_gain_current_mA:.2f} mA -> gain {sol.I_left_mA / p.unity_gain_current_mA:.2f})")
    for st in sol.log:
        print("  stage", st["stage"], "converged", st.get("converged"), "t_sim %.2e s" % st.get("t_sim", 0))
    for depth in (0.2, 0.5, 1.0, 2.0):
        pr = surface_profiles(sim, sol, depth)
        print(f"\n--- depth {pr['depth']:.2f} um ---")
        print("     x    E_x    E_z      n         p        EL2+      f    w_int")
        for x in (-19, -15, -10, -7.5, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7.5, 10, 15, 19):
            i = np.argmin(np.abs(pr["x"] - x))
            print(f"{x:6.1f} {pr['Ex'][i]:6.2f} {pr['Ez'][i]:6.2f}  {pr['n'][i]:9.2e} {pr['p'][i]:9.2e} {pr['NTp'][i]:9.2e} "
                  f"{pr['f'][i]:5.3f} {pr['w_int'][i]/pr['w_int'].max():6.3f}")
    ts = tip_scan(sim, sol)
    print("\n--- tip scan (X = spot - tip, um) ---")
    print("     X      S/max   Sx/max   Sz/max")
    for X in (-10, -7.5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7.5, 10):
        i = np.argmin(np.abs(ts.X_exp - X))
        m = np.abs(ts.S).max()
        print(f"{X:6.1f} {ts.S[i]/m:8.3f} {ts.Sx[i]/m:8.3f} {ts.Sz[i]/m:8.3f}")
    print(f"peak |S| at X = {ts.X_exp[np.argmax(np.abs(ts.S))]:+.1f} um; far-field dipoles Dx={ts.Dx:.3g} Dz={ts.Dz:.3g}")


if __name__ == "__main__":
    main(sys.argv[1])
