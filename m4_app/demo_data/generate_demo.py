"""Generate deterministic, obviously synthetic 2-trip smoke-test fixtures."""
from __future__ import annotations
from pathlib import Path
import random

OUT = Path(__file__).resolve().parent
# small illustrative street grid. Lon, lat; NOT a real measured trajectory.
NODES = [(77.6000,12.9500), (77.6010,12.9500), (77.6020,12.9500),
         (77.6020,12.9510), (77.6010,12.9510), (77.6000,12.9510)]
ARCS = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(1,4),(4,1)]


def write_trip(name: str, arc_ids: list[int], seed: int) -> None:
    rng = random.Random(seed)
    (OUT / f"{name}.nodes").write_text("".join(f"{lon:.6f} {lat:.6f}\n" for lon,lat in NODES))
    (OUT / f"{name}.arcs").write_text("".join(f"{a} {b}\n" for a,b in ARCS))
    (OUT / f"{name}.route").write_text("".join(f"{a}\n" for a in arc_ids))
    obs = []
    t = 0
    for arc_id in arc_ids:
        start, end = ARCS[arc_id]
        ax, ay = NODES[start]
        bx, by = NODES[end]
        for step in range(7):
            factor = step / 7.0
            x = ax + (bx-ax)*factor + rng.uniform(-.000025,.000025)
            y = ay + (by-ay)*factor + rng.uniform(-.000025,.000025)
            obs.append(f"{x:.7f} {y:.7f} {t}\n")
            t += 1
    ax, ay = NODES[ARCS[arc_ids[-1]][1]]
    obs.append(f"{ax:.7f} {ay:.7f} {t}\n")
    (OUT / f"{name}.track").write_text("".join(obs))


if __name__ == '__main__':
    write_trip('demo_01', [0, 1, 2], 15)
    write_trip('demo_02', [3, 4, 5], 37)
    print('Created synthetic M4 smoke-test trips (not for evaluation).')
