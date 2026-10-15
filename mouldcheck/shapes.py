"""Test parts with known answers, made by revolving a closed (r, z) profile.

Every check in this package is tested against parts like these, where the draft,
wall, undercut depth and parting line can be worked out by hand.
"""
import math

import numpy as np


def revolve(profile, segments=128):
    """Triangles of the solid swept by a closed, counter-clockwise (r, z) profile
    (r > 0) about the z axis."""
    a = np.linspace(0, 2 * math.pi, segments + 1)
    tris = []
    for i in range(len(profile)):
        (r0, z0), (r1, z1) = profile[i], profile[(i + 1) % len(profile)]
        for k in range(segments):
            c0, s0, c1, s1 = math.cos(a[k]), math.sin(a[k]), math.cos(a[k + 1]), math.sin(a[k + 1])
            p00, p01 = (r0 * c0, r0 * s0, z0), (r0 * c1, r0 * s1, z0)
            p10, p11 = (r1 * c0, r1 * s0, z1), (r1 * c1, r1 * s1, z1)
            if r0 > 1e-9:
                tris.append((p00, p10, p11) if r1 > 1e-9 else (p00, p10, p01))
            if r1 > 1e-9 and r0 > 1e-9:
                tris.append((p00, p11, p01))
            elif r0 <= 1e-9 and r1 > 1e-9:
                tris.append((p00, p10, p11))
    return np.array(tris, float)


def cup(radius=20.0, height=30.0, wall=2.0, floor=2.0, draft_deg=2.0, groove=None, segments=128):
    """Open-topped cup, pull +z: outer and inner walls drafted by draft_deg so it
    leaves a two-plate tool cleanly. groove=(z, height, depth) cuts an internal
    groove into the inner wall: an undercut of exactly 'depth' for the core."""
    k = math.tan(math.radians(draft_deg))
    ro = lambda z: radius + z * k                  # outer wall widens upward
    ri = lambda z: radius - wall + z * k           # inner wall too: the core pulls up
    prof = [(0.0, 0.0), (ro(0), 0.0), (ro(height), height), (ri(height), height)]
    if groove:
        gz, gh, gd = groove
        prof += [(ri(gz + gh), gz + gh), (ri(gz + gh) + gd, gz + gh), (ri(gz) + gd, gz), (ri(gz), gz)]
    prof += [(ri(floor), floor), (0.0, floor)]
    return revolve(prof, segments)
