"""Mouldability checks on a closed mesh for one straight pull direction.

Conventions: 'pull' is the direction the moving half (B side, core) opens, and the
fixed half (A side, cavity) releases the opposite way. Every non-vertical face is
formed by the half it faces:
    n . pull > 0  ->  released along +pull
    n . pull < 0  ->  released along -pull
Draft is the face's angle to the pull direction, asin(n . pull): 0 is a wall
exactly along the pull, 90 a face square to it.

Units are the mesh's own; every length default below (depths, sample spacing)
assumes millimetres, which is what CAD STL exports normally use.
"""
import math
import numbers

import numpy as np

from .mesh import components
from .raycast import RayCaster


# a face within 0.5 deg of square to the pull has no usable sideways direction:
# below this, mesh rounding (a 1e-6 mm weld grid tilts flat faces by ~1e-6 rad)
# would hand it an arbitrary one
FLAT_SIDE = math.sin(math.radians(0.5))


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError('a direction must be a non-zero vector, got %r' % (v.tolist(),))
    return v / n


# ------------------------------------------------------------------ draft
def draft(mesh, pull, min_draft=0.5, flat=89.5, facet_tol=0.01):
    """Per-face draft (degrees) and class:
    'plus' / 'minus'  released along +pull / -pull with at least min_draft
    'flat'            square to the pull (floors, shut-offs, parting faces)
    'low'             under min_draft: drags, scuffs or needs more draft
    Returns (angles, classes, summary) with areas per class and the low-draft
    regions grouped into connected patches, largest first.

    facet_tol (degrees) allows for faceting: the flat facets of a tapered
    cylinder sit slightly under the true taper (by a factor cos(pi/N) for N
    facets), so a boss drafted exactly at the minimum would otherwise fail."""
    d = unit(pull)
    ang = np.degrees(np.arcsin(np.clip(mesh.normals @ d, -1, 1)))
    cls = np.where(np.abs(ang) >= flat, 'flat',
                   np.where(ang >= min_draft - facet_tol, 'plus',
                            np.where(ang <= -min_draft + facet_tol, 'minus', 'low')))
    summary = {'min_draft': min_draft, 'area': {}, 'low_regions': []}
    for c in ('plus', 'minus', 'flat', 'low'):
        summary['area'][c] = float(mesh.areas[cls == c].sum())
    walls = (cls == 'plus') | (cls == 'minus')
    summary['smallest_wall_draft'] = float(np.abs(ang[walls]).min()) if walls.any() else None
    for g in sorted(components(cls == 'low', mesh.neighbours()), key=lambda g: -mesh.areas[g].sum()):
        summary['low_regions'].append({
            'faces': int(len(g)), 'area': float(mesh.areas[g].sum()),
            'draft_range': (float(ang[g].min()), float(ang[g].max())),
            'centre': tuple(float(x) for x in np.average(mesh.centroids[g], axis=0, weights=mesh.areas[g])),
        })
    return ang, cls, summary


# ------------------------------------------------------------------ undercuts
def undercuts(mesh, pull, vertical_tol=0.25, caster=None, min_depth=0.02, max_depth=20.0,
              prefer_backdraft=0.01):
    """Faces whose steel cannot pull straight: from the face, a ray along the
    direction it releases (by the sign of n . pull) runs into the part itself.
    Faces within vertical_tol degrees of the pull are left to the draft check.

    Each blocked region also gets a depth, the cheaper of two ways to free it:
      straight   the sideways travel the steel forming it needs before a straight
                 path out is clear (what a side action or lifter would have to move)
      backdraft  form it from the other half instead: the overlap is then the
                 face's back-draft, height along the pull x tan(draft), provided
                 that half's path is clear of the rest of the part
    A back-draft needs no moving steel, so it is chosen unless the side action is
    shallower by more than prefer_backdraft (mm).
    Regions shallower than min_depth (mm) are reported as negligible: tolerance-
    level overlaps, such as a drafted face running a few microns under a feature.
    A region with neither fix inside max_depth gets depth None.

    Detection uses one ray per face, from its centroid. A large face that is only
    partly overhung can be missed, so export the STL fine enough that faces are
    small next to the features that might trap them.

    Returns (mask, summary). mask is per face: 0 releases straight, 1 undercut,
    2 negligible undercut. The summary groups the blocked faces into regions."""
    d = unit(pull)
    caster = caster or RayCaster(mesh.tris)
    s = mesh.normals @ d
    lim = math.sin(math.radians(vertical_tol))
    test = np.abs(s) > lim
    idx = np.flatnonzero(test)
    dirs = np.where(s[idx, None] > 0, d, -d)
    starts = mesh.centroids[idx] + 1e-4 * mesh.normals[idx]
    hit = caster.first_hit(starts, dirs, skip=idx)
    blocked = np.zeros(len(mesh.faces), bool)
    blocked[idx[np.isfinite(hit)]] = True
    mask = blocked.astype(np.int8)
    groups = sorted(components(blocked, mesh.neighbours()), key=lambda g: -mesh.areas[g].sum())
    regions = []
    for g in groups:
        straight = side_travel(mesh, caster, g, d, max_depth)
        back = backdraft_travel(mesh, caster, g, d)
        if back is not None and (straight is None or back <= straight + prefer_backdraft):
            depth, fix = back, 'backdraft'
        elif straight is not None:
            depth, fix = straight, 'side action'
        else:
            depth, fix = None, None
        regions.append({
            'faces': int(len(g)), 'area': float(mesh.areas[g].sum()),
            'released_by': '+pull' if s[g].mean() > 0 else '-pull',
            'depth': depth, 'straight_travel': straight, 'backdraft_travel': back,
            'best_fix': fix,
            'negligible': depth is not None and depth < min_depth,
            'centre': tuple(float(x) for x in np.average(mesh.centroids[g], axis=0, weights=mesh.areas[g])),
        })
        if regions[-1]['negligible']:
            mask[g] = 2
    real = [r for r in regions if not r['negligible']]
    summary = {'area': float(sum(r['area'] for r in real)),
               'negligible_area': float(sum(r['area'] for r in regions if r['negligible'])),
               'min_depth': min_depth, 'max_depth': max_depth, 'regions': regions}
    return mask, summary


def _samples(mesh, faces):
    """Four points per face, nudged off the surface along the normal: the centroid,
    and each corner moved 2 % towards the centroid so it stays on its own face.
    Returns (points, face index per point)."""
    tri, cen = mesh.tris[faces], mesh.centroids[faces]
    pts = np.concatenate([cen[:, None, :], 0.98 * tri + 0.02 * cen[:, None, :]], axis=1)
    pts = pts + 1e-4 * mesh.normals[faces][:, None, :]
    return pts.reshape(-1, 3), np.repeat(np.arange(len(faces)), 4)


def backdraft_travel(mesh, caster, faces, d):
    """Overlap if the region is formed by the opposite half: the region's height
    along the pull times the steepest back-draft in it. None if that half's
    straight path (after clearing the overlap) still runs into the part."""
    n = mesh.normals[faces]
    along = np.abs(n @ d)
    perp = np.linalg.norm(n - np.outer(n @ d, d), axis=1)
    if np.any(perp < FLAT_SIDE):
        return None
    h = mesh.tris[faces].reshape(-1, 3) @ d
    travel = float((h.max() - h.min()) * np.max(along / perp))
    side = (n - np.outer(n @ d, d)) / perp[:, None]
    dirs = np.where((n @ d)[:, None] > 0, -d, d)               # the opposite half's direction
    pts, k = _samples(mesh, faces)
    if np.any(np.isfinite(caster.first_hit(pts + (travel + 1e-3) * side[k], dirs[k]))):
        return None
    return travel


def side_travel(mesh, caster, faces, d, max_depth=20.0, first_step=0.01, iters=16):
    """Sideways travel that clears a blocked region, or None if no shift up to
    max_depth does.

    A side action forms the whole region and is then carried out by one half, so
    every sample of the region is shifted sideways and must then leave along that
    one direction; both halves are tried and the cheaper kept. (A groove roof faces
    the other half from the groove's back wall, yet the same collapsing core forms
    both.) Every face is sampled at its centroid and near its corners, and each
    sample needs the smallest shift along its face's sideways normal that frees
    it: found by doubling steps from first_step, then bisection, so an obstacle
    further along the shift does not fool it. The region needs the largest of
    those.

    A face square to the pull has no sideways normal of its own. It borrows the
    one of the nearest face that has one and faces it across open space (each in
    front of the other, so never a face on the far side of the material): from the
    region itself if it can, otherwise from the rest of the part. For a groove roof
    that is the groove's back wall, for the underside of a lip the wall below it.
    Where some face finds none, or the region has no sloped face, eight fixed
    directions square to the pull are tried as well."""
    side_all = mesh.normals - np.outer(mesh.normals @ d, d)
    len_all = np.linalg.norm(side_all, axis=1)
    sloped_all = len_all >= FLAT_SIDE
    side = np.zeros((len(faces), 3))
    own = sloped_all[faces]
    side[own] = side_all[faces[own]] / len_all[faces[own], None]
    got = own.copy()
    pools = (faces[own], np.flatnonzero(sloped_all))
    for i in np.flatnonzero(~own):
        f = faces[i]
        for pool in pools:
            if len(pool) == 0:
                continue
            v = mesh.centroids[pool] - mesh.centroids[f]
            facing = (v @ mesh.normals[f] > 0) & ((v * mesh.normals[pool]).sum(1) < 0)
            if facing.any():
                j = pool[facing][np.argmin((v[facing] ** 2).sum(1))]
                side[i] = side_all[j] / len_all[j]
                got[i] = True
                break
    candidates = [side] if got.all() else []
    if not own.any() or not got.all():
        a = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        candidates += [np.tile(math.cos(k * math.pi / 4) * a + math.sin(k * math.pi / 4) * b, (len(faces), 1))
                       for k in range(8)]
    all_pts, all_k = _samples(mesh, faces)
    steps = []
    s = first_step
    while s < max_depth:
        steps.append(s)
        s *= 2
    steps.append(float(max_depth))
    best = None
    for carry in (d, -d):
        dirs_all = np.tile(carry, (len(all_pts), 1))
        stuck = np.isfinite(caster.first_hit(all_pts, dirs_all))
        if not stuck.any():
            return 0.0
        pts, k, dirs = all_pts[stuck], all_k[stuck], dirs_all[stuck]
        for cand in candidates:
            sd = cand[k]
            lo = np.zeros(len(pts))
            hi = np.full(len(pts), np.inf)
            prev = 0.0
            for s in steps:
                todo = np.flatnonzero(~np.isfinite(hi))
                if len(todo) == 0:
                    break
                free = ~np.isfinite(caster.first_hit(pts[todo] + s * sd[todo], dirs[todo]))
                hi[todo[free]] = s
                lo[todo[free]] = prev
                prev = s
            if not np.all(np.isfinite(hi)):
                continue                                          # some sample never clears this way
            for _ in range(iters):
                mid = (lo + hi) / 2
                blocked = np.isfinite(caster.first_hit(pts + mid[:, None] * sd, dirs))
                lo = np.where(blocked, mid, lo)
                hi = np.where(blocked, hi, mid)
            travel = float(hi.max())
            best = travel if best is None else min(best, travel)
    return best


# ------------------------------------------------------------------ wall thickness
def thickness(mesh, caster=None, thick=1.5, thin=0.5, max_dist=None):
    """Local wall thickness at each face.

    ray   the distance along the inward normal to the far side of the wall. Right
          almost everywhere. From an edge face (the rim of a plate, the side of a
          slot, the small faces of an edge round) it runs along the wall instead
          and reads long.
    For those faces, where the ray reads over thick x a first-pass nominal:
    wall  the ray reading of the nearest face whose ray is not long: the wall the
          edge belongs to.
    ball  the diameter of the largest ball inside the material touching the face
          at its centroid, found against the triangles themselves. It never
          exceeds the true thickness, so it can only raise the wall value: where
          it is larger, the section really is thicker (a boss or rib root, a
          thick floor). A ball alone would read thin along every edge, since a
          ball touching a face near a convex edge is pinched by that edge.
    So an edge face reads max(wall, ball), never more than its own ray.
    The nominal wall is the area-weighted median. Faces over thick x nominal are
    flagged as sink / void risk, under thin x nominal as hard to fill. Faces with
    no reading (an open mesh) are left out; if none has one, the summary's
    nominal is None."""
    caster = caster or RayCaster(mesh.tris)
    lo, hi = mesh.bounds
    max_dist = max_dist or float(np.linalg.norm(hi - lo))
    nudge = 1e-4
    ray = caster.first_hit(mesh.centroids - nudge * mesh.normals, -mesh.normals, max_dist=max_dist,
                           skip=np.arange(len(mesh.faces))) + nudge     # measured from the surface itself
    ok = np.isfinite(ray)
    t = ray.copy()
    if ok.any():
        first = weighted_median(ray[ok], mesh.areas[ok])
        long = ~ok | (ray > thick * first)
        L, S = np.flatnonzero(long), np.flatnonzero(~long)
        if len(L):
            wall = np.full(len(L), np.nan)
            if len(S):
                cs = mesh.centroids[S]
                for k in range(0, len(L), 256):
                    d2 = ((mesh.centroids[L[k:k + 256], None, :] - cs[None, :, :]) ** 2).sum(-1)
                    wall[k:k + 256] = ray[S][np.argmin(d2, axis=1)]
            ball = inscribed_ball(mesh, L, at_least=wall, start=first / 8)
            t[L] = np.minimum(ball, np.where(ok[L], ray[L], np.inf))
    ok = np.isfinite(t)
    empty = {'nominal': None, 'max': None, 'min': None, 'thick_area': 0.0, 'thin_area': 0.0,
             'thick_regions': []}
    if not ok.any():
        return t, empty
    nominal = weighted_median(t[ok], mesh.areas[ok])
    thick_mask = ok & (t > thick * nominal)
    thin_mask = ok & (t < thin * nominal)
    nb = mesh.neighbours()
    summary = {
        'nominal': nominal, 'max': float(t[ok].max()), 'min': float(t[ok].min()),
        'thick_area': float(mesh.areas[thick_mask].sum()), 'thin_area': float(mesh.areas[thin_mask].sum()),
        'thick_regions': [{
            'faces': int(len(g)), 'area': float(mesh.areas[g].sum()), 'max': float(t[g].max()),
            'centre': tuple(float(x) for x in np.average(mesh.centroids[g], axis=0, weights=mesh.areas[g])),
        } for g in sorted(components(thick_mask, nb), key=lambda g: -mesh.areas[g].sum())],
    }
    return t, summary


def weighted_median(values, weights):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def point_triangle_distance(p, a, b, c):
    """Distance from point p to each triangle (a, b, c), vectorised over triangles
    (closest-point regions after Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide='ignore', invalid='ignore'):
        regions = [(d1 <= 0) & (d2 <= 0),
                   (d3 >= 0) & (d4 <= d3),
                   (vc <= 0) & (d1 >= 0) & (d3 <= 0),
                   (d6 >= 0) & (d5 <= d6),
                   (vb <= 0) & (d2 >= 0) & (d6 <= 0),
                   (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)]
        q = np.select(
            [r[:, None] for r in regions],
            [a, b,
             a + ab * (d1 / (d1 - d3))[:, None],
             c,
             a + ac * (d2 / (d2 - d6))[:, None],
             b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None]],
            a + ab * (vb / (va + vb + vc))[:, None] + ac * (vc / (va + vb + vc))[:, None])
    return np.linalg.norm(p - q, axis=1)


def inscribed_ball(mesh, faces=None, at_least=None, start=0.1, iters=12):
    """Diameter of the largest ball inside the material that touches each face at
    its centroid, or at_least (per face) if the ball is smaller than that; inf
    where no ball is ever stopped (an open mesh).

    The ball's centre moves inward along the normal. It is first tried at
    at_least: if that ball already reaches another triangle the answer is
    at_least, at the cost of one distance test. Otherwise it grows by doubling
    (from 'start' where at_least is not given) until it reaches another
    triangle, then bisects to about 1e-4 of its size."""
    faces = np.arange(len(mesh.faces)) if faces is None else np.asarray(faces)
    floor = np.full(len(faces), np.nan) if at_least is None else np.asarray(at_least, float)
    lo_t, hi_t = mesh.tris.min(1), mesh.tris.max(1)
    r_max = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) / 2
    a, b, c = mesh.tris[:, 0], mesh.tris[:, 1], mesh.tris[:, 2]
    out = np.full(len(faces), np.inf)
    for i, f in enumerate(faces):
        p, m = mesh.centroids[f], -mesh.normals[f]

        def near(r):
            ctr = p + r * m
            sel = np.flatnonzero(np.all(hi_t >= ctr - r, 1) & np.all(lo_t <= ctr + r, 1))
            return sel[sel != f]

        def blocked(r, sel):
            return len(sel) > 0 and bool(np.any(
                point_triangle_distance(p + r * m, a[sel], b[sel], c[sel]) < r * (1 - 1e-9)))

        if np.isfinite(floor[i]):
            r = min(floor[i] / 2, r_max)
            if blocked(r, near(r)):
                out[i] = floor[i]
                continue
            r_lo, r = r, min(2 * r, r_max)
        else:
            r_lo, r = 0.0, min(max(start, 1e-6), r_max)
        while not blocked(r, near(r)):
            if r >= r_max:
                break
            r_lo, r = r, min(2 * r, r_max)
        else:
            sel = near(r)
            for _ in range(iters):
                mid = (r_lo + r) / 2
                if blocked(mid, sel):
                    r = mid
                else:
                    r_lo = mid
            out[i] = r_lo + r
    return out


# ------------------------------------------------------------------ parting line
def parting_lines(mesh, pull, vertical_tol=0.25):
    """Candidate parting lines: edges where the release direction flips from +pull
    to -pull, chained into loops. A loop with no extent along the pull is a flat
    parting line; one with extent is stepped (the tool needs a stepped shut-off).

    Faces within vertical_tol of the pull belong to neither half. Where such a
    zero-draft land separates the two halves, the flip happens across the land
    rather than at one edge, so no loop is found there; the summary's
    zero_draft_area says how much of the part this could affect."""
    d = unit(pull)
    s = mesh.normals @ d
    lim = math.sin(math.radians(vertical_tol))
    side = np.where(s > lim, 1, np.where(s < -lim, -1, 0))
    edges = []
    for (u, v), fs in mesh.edges().items():
        if len(fs) == 2 and side[fs[0]] * side[fs[1]] == -1:
            edges.append((u, v))
    out = []
    for path, closed in chain_edges(edges):
        p = mesh.vertices[path]
        h = p @ d
        q = np.vstack([p, p[:1]]) if closed else p
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        out.append({'length': float(seg.sum()), 'height': (float(h.min()), float(h.max())),
                    'planar': bool(h.max() - h.min() < 1e-3), 'closed': closed, 'points': p})
    out.sort(key=lambda x: -x['length'])
    ambiguous = float(mesh.areas[side == 0].sum())
    return out, {'loops': len(out), 'longest': out[0]['length'] if out else 0.0,
                 'zero_draft_area': ambiguous}


def chain_edges(edges):
    """Join undirected vertex-index edges into paths: [(vertices, closed), ...].
    A path that does not close is grown from both ends, so it comes back whole."""
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen = set()

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def grow(path):
        while True:
            nxt = [w for w in adj[path[-1]] if key(path[-1], w) not in seen]
            if not nxt:
                return False
            seen.add(key(path[-1], nxt[0]))
            if nxt[0] == path[0]:
                return True
            path.append(nxt[0])

    out = []
    for u, v in edges:
        if key(u, v) in seen:
            continue
        seen.add(key(u, v))
        path = [u, v]
        closed = grow(path)
        if not closed:
            path.reverse()
            grow(path)
        out.append((path, closed))
    return out


# ------------------------------------------------------------------ snap fits
def cantilever_snap(L, t, b, y, E, yield_strain, lead_deg=30.0, mu=0.5, repeated=True):
    """Straight cantilever snap arm of constant section with a rigid root.

    L length to the catch, t thickness, b width (mm); y deflection to clear the
    catch (mm); E flexural modulus (MPa); yield_strain as a fraction.
    Allowable strain follows the common resin-supplier rule: 70 % of yield strain
    for one assembly, 60 % of that if it is assembled repeatedly.
    """
    for name, val in (('L', L), ('t', t), ('b', b), ('y', y), ('E', E), ('yield_strain', yield_strain)):
        if not (isinstance(val, numbers.Real) and math.isfinite(val) and val > 0):
            raise ValueError('%s must be a positive number, got %r' % (name, val))
    if not 0 <= lead_deg < 90:
        raise ValueError('lead_deg must be from 0 up to (not including) 90 degrees, got %r' % lead_deg)
    if not mu >= 0:
        raise ValueError('mu must be zero or positive, got %r' % mu)
    strain = 1.5 * t * y / L ** 2
    allow = 0.7 * yield_strain * (0.6 if repeated else 1.0)
    P = b * t ** 2 / 6.0 * E * strain / L                     # tip force to deflect, N
    ta = math.tan(math.radians(lead_deg))
    if mu * ta >= 1 - 1e-9:
        raise ValueError('lead-in angle self-locks at this friction: assembly force is unbounded')
    W = P * (mu + ta) / (1 - mu * ta)                          # axial push-on force, N
    return {'strain': strain, 'allowable': allow, 'safety': allow / strain, 'deflection_force': P,
            'mating_force': W, 'max_deflection': allow * L ** 2 / (1.5 * t)}
