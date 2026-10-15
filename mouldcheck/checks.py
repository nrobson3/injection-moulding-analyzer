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

import numpy as np

from .mesh import components
from .raycast import RayCaster


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


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
    if np.any(perp < 1e-6):
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
    those. A face square to the pull has no sideways normal and borrows that of
    the nearest sloped face in the region; a region with no sloped face at all
    tries eight directions square to the pull."""
    n = mesh.normals[faces]
    side = n - np.outer(n @ d, d)
    length = np.linalg.norm(side, axis=1)
    sloped = length > 1e-6
    if sloped.any():
        side = side / np.where(sloped, length, 1.0)[:, None]
        if not sloped.all():
            cen = mesh.centroids[faces]
            src = np.flatnonzero(sloped)
            for i in np.flatnonzero(~sloped):
                side[i] = side[src[np.argmin(np.linalg.norm(cen[src] - cen[i], axis=1))]]
        candidates = [side]
    else:
        a = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        candidates = [np.tile(math.cos(k * math.pi / 4) * a + math.sin(k * math.pi / 4) * b, (len(faces), 1))
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
    """Local wall thickness at each face from two measures:
      ray   distance along the inward normal to the far side of the wall. Right
            almost everywhere, but from the edge face of a thin plate (or the end
            of a slot) it runs along the plate and reads its width.
      ball  diameter of the largest ball inside the material touching the face
            (shrinking-ball medial-axis estimate over surface samples). Right on
            those edge faces, but tangent at a point near a convex edge it is
            genuinely small, so on its own it reads thin along every edge.
    The ray is used unless it reads thick (over thick x a first-pass nominal);
    then the ball decides. A face is only called thick when both agree, which
    keeps real thick sections (boss and rib roots) and drops rays that were
    just running along a wall.
    The nominal wall is the area-weighted median. Faces over thick x nominal are
    flagged as sink / void risk (bosses, rib roots), under thin x nominal as hard
    to fill.

    The ball is an upper bound: it is limited by the surface samples it can see,
    and a missing sample can only let it grow. Samples every 0.5 mm keep the
    error well under that on the walls of a moulded part."""
    caster = caster or RayCaster(mesh.tris)
    lo, hi = mesh.bounds
    max_dist = max_dist or float(np.linalg.norm(hi - lo))
    ray = caster.first_hit(mesh.centroids - 1e-4 * mesh.normals, -mesh.normals, max_dist=max_dist,
                           skip=np.arange(len(mesh.faces)))
    ok = np.isfinite(ray)
    first = weighted_median(ray[ok], mesh.areas[ok])
    long = ~ok | (ray > thick * first)
    t = ray.copy()
    if long.any():
        t[long] = inscribed_ball(mesh, faces=np.flatnonzero(long))
    ok = np.isfinite(t)
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


def surface_samples(mesh, spacing=0.5, cap=150000):
    """Points spread over the surface about 'spacing' apart: along every edge (long
    thin triangles are mostly edge) and on a barycentric grid inside each face."""
    pts = [mesh.vertices]
    e = np.array(list(mesh.edges().keys()))
    a, b = mesh.vertices[e[:, 0]], mesh.vertices[e[:, 1]]
    n = np.ceil(np.linalg.norm(b - a, axis=1) / spacing).astype(int)
    for k in np.unique(n[n > 1]):
        sel = n == k
        t = (np.arange(1, k) / k)[None, :, None]
        pts.append((a[sel][:, None, :] + t * (b[sel] - a[sel])[:, None, :]).reshape(-1, 3))
    for tri, area in zip(mesh.tris, mesh.areas):
        k = int(min(30, math.ceil(math.sqrt(area) / spacing)))
        if k <= 1:
            continue
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1))
        keep = i + j <= k
        a, b = i[keep] / k, j[keep] / k
        pts.append(tri[0] + np.outer(a, tri[1] - tri[0]) + np.outer(b, tri[2] - tri[0]))
    pts = np.vstack(pts)
    if len(pts) > cap:
        pts = pts[np.random.default_rng(0).choice(len(pts), cap, replace=False)]
    return pts


def inscribed_ball(mesh, faces=None, chunk=32, spacing=0.5):
    """Diameter of the largest ball inside the material that touches each face at
    its centroid. A ball tangent at p (inward normal m) that passes through a
    surface point q has radius |q - p|^2 / (2 (q - p).m); the inscribed ball is the
    smallest such radius over the surface samples on the inward side."""
    faces = np.arange(len(mesh.faces)) if faces is None else np.asarray(faces)
    samples = surface_samples(mesh, spacing)
    out = np.full(len(faces), np.inf)
    for s0 in range(0, len(faces), chunk):
        f = faces[s0:s0 + chunk]
        p = mesh.centroids[f]
        m = -mesh.normals[f]
        dvec = samples[None, :, :] - p[:, None, :]
        dm = np.einsum('kni,ki->kn', dvec, m)
        d2 = np.einsum('kni,kni->kn', dvec, dvec)
        r = np.where(dm > 1e-6 * np.sqrt(d2) + 1e-9, d2 / (2 * np.where(dm > 0, dm, 1)), np.inf)
        out[s0:s0 + chunk] = 2 * r.min(1)
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
    loops = chain_edges(edges)
    out = []
    for loop in loops:
        p = mesh.vertices[loop]
        h = p @ d
        seg = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
        out.append({'length': float(seg.sum()), 'height': (float(h.min()), float(h.max())),
                    'planar': bool(h.max() - h.min() < 1e-3), 'points': p})
    out.sort(key=lambda x: -x['length'])
    ambiguous = float(mesh.areas[side == 0].sum())
    return out, {'loops': len(out), 'longest': out[0]['length'] if out else 0.0,
                 'zero_draft_area': ambiguous}


def chain_edges(edges):
    """Join undirected vertex-index edges into vertex loops (open chains kept too)."""
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen = set()
    loops = []
    for u, v in edges:
        if (u, v) in seen or (v, u) in seen:
            continue
        seen.add((u, v))
        loop = [u, v]
        while True:
            nxt = [w for w in adj[loop[-1]] if (loop[-1], w) not in seen and (w, loop[-1]) not in seen]
            if not nxt:
                break
            seen.add((loop[-1], nxt[0]))
            if nxt[0] == loop[0]:
                break
            loop.append(nxt[0])
        loops.append(loop)
    return loops


# ------------------------------------------------------------------ snap fits
def cantilever_snap(L, t, b, y, E, yield_strain, lead_deg=30.0, mu=0.5, repeated=True):
    """Straight cantilever snap arm of constant section with a rigid root.

    L length to the catch, t thickness, b width (mm); y deflection to clear the
    catch (mm); E flexural modulus (MPa); yield_strain as a fraction.
    Allowable strain follows the common resin-supplier rule: 70 % of yield strain
    for one assembly, 60 % of that if it is assembled repeatedly.
    """
    strain = 1.5 * t * y / L ** 2
    allow = 0.7 * yield_strain * (0.6 if repeated else 1.0)
    P = b * t ** 2 / 6.0 * E * strain / L                     # tip force to deflect, N
    ta = math.tan(math.radians(lead_deg))
    if mu * ta >= 1:
        raise ValueError('lead-in angle self-locks at this friction: assembly force is unbounded')
    W = P * (mu + ta) / (1 - mu * ta)                          # axial push-on force, N
    return {'strain': strain, 'allowable': allow, 'safety': allow / strain, 'deflection_force': P,
            'mating_force': W, 'max_deflection': allow * L ** 2 / (1.5 * t)}
