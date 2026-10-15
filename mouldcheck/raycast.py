"""Ray / mesh intersection (Moller-Trumbore), vectorised over rays and triangles.

Brute force in chunks: every ray is tested against every triangle, with a
bounding-box prefilter per chunk. That keeps the code short and exact; it is
quick for the meshes a moulded part exports at (a few thousand to ~30k
triangles) and slow beyond that.
"""
import numpy as np


class RayCaster:
    def __init__(self, tris):
        t = np.asarray(tris, float)
        self.v0 = t[:, 0]
        self.e1 = t[:, 1] - t[:, 0]
        self.e2 = t[:, 2] - t[:, 0]
        self.lo = t.min(1)
        self.hi = t.max(1)
        self.box = (t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0))

    def first_hit(self, origins, dirs, max_dist=np.inf, chunk=48, skip=None):
        """Distance to the first triangle hit along each ray (inf for none).
        'skip' is an optional face index per ray to ignore (the face it starts on)."""
        origins = np.asarray(origins, float).reshape(-1, 3)
        dirs = np.asarray(dirs, float).reshape(-1, 3)
        out = np.full(len(origins), np.inf)
        if len(origins) == 0:
            return out
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        for s in range(0, len(origins), chunk):
            o, d = origins[s:s + chunk], dirs[s:s + chunk]
            # prefilter: triangles whose box meets the boxes of these ray segments
            # an unlimited ray only needs to reach the far corner of the mesh's box
            far = np.maximum(np.abs(o - self.box[0]), np.abs(o - self.box[1]))
            reach = min(max_dist, float(np.linalg.norm(far, axis=1).max()) + 1.0)
            ends = o + d * reach
            seg_lo = np.minimum(o, ends).min(0)
            seg_hi = np.maximum(o, ends).max(0)
            cand = np.flatnonzero(np.all(self.hi >= seg_lo, 1) & np.all(self.lo <= seg_hi, 1))
            if len(cand) == 0:
                continue
            v0, e1, e2 = self.v0[cand], self.e1[cand], self.e2[cand]
            p = np.cross(d[:, None, :], e2[None, :, :])                  # (k, n, 3)
            det = np.einsum('kni,ni->kn', p, e1)
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            tv = o[:, None, :] - v0[None, :, :]
            u = np.einsum('kni,kni->kn', tv, p) * inv
            q = np.cross(tv, e1[None, :, :])
            v = np.einsum('ki,kni->kn', d, q) * inv
            dist = np.einsum('ni,kni->kn', e2, q) * inv
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (dist > 1e-6) & (dist <= max_dist)
            if skip is not None:
                hit &= cand[None, :] != np.asarray(skip[s:s + chunk])[:, None]
            dist = np.where(hit, dist, np.inf)
            out[s:s + chunk] = dist.min(1)
        return out
