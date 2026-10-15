"""Triangle mesh with the topology the checks need: welded vertices, outward
normals, areas, centroids and edge-to-face adjacency."""
import struct

import numpy as np


def read_stl(path):
    """Triangles as an (n, 3, 3) float array, ASCII or binary.

    A binary file whose size matches its triangle count is read as binary even if
    its 80-byte header starts with 'solid' (some exporters write that); a binary
    file with padding after the triangles is read too. Anything else that yields
    no triangles raises ValueError."""
    with open(path, 'rb') as f:
        data = f.read()
    n = struct.unpack('<I', data[80:84])[0] if len(data) >= 84 else None

    def binary():
        rec = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]),
                            count=n, offset=84)
        return rec['v'].astype(float)

    if n is not None and 84 + 50 * n == len(data):
        return binary()
    verts = []
    if data.lstrip()[:5].lower() == b'solid':
        verts = [list(map(float, line.split()[1:4])) for line in data.decode('ascii', 'replace').splitlines()
                 if line.strip().lower().startswith('vertex')]
    if verts and len(verts) % 3 == 0:
        return np.array(verts, float).reshape(-1, 3, 3)
    if n and 84 + 50 * n <= len(data):
        return binary()
    raise ValueError('%s: no triangles read (not a valid ASCII or binary STL)' % path)


def write_stl(path, tris):
    tris = np.asarray(tris, np.float32)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80 + struct.pack('<I', len(tris)))
        for t in tris:
            f.write(struct.pack('<12fH', 0, 0, 0, *t.ravel(), 0))


class Mesh:
    """A closed triangle mesh. Normals are flipped as a whole if the signed volume
    comes out negative, so they point out of the material."""

    def __init__(self, tris, decimals=6):
        tris = np.asarray(tris, float)
        if tris.ndim != 3 or tris.shape[1:] != (3, 3) or len(tris) == 0:
            raise ValueError('expected a non-empty (n, 3, 3) array of triangles')
        if not np.all(np.isfinite(tris)):
            raise ValueError('the mesh has NaN or infinite coordinates')
        keys = np.round(tris.reshape(-1, 3), decimals)
        self.vertices, inv = np.unique(keys, axis=0, return_inverse=True)
        self.faces = inv.reshape(-1, 3)
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        cross = np.cross(b - a, c - a)
        twice = np.linalg.norm(cross, axis=1)
        keep = twice > 1e-12                      # drop zero-area triangles
        _, first = np.unique(np.sort(self.faces, axis=1), axis=0, return_index=True)
        unique = np.zeros(len(self.faces), bool)
        unique[first] = True
        keep &= unique                            # and exact duplicates (a file with a face written twice)
        self.faces, cross, twice = self.faces[keep], cross[keep], twice[keep]
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        self.volume = float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)
        if self.volume < 0:                       # inward-wound file: flip every face
            self.faces = self.faces[:, ::-1]
            cross, self.volume = -cross, -self.volume
            a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        self.normals = cross / twice[:, None]
        self.areas = twice / 2.0
        self.centroids = (a + b + c) / 3.0
        self.tris = np.stack([a, b, c], axis=1)
        self._edges = None

    @classmethod
    def from_stl(cls, path):
        return cls(read_stl(path))

    @property
    def area(self):
        return float(self.areas.sum())

    @property
    def bounds(self):
        used = self.vertices[np.unique(self.faces)]          # not the vertices of dropped faces
        return used.min(0), used.max(0)

    def edges(self):
        """{(v0, v1) sorted: [face, ...]} for every edge."""
        if self._edges is None:
            e = {}
            for f, (i, j, k) in enumerate(self.faces):
                for u, v in ((i, j), (j, k), (k, i)):
                    e.setdefault((u, v) if u < v else (v, u), []).append(f)
            self._edges = e
        return self._edges

    def neighbours(self):
        """Face adjacency lists through shared edges."""
        nb = [[] for _ in range(len(self.faces))]
        for fs in self.edges().values():
            for f in fs:
                nb[f] += [g for g in fs if g != f]
        return nb

    def is_closed(self):
        return all(len(fs) == 2 for fs in self.edges().values())

    def winding_consistent(self):
        """True if no edge is run the same way by two faces. A face wound against
        its neighbours has its normal pointing into the material, and every check
        reads it backwards; the global flip in __init__ cannot fix that."""
        directed = self.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        _, counts = np.unique(directed, axis=0, return_counts=True)
        return bool(np.all(counts == 1))


def components(mask, neighbours):
    """Connected groups of the faces where mask is True."""
    seen = np.zeros(len(mask), bool)
    groups = []
    for start in np.flatnonzero(mask):
        if seen[start]:
            continue
        stack, group = [start], []
        seen[start] = True
        while stack:
            f = stack.pop()
            group.append(f)
            for g in neighbours[f]:
                if mask[g] and not seen[g]:
                    seen[g] = True
                    stack.append(g)
        groups.append(np.array(group))
    return groups
