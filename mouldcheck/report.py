"""Run every check, print a summary, write JSON and figures.

    python -m mouldcheck part.stl --pull 0 0 1 --min-draft 0.5 --out results/
"""
import argparse
import json
import os

import numpy as np

from . import checks
from .mesh import Mesh
from .raycast import RayCaster

DRAFT_COLOURS = {'plus': (0.27, 0.60, 0.86), 'minus': (0.93, 0.64, 0.27), 'flat': (0.80, 0.81, 0.83),
                 'low': (0.98, 0.80, 0.18)}
UNDERCUT = (0.86, 0.16, 0.16)
NEGLIGIBLE = (0.85, 0.55, 0.85)


def analyse(mesh, pull, min_draft=0.5):
    caster = RayCaster(mesh.tris)
    ang, cls, dsum = checks.draft(mesh, pull, min_draft)
    um, usum = checks.undercuts(mesh, pull, caster=caster)
    t, tsum = checks.thickness(mesh, caster=caster)
    loops, psum = checks.parting_lines(mesh, pull)
    lo, hi = mesh.bounds
    return {
        'part': {'triangles': int(len(mesh.faces)), 'closed': mesh.is_closed(),
                 'consistent_winding': mesh.winding_consistent(), 'volume': mesh.volume,
                 'area': mesh.area, 'bounds': [lo.tolist(), hi.tolist()]},
        'pull': [float(x) for x in checks.unit(pull)],
        'draft': dsum, 'undercuts': usum, 'thickness': tsum,
        'parting': {**psum, 'loops_detail': [{k: v for k, v in l.items() if k != 'points'} for l in loops]},
        '_faces': {'draft': ang, 'class': cls, 'undercut': um, 'thickness': t, 'loops': loops},
    }


def text_summary(r, name='part'):
    d, u, t, p = r['draft'], r['undercuts'], r['thickness'], r['parting']
    total = r['part']['area']
    lines = [
        '%s: %d triangles, %s, volume %.1f mm3' % (name, r['part']['triangles'],
                                                   'closed' if r['part']['closed'] else 'NOT CLOSED',
                                                   r['part']['volume']),
    ]
    if not r['part'].get('consistent_winding', True):
        lines.append('WARNING: some faces are wound against their neighbours (normals point inward); '
                     'repair the mesh, the results are unreliable')
    lines += [
        'pull %s' % ', '.join('%.3g' % x for x in r['pull']),
        'draft (min %.2f deg): smallest wall draft %s deg; under minimum %.1f mm2 (%.1f %%) in %d region(s)'
        % (d['min_draft'], '%.2f' % d['smallest_wall_draft'] if d['smallest_wall_draft'] is not None else '-',
           d['area']['low'], 100 * d['area']['low'] / total, len(d['low_regions'])),
        'undercuts: %.1f mm2 needing side action; %.1f mm2 negligible (under %.2f mm deep)'
        % (u['area'], u['negligible_area'], u['min_depth']),
        ('wall: nominal %.2f mm (min %.2f, max %.2f); thick (>1.5x) %.1f mm2, thin (<0.5x) %.1f mm2'
         % (t['nominal'], t['min'], t['max'], t['thick_area'], t['thin_area'])) if t['nominal'] is not None
        else 'wall: not measured (no inward ray met a far wall; is the mesh closed?)',
        'parting line candidates: %d loop(s), longest %.1f mm; zero-draft area %.1f mm2'
        % (p['loops'], p['longest'], p['zero_draft_area']),
    ]
    for reg in u['regions'][:5]:
        if reg['depth'] is None:
            fix = 'no side action or back-draft clears it within %.0f mm' % u['max_depth']
        else:
            fix = 'depth %.3f mm via %s%s' % (reg['depth'], reg['best_fix'],
                                              ' (negligible)' if reg['negligible'] else '')
        lines.append('  undercut %.2f mm2 at (%.1f, %.1f, %.1f), released by %s, %s'
                     % ((reg['area'],) + reg['centre'] + (reg['released_by'], fix)))
    for reg in d['low_regions'][:5]:
        lines.append('  low draft %.2f mm2 at (%.1f, %.1f, %.1f), draft %.2f..%.2f deg'
                     % ((reg['area'],) + reg['centre'] + reg['draft_range']))
    return '\n'.join(lines)


def render(mesh, colours, view, ax, loops=None, title=''):
    """Orthographic view looking along 'view', painter-sorted, flat-shaded."""
    from matplotlib.collections import PolyCollection
    v = checks.unit(view)
    up = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(up, v)
    x /= np.linalg.norm(x)
    y = np.cross(v, x)
    facing = mesh.normals @ v < 0                              # faces turned towards the viewer
    depth = mesh.centroids @ v
    order = np.argsort(-depth)
    order = order[facing[order]]
    light = checks.unit(-v + 0.6 * y + 0.3 * x)
    shade = 0.55 + 0.45 * np.clip(mesh.normals[order] @ light, 0, 1)
    cols = np.asarray(colours)[order] * shade[:, None]
    polys = np.stack([mesh.tris[order] @ x, mesh.tris[order] @ y], axis=-1)
    cols = np.clip(cols, 0, 1)
    # edges in the face colour close the hairline gaps antialiasing leaves between triangles
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.3))
    if loops:
        mid = float(np.median(mesh.centroids @ v))
        for l in loops:
            if float(np.mean(l['points'] @ v)) > mid:     # loop on the far side: hidden
                continue
            p = np.vstack([l['points'], l['points'][:1]])
            ax.plot(p @ x, p @ y, color='k', lw=1.4)
    pts = mesh.vertices
    ax.set_xlim((pts @ x).min() - 2, (pts @ x).max() + 2)
    ax.set_ylim((pts @ y).min() - 2, (pts @ y).max() + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10)


def figures(mesh, r, out, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    f = r['_faces']
    pull = np.array(r['pull'])
    draft_cols = np.array([DRAFT_COLOURS[c] for c in f['class']])
    draft_cols[f['undercut'] == 1] = UNDERCUT
    draft_cols[f['undercut'] == 2] = NEGLIGIBLE
    nominal = r['thickness']['nominal'] or 1.0
    t = np.where(np.isfinite(f['thickness']), f['thickness'], nominal)
    # display only: average each face with its neighbours, so long sliver triangles in
    # a CAD export do not paint streaks across the map (the numbers stay per face)
    nb = mesh.neighbours()
    t = np.array([np.average(t[[i] + nb[i]], weights=mesh.areas[[i] + nb[i]]) for i in range(len(t))])
    tn = np.clip(t / nominal, 0, 2) / 2                                    # 0..2x nominal -> 0..1
    cmap = plt.get_cmap('coolwarm')
    thick_cols = cmap(tn)[:, :3]
    # look at each half from outside: the +pull half from above, the -pull half from below,
    # tilted a little so walls show
    tilt = checks.unit(np.cross(pull, [1, 0, 0]) if abs(pull[0]) < 0.9 else np.cross(pull, [0, 1, 0]))
    views = [(-pull + 0.45 * tilt, '+pull side (moving half)'), (pull + 0.45 * tilt, '-pull side (fixed half)')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for i, (vd, lab) in enumerate(views):
        render(mesh, draft_cols, vd, axes[0, i], f['loops'], 'Draft: ' + lab)
        render(mesh, thick_cols, vd, axes[1, i], None, 'Wall thickness: ' + lab)
    fig.legend(handles=[Patch(color=DRAFT_COLOURS['plus'], label='released +pull'),
                               Patch(color=DRAFT_COLOURS['minus'], label='released -pull'),
                               Patch(color=DRAFT_COLOURS['flat'], label='square to pull'),
                               Patch(color=DRAFT_COLOURS['low'], label='under min draft'),
                               Patch(color=UNDERCUT, label='undercut'),
                               Patch(color=NEGLIGIBLE, label='negligible undercut')],
               loc='upper center', bbox_to_anchor=(0.45, 0.535), ncol=6, fontsize=8, frameon=False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 2 * nominal))
    nom_txt = '%.2f' % r['thickness']['nominal'] if r['thickness']['nominal'] is not None else 'not measured'
    fig.colorbar(sm, ax=axes[1, :], shrink=0.6, label='wall thickness, mm (nominal %s)' % nom_txt)
    fig.suptitle('%s: undercuts %.1f mm2, smallest wall draft %s deg, nominal wall %s mm'
                 % (name, r['undercuts']['area'],
                    '%.2f' % r['draft']['smallest_wall_draft'] if r['draft']['smallest_wall_draft'] is not None
                    else '-',
                    nom_txt), fontsize=11)
    path = os.path.join(out, name + '.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog='mouldcheck', description=__doc__.split('\n')[0])
    ap.add_argument('stl')
    ap.add_argument('--pull', type=float, nargs=3, default=(0, 0, 1), help='mould opening direction')
    ap.add_argument('--min-draft', type=float, default=0.5, help='degrees')
    ap.add_argument('--out', default=None, help='folder for the JSON report and figure')
    a = ap.parse_args(argv)
    mesh = Mesh.from_stl(a.stl)
    r = analyse(mesh, a.pull, a.min_draft)
    name = os.path.splitext(os.path.basename(a.stl))[0]
    print(text_summary(r, name))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, name + '.json'), 'w') as fp:
            json.dump({k: v for k, v in r.items() if k != '_faces'}, fp, indent=1)
        print('figure', figures(mesh, r, a.out, name))
    return r
