"""Tests against parts whose answers can be worked out by hand.
Run from the repository root: python -m unittest discover tests"""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mouldcheck import Mesh, analyse, checks, read_stl, shapes, write_stl  # noqa: E402
from mouldcheck.raycast import RayCaster  # noqa: E402

PULL = (0, 0, 1)


class MeshTests(unittest.TestCase):
    def test_cup_is_closed_with_the_right_volume(self):
        m = Mesh(shapes.cup(segments=256))
        self.assertTrue(m.is_closed())
        # revolved polygon: area x 2 pi x centroid radius (Pappus), for the 256-gon it is a hair less
        self.assertGreater(m.volume, 0)
        self.assertAlmostEqual(m.volume, 9400, delta=30)

    def test_inside_out_file_is_flipped(self):
        t = shapes.cup()
        m = Mesh(t[:, ::-1, :])                    # reverse every triangle's winding
        self.assertGreater(m.volume, 0)
        self.assertLess(m.normals[np.argmin(m.centroids[:, 2])] @ np.array(PULL), -0.99)   # base faces down

    def test_winding_check(self):
        t = shapes.cup()
        self.assertTrue(Mesh(t).winding_consistent())
        t[5] = t[5][::-1]                          # one face wound against its neighbours
        self.assertFalse(Mesh(t).winding_consistent())

    def test_stl_roundtrip(self):
        path = os.path.join(tempfile.mkdtemp(), 'c.stl')
        write_stl(path, shapes.cup())
        self.assertEqual(read_stl(path).shape, shapes.cup().shape)


class RayTests(unittest.TestCase):
    def test_distance_through_a_plate(self):
        t = shapes.revolve([(0, 0), (50, 0), (50, 3), (0, 3)], segments=32)
        c = RayCaster(t)
        d = c.first_hit(np.array([[1.0, 2.0, -5.0], [1.0, 2.0, 10.0]]), np.array([[0, 0, 1.0], [0, 0, 1.0]]))
        self.assertAlmostEqual(d[0], 5.0, places=6)
        self.assertTrue(np.isinf(d[1]))


class DraftTests(unittest.TestCase):
    def test_draft_of_a_drafted_cup(self):
        for deg in (0.5, 1.0, 2.0, 3.0):
            m = Mesh(shapes.cup(draft_deg=deg, segments=128))
            ang, cls, s = checks.draft(m, PULL, min_draft=0.25)
            self.assertAlmostEqual(s['smallest_wall_draft'], deg, delta=0.01)
            self.assertEqual(s['area']['low'], 0.0)
            # outer wall leaves the -pull half, inner wall the +pull half
            outer = np.hypot(*m.centroids[:, :2].T) > 20 - 1e-6
            walls = np.abs(m.normals[:, 2]) < 0.5
            self.assertTrue(np.all(cls[outer & walls] == 'minus'))
            self.assertTrue(np.all(cls[~outer & walls] == 'plus'))

    def test_zero_draft_is_flagged(self):
        m = Mesh(shapes.cup(draft_deg=0.0))
        ang, cls, s = checks.draft(m, PULL, min_draft=0.5)
        wall_area = 2 * math.pi * 20 * 30 + 2 * math.pi * 18 * 28
        self.assertAlmostEqual(s['area']['low'], wall_area, delta=wall_area * 0.002)
        self.assertGreaterEqual(len(s['low_regions']), 2)


class UndercutTests(unittest.TestCase):
    def test_plain_cup_has_none(self):
        m = Mesh(shapes.cup())
        mask, s = checks.undercuts(m, PULL)
        self.assertFalse(mask.any())

    def test_internal_groove_depth(self):
        # the steel at the top of the groove's back wall must move the full groove
        # depth to pass the lip above it (the wall's draft only helps lower down)
        for depth in (0.5, 1.0, 1.5):              # the wall is 2.0, so 2.0 would cut through it
            m = Mesh(shapes.cup(groove=(15, 2, depth), segments=128))
            mask, s = checks.undercuts(m, PULL)
            self.assertEqual(len(s['regions']), 1)
            reg = s['regions'][0]
            self.assertFalse(reg['negligible'])
            self.assertEqual(reg['best_fix'], 'side action')
            self.assertAlmostEqual(reg['depth'], depth, delta=0.005)
            self.assertTrue(np.all(mask[mask > 0] == 1))

    def test_no_fix_within_reach_is_unbounded(self):
        m = Mesh(shapes.cup(groove=(15, 2, 1.0), segments=64))
        mask, s = checks.undercuts(m, PULL, max_depth=0.5)
        self.assertIsNone(s['regions'][0]['depth'])
        self.assertIsNone(s['regions'][0]['best_fix'])
        self.assertFalse(s['regions'][0]['negligible'])

    def test_pull_reversed_gives_same_answer(self):
        m = Mesh(shapes.cup(groove=(15, 2, 1.0)))
        _, a = checks.undercuts(m, (0, 0, 1))
        _, b = checks.undercuts(m, (0, 0, -1))
        self.assertAlmostEqual(a['area'], b['area'], places=6)


class ThicknessTests(unittest.TestCase):
    def test_nominal_and_flags(self):
        m = Mesh(shapes.cup())
        t, s = checks.thickness(m)
        self.assertAlmostEqual(s['nominal'], 2.0, delta=0.01)
        self.assertEqual(s['thick_area'], 0.0)
        self.assertEqual(s['thin_area'], 0.0)

    def test_thick_floor_is_flagged(self):
        m = Mesh(shapes.cup(floor=6.0))
        t, s = checks.thickness(m)
        self.assertAlmostEqual(s['nominal'], 2.0, delta=0.01)
        self.assertAlmostEqual(s['max'], 6.0, delta=0.1)
        floor_area = math.pi * 20 ** 2 + math.pi * (18 + 6 * math.tan(math.radians(2))) ** 2
        self.assertAlmostEqual(s['thick_area'], floor_area, delta=0.12 * floor_area)

    def test_groove_wall_is_thin(self):
        m = Mesh(shapes.cup(groove=(15, 2, 1.0)))
        t, s = checks.thickness(m)
        self.assertGreater(s['thin_area'], 2 * math.pi * 19 * 2 * 0.9)
        self.assertAlmostEqual(s['min'], 1.0, delta=0.2)


class PartingTests(unittest.TestCase):
    def test_one_flat_loop_at_the_rim(self):
        m = Mesh(shapes.cup(segments=128))
        loops, s = checks.parting_lines(m, PULL)
        self.assertEqual(s['loops'], 1)
        self.assertTrue(loops[0]['planar'])
        r_top = 20 + 30 * math.tan(math.radians(2))
        self.assertAlmostEqual(loops[0]['length'], 2 * 128 * r_top * math.sin(math.pi / 128), places=6)
        self.assertAlmostEqual(loops[0]['height'][0], 30.0, places=9)


class SnapTests(unittest.TestCase):
    def test_matches_hand_calculation(self):
        # ABS arm: L 12.1, t 1.2, b 12, y 0.6, E 2200 MPa, yield 2.6 %, 30 deg lead-in, mu 0.6
        r = checks.cantilever_snap(12.1, 1.2, 12.0, 0.6, 2200.0, 0.026, lead_deg=30, mu=0.6)
        self.assertAlmostEqual(r['strain'], 1.5 * 1.2 * 0.6 / 12.1 ** 2)
        self.assertAlmostEqual(r['allowable'], 0.7 * 0.026 * 0.6)
        self.assertAlmostEqual(r['deflection_force'], 3.8626, places=3)
        self.assertAlmostEqual(r['mating_force'], 6.958, places=2)
        self.assertAlmostEqual(r['strain'] * r['max_deflection'] / 0.6, r['allowable'], places=9)

    def test_self_locking_lead_in(self):
        with self.assertRaises(ValueError):
            checks.cantilever_snap(10, 1, 5, 0.5, 2000, 0.03, lead_deg=65, mu=0.6)


class ReportTests(unittest.TestCase):
    def test_analyse_and_figure(self):
        from mouldcheck.report import figures, text_summary
        m = Mesh(shapes.cup(groove=(15, 2, 1.0)))
        r = analyse(m, PULL)
        txt = text_summary(r, 'grooved')
        self.assertIn('needing side action', txt)
        path = figures(m, r, tempfile.mkdtemp(), 'grooved')
        self.assertGreater(os.path.getsize(path), 10000)


if __name__ == '__main__':
    unittest.main()
