import unittest
from fractions import Fraction as F

from rt_visibility_oracle import Hit, first_hit, is_visible


TRI = ((0, 0, 0), (4, 0, 0), (0, 4, 0))


def require_hit(hit: Hit | None) -> Hit:
    if hit is None:
        raise AssertionError('Expected an exact triangle hit')
    return hit


class CrossingTests(unittest.TestCase):
    def test_clear_hit_at_exact_fraction(self):
        hit = require_hit(first_hit((1, 1, -1), (1, 1, 2), [TRI]))
        self.assertIsInstance(hit.entry_t, F)
        self.assertEqual((hit.entry_t, hit.triangle_indices), (F(1, 3), (0,)))
        self.assertFalse(is_visible((1, 1, -1), (1, 1, 2), [TRI]))

    def test_plane_crossing_outside_triangle_is_visible(self):
        self.assertTrue(is_visible((3, 3, -1), (3, 3, 2), [TRI]))

    def test_plane_crossing_beyond_segment_is_visible(self):
        self.assertTrue(is_visible((1, 1, 1), (1, 1, 2), [TRI]))

    def test_parallel_off_plane_is_visible(self):
        self.assertTrue(is_visible((1, 1, 1), (2, 1, 1), [TRI]))

    def test_reversed_winding_still_blocks(self):
        self.assertFalse(is_visible((1, 1, -1), (1, 1, 2), [TRI[::-1]]))

    def test_empty_scene_is_visible(self):
        self.assertTrue(is_visible((1, 1, -1), (1, 1, 2), []))


class SegmentBoundaryTests(unittest.TestCase):
    def test_endpoint_only_contacts_are_excluded_by_default(self):
        on, off = (1, 1, 0), (1, 1, 1)
        for start, end in ((on, off), (off, on)):
            with self.subTest(start=start, end=end):
                self.assertTrue(is_visible(start, end, [TRI]))

    def test_closed_segment_reports_both_endpoint_parameters(self):
        on, off = (1, 1, 0), (1, 1, 1)
        for start, end, t in ((on, off, F(0)), (off, on, F(1))):
            with self.subTest(t=t):
                hit = require_hit(first_hit(start, end, [TRI], include_endpoints=True))
                self.assertEqual(hit.entry_t, t)

    def test_coplanar_crossing_reports_entry(self):
        hit = require_hit(first_hit((-1, 1, 0), (5, 1, 0), [TRI]))
        self.assertEqual(hit.entry_t, F(1, 6))

    def test_coplanar_overlap_starting_inside_has_entry_infimum_zero(self):
        hit = require_hit(first_hit((1, 1, 0), (5, 1, 0), [TRI]))
        self.assertEqual(hit.entry_t, F(0))

    def test_coplanar_endpoint_only_touch_is_not_overlap(self):
        start, end = (-1, 1, 0), (0, 1, 0)
        self.assertTrue(is_visible(start, end, [TRI]))
        self.assertFalse(is_visible(start, end, [TRI], include_endpoints=True))

    def test_coplanar_interior_vertex_tangency_blocks(self):
        hit = require_hit(first_hit((-1, 1, 0), (1, -1, 0), [TRI]))
        self.assertEqual(hit.entry_t, F(1, 2))

    def test_coplanar_disjoint_segment_is_visible(self):
        self.assertTrue(is_visible((-2, 1, 0), (-1, 1, 0), [TRI]))

    def test_zero_length_is_empty_by_default_or_a_closed_point_query(self):
        on, outside, off = (1, 1, 0), (3, 3, 0), (1, 1, 1)
        self.assertTrue(is_visible(on, on, [TRI]))
        self.assertEqual(require_hit(first_hit(on, on, [TRI], include_endpoints=True)).entry_t, F(0))
        for p in (outside, off):
            with self.subTest(point=p):
                self.assertTrue(is_visible(p, p, [TRI], include_endpoints=True))

    def test_zero_area_triangles_are_ignored_even_on_contact(self):
        for triangle in (
            ((0, 0, 0), (1, 1, 0), (2, 2, 0)),
            ((1, 1, 0), (1, 1, 0), (1, 1, 0)),
        ):
            with self.subTest(triangle=triangle):
                self.assertTrue(is_visible((1, 1, -1), (1, 1, 1), [triangle]))
                self.assertTrue(is_visible((0, 0, 0), (2, 2, 0), [triangle]))
                self.assertTrue(is_visible((1, 1, 0), (1, 1, 0), [triangle],
                                           include_endpoints=True))


def triangle_at(z):
    return tuple((x, y, z) for x, y, _ in TRI)


class SceneOrderingTests(unittest.TestCase):
    def test_shared_edge_reports_all_first_hit_indices_in_scene_order(self):
        neighbor = ((4, 0, 0), (4, 4, 0), (0, 4, 0))
        degenerate = ((2, 2, 0),) * 3
        scene = iter([triangle_at(1), TRI, degenerate, neighbor[::-1]])
        hit = require_hit(first_hit((2, 2, -1), (2, 2, 2), scene))
        self.assertEqual((hit.entry_t, hit.triangle_indices), (F(1, 3), (1, 3)))

    def test_coplanar_shared_edge_overlap_ties_at_entry(self):
        neighbor = ((0, 0, 0), (4, 0, 0), (0, -4, 0))
        hit = require_hit(first_hit((-1, 0, 0), (5, 0, 0), [TRI, neighbor]))
        self.assertEqual((hit.entry_t, hit.triangle_indices), (F(1, 6), (0, 1)))

    def test_distinct_nearly_coincident_layers_do_not_tie(self):
        near = F(1, 2)
        far = near + F(1, 2**100)
        scene = [triangle_at(far), triangle_at(near), triangle_at(far)]
        hit = require_hit(first_hit((1, 1, 0), (1, 1, 1), scene))
        self.assertEqual((hit.entry_t, hit.triangle_indices), (near, (1,)))

    def test_float_coordinate_is_interpreted_as_stored(self):
        hit = require_hit(first_hit((1, 1, 0), (1, 1, 1), [triangle_at(0.1)]))
        self.assertEqual(hit.entry_t, F(*0.1.as_integer_ratio()))
        self.assertNotEqual(hit.entry_t, F(1, 10))

    def test_oblique_triangle_with_rational_coordinates(self):
        triangle = ((0, 0, 0), (2, 0, 2), (0, 2, 2))
        hit = require_hit(first_hit((F(1, 3), F(1, 2), 0), (F(1, 3), F(1, 2), 2), [triangle]))
        self.assertEqual(hit.entry_t, F(5, 12))
