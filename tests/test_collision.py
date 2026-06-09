"""Tests for AABB broadphase + SAT narrowphase collision detection."""
import jax.numpy as jnp
import pytest

from spec_driven_xpbd.engine.collision import (
    _broadphase,
    _sat_result,
    _vertex_contacts,
    _edge_edge_contact,
    collect_contacts,
)
from spec_driven_xpbd.engine.state import SimState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_boxes(x0, h0, minv0, x1, h1, minv1, v0=None, v1=None):
    """Build a 2-body SimState with identity quaternions."""
    if v0 is None:
        v0 = [0., 0., 0.]
    if v1 is None:
        v1 = [0., 0., 0.]
    return SimState(
        x=jnp.array([x0, x1]),
        v=jnp.array([v0, v1]),
        q=jnp.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        omega=jnp.zeros((2, 3)),
        m_inv=jnp.array([minv0, minv1]),
        I_inv_body=jnp.zeros((2, 3)),
        half_ext=jnp.array([h0, h1]),
    )


# Canonical scene reused across tests:
#   body 0 = static slab  half_ext (2, 0.1, 2) at y=0
#   body 1 = dynamic cube half_ext (0.25, 0.25, 0.25) at y=0.33
#   cube bottom = 0.33 - 0.25 = 0.08  <  slab top = 0.1  → 0.02 m overlap
_SLAB_H = [2., 0.1, 2.]
_CUBE_H = [0.25, 0.25, 0.25]
_DT = 1 / 60.


def _penetrating_state(cube_y=0.33):
    return _two_boxes(
        x0=[0., 0., 0.], h0=_SLAB_H, minv0=0.,
        x1=[0., cube_y, 0.], h1=_CUBE_H, minv1=1.,
    )


def _separated_state(cube_y=2.0):
    return _two_boxes(
        x0=[0., 0., 0.], h0=_SLAB_H, minv0=0.,
        x1=[0., cube_y, 0.], h1=_CUBE_H, minv1=1.,
    )


# ---------------------------------------------------------------------------
# Broadphase
# ---------------------------------------------------------------------------

class TestBroadphase:
    def test_overlapping_aabbs_are_candidates(self):
        state = _penetrating_state()
        assert _broadphase(state, 0, 1, _DT)

    def test_separated_aabbs_are_not_candidates(self):
        state = _separated_state(cube_y=5.)
        assert not _broadphase(state, 0, 1, _DT)

    def test_static_static_pair_is_skipped(self):
        # Both m_inv=0 → never a candidate even if overlapping.
        state = _two_boxes(
            x0=[0., 0., 0.], h0=_SLAB_H, minv0=0.,
            x1=[0., 0.33, 0.], h1=_CUBE_H, minv1=0.,
        )
        assert not _broadphase(state, 0, 1, _DT)

    def test_velocity_expansion_catches_near_miss(self):
        # Cube is 0.01 m above the slab at rest but moving down fast enough
        # that the expanded AABB should overlap.
        cube_y = 0.1 + 0.25 + 0.01  # just above contact, gap = 0.01 m
        v_cube = [0., -5., 0.]        # k=2, dt=1/60 → expansion = 2*(1/60)*5 ≈ 0.167 > 0.01
        state = _two_boxes(
            x0=[0., 0., 0.], h0=_SLAB_H, minv0=0.,
            x1=[0., cube_y, 0.], h1=_CUBE_H, minv1=1.,
            v1=v_cube,
        )
        assert _broadphase(state, 0, 1, _DT)


# ---------------------------------------------------------------------------
# SAT
# ---------------------------------------------------------------------------

class TestSAT:
    def test_penetrating_boxes_not_separated(self):
        state = _penetrating_state()
        not_sep, _, _ = _sat_result(state, 0, 1)
        assert not_sep
        not_sep, _, _ = _sat_result(state, 1, 0)
        assert not_sep

    def test_separated_boxes_are_separated(self):
        state = _separated_state(cube_y=5.)
        not_sep, _, _ = _sat_result(state, 0, 1)
        assert not not_sep
        not_sep, _, _ = _sat_result(state, 1, 0)
        assert not not_sep

    def test_just_touching_not_separated(self):
        state = _separated_state(cube_y=0.35)
        not_sep, _, _ = _sat_result(state, 0, 1)
        assert not_sep

    def test_boxes_separated_along_x(self):
        state = _two_boxes(
            x0=[0., 0., 0.], h0=[0.5, 0.5, 0.5], minv0=1.,
            x1=[2., 0., 0.], h1=[0.5, 0.5, 0.5], minv1=1.,
        )
        not_sep, _, _ = _sat_result(state, 0, 1)
        assert not not_sep

    def test_returns_three_tuple(self):
        state = _penetrating_state()
        not_sep, min_axis, min_depth = _sat_result(state, 0, 1)
        assert not_sep.shape == ()
        assert int(min_axis) in range(15)
        assert float(min_depth) >= 0.0

    def test_face_contact_axis_is_face(self):
        # Axis-aligned stack: min axis must be a face normal (0–5), not edge-edge.
        state = _penetrating_state()
        _, min_axis, _ = _sat_result(state, 0, 1)
        assert int(min_axis) < 6


# ---------------------------------------------------------------------------
# Vertex contact generation
# ---------------------------------------------------------------------------

class TestVertexContacts:
    def test_four_bottom_vertices_active(self):
        # Cube at y=0.33: bottom face at y=0.08 inside slab (slab top y=0.1).
        # pair (1, 0): 8 vertices of body 1 (cube) tested against body 0 (slab).
        state = _penetrating_state()
        r_a, r_b, n, d = _vertex_contacts(state, i=1, j=0)
        active = d > 0
        assert int(jnp.sum(active)) == 4

    def test_penetration_depth_correct(self):
        # 0.02 m penetration expected.
        state = _penetrating_state(cube_y=0.33)
        _, _, _, d = _vertex_contacts(state, i=1, j=0)
        active_d = d[d > 0]
        assert jnp.allclose(active_d, 0.02, atol=1e-5)

    def test_normal_points_from_slab_toward_cube(self):
        # n should be (0, 1, 0) for all active contacts.
        state = _penetrating_state()
        _, _, n, d = _vertex_contacts(state, i=1, j=0)
        active_n = n[d > 0]
        expected = jnp.array([0., 1., 0.])
        assert jnp.allclose(active_n, expected[None, :], atol=1e-6)

    def test_no_active_contacts_when_separated(self):
        state = _separated_state(cube_y=2.)
        _, _, _, d = _vertex_contacts(state, i=1, j=0)
        assert not jnp.any(d > 0)

    def test_r_a_relative_to_body_a_center(self):
        state = _penetrating_state()
        r_a, _, _, d = _vertex_contacts(state, i=1, j=0)
        # Active vertices are the 4 bottom corners of the cube:
        # r_a = vertex - x_cube = (±0.25, -0.25, ±0.25) in cube frame (cube at y=0.33)
        active_r_a = r_a[d > 0]
        assert jnp.allclose(jnp.abs(active_r_a[:, 0]), 0.25, atol=1e-6)
        assert jnp.allclose(active_r_a[:, 1], -0.25, atol=1e-6)
        assert jnp.allclose(jnp.abs(active_r_a[:, 2]), 0.25, atol=1e-6)


# ---------------------------------------------------------------------------
# collect_contacts (end-to-end)
# ---------------------------------------------------------------------------

class TestCollectContacts:
    def test_single_body_returns_empty_buffer(self):
        state = SimState(
            x=jnp.array([[0., 1., 0.]]),
            v=jnp.zeros((1, 3)),
            q=jnp.array([[1., 0., 0., 0.]]),
            omega=jnp.zeros((1, 3)),
            m_inv=jnp.array([1.]),
            I_inv_body=jnp.zeros((1, 3)),
            half_ext=jnp.array([[0.5, 0.5, 0.5]]),
        )
        contacts = collect_contacts(state, _DT)
        assert contacts.d.shape == (0,)

    def test_buffer_size_is_8_N_Nm1(self):
        state = _penetrating_state()
        contacts = collect_contacts(state, _DT)
        N = 2
        assert contacts.d.shape == (8 * N * (N - 1),)

    def test_four_active_contacts_when_penetrating(self):
        state = _penetrating_state()
        contacts = collect_contacts(state, _DT)
        assert int(jnp.sum(contacts.d > 0)) == 4

    def test_no_active_contacts_when_separated(self):
        state = _separated_state(cube_y=5.)
        contacts = collect_contacts(state, _DT)
        assert not jnp.any(contacts.d > 0)

    def test_active_contact_normals_point_upward(self):
        state = _penetrating_state()
        contacts = collect_contacts(state, _DT)
        active = contacts.d > 0
        active_n = contacts.n[active]
        assert jnp.allclose(active_n, jnp.array([[0., 1., 0.]]), atol=1e-6)

    def test_active_contact_depth(self):
        state = _penetrating_state(cube_y=0.33)
        contacts = collect_contacts(state, _DT)
        active_d = contacts.d[contacts.d > 0]
        assert jnp.allclose(active_d, 0.02, atol=1e-5)

    def test_lambdas_initialised_to_zero(self):
        state = _penetrating_state()
        contacts = collect_contacts(state, _DT)
        assert jnp.all(contacts.lambda_n == 0)
        assert jnp.all(contacts.lambda_t == 0)


# ---------------------------------------------------------------------------
# Regression: argmin face selection for degenerate depth configurations
# ---------------------------------------------------------------------------

class TestVertexContactsArgminFix:
    """Regression tests for the argmin bug that selected a zero-depth (boundary)
    face over the actual penetrating face, producing d=0 (inactive contact).

    Two degenerate configurations:
    A. Same-size boxes stacked: corner vertices land exactly on the containing
       box's edge, giving depths=[0.05, δ, 0.0].  argmin must select y (δ), not z (0).
    B. Box vertex exactly on ground top face: depths=[1.75, 0.0, 1.75].
       argmin must keep y (depth=0, inactive), NOT switch to x (depth=1.75, wrong).
    """

    def test_aligned_stack_normals_point_up(self):
        """Same-size boxes, no x-offset: all bottom-corner depths are [0, δ, 0].
        All 4 active contacts must have n=(0,1,0)."""
        # box1 at y=0.35, box2 1mm inside: bottom at y=0.599, box1 top at 0.60
        state = _two_boxes(
            x0=[0., 0.35, 0.], h0=[0.25, 0.25, 0.25], minv0=1.,
            x1=[0., 0.849, 0.], h1=[0.25, 0.25, 0.25], minv1=1.,
        )
        _, _, n, d = _vertex_contacts(state, i=1, j=0)
        active = d > 0
        assert int(jnp.sum(active)) == 4
        assert jnp.allclose(n[active], jnp.array([0., 1., 0.]), atol=1e-5)

    def test_offset_stack_normals_point_up(self):
        """Same-size boxes, 0.05m x-offset (stack scene config): corner vertex depths
        are [0.05, δ, 0.0].  Active contacts must have n=(0,1,0), not n=(0,0,±1)."""
        state = _two_boxes(
            x0=[0., 0.35, 0.], h0=[0.25, 0.25, 0.25], minv0=1.,
            x1=[0.05, 0.849, 0.], h1=[0.25, 0.25, 0.25], minv1=1.,
        )
        _, _, n, d = _vertex_contacts(state, i=1, j=0)
        active = d > 0
        assert int(jnp.sum(active)) >= 1
        assert jnp.allclose(n[active], jnp.array([0., 1., 0.]), atol=1e-5)

    def test_vertex_on_ground_top_normal_is_y_not_x(self):
        """Box bottom vertices exactly at ground top (depth[y]=0.0, depths[x,z]=1.75).
        The fix must NOT switch to x-axis (which would produce d=1.75 active contacts
        with wrong normal).  Use power-of-two values so vertex y is exact in float32:
        ground h_b_y=0.25, box center y=0.5 → bottom at y=0.25 = ground top exactly."""
        state = _two_boxes(
            x0=[0., 0., 0.], h0=[2., 0.25, 2.], minv0=0.,
            x1=[0., 0.5, 0.], h1=[0.25, 0.25, 0.25], minv1=1.,
        )
        _, _, n, d = _vertex_contacts(state, i=1, j=0)
        # No contact should be active with an x- or z-direction normal.
        # (A large spurious d=1.75 in x-direction would cause massive instability.)
        spurious = (d > 0) & (jnp.abs(n[:, 1]) < 0.5)
        assert not jnp.any(spurious)


# ---------------------------------------------------------------------------
# Edge-edge contact generation
# ---------------------------------------------------------------------------

import math
import dataclasses

def _edge_edge_boxes():
    """Two unit cubes set up for a genuine edge-edge SAT contact.

    Box 0: rotated 45° around Z, at origin.
    Box 1: rotated 45° around X, at (D/√3, -D/√3, D/√3) with D=1.35.

    The cross product of box 0's X-edge (≈(√2/2, √2/2, 0)) and box 1's
    Y-edge (≈(0, √2/2, √2/2)) is (1,−1,1)/√3, which is NOT a face normal of
    either box.  At D=1.35 the overlap along that axis is ≈0.044 m, while all
    six face-normal overlaps are ≥ 0.40 m.  The SAT minimum is axis index 7
    (ea=0, eb=1 → 6 + 0*3 + 1).
    """
    h = 0.5
    c = math.cos(math.pi / 8)   # cos(22.5°)
    s = math.sin(math.pi / 8)   # sin(22.5°)
    q0 = [c, 0., 0., s]         # 45° around Z
    q1 = [c, s,  0., 0.]        # 45° around X

    D = 1.35
    r3 = math.sqrt(3.0)
    state = _two_boxes(
        x0=[0., 0., 0.],           h0=[h, h, h], minv0=1.,
        x1=[D/r3, -D/r3, D/r3],   h1=[h, h, h], minv1=1.,
    )
    q = state.q.at[0].set(jnp.array(q0)).at[1].set(jnp.array(q1))
    return dataclasses.replace(state, q=q)


class TestEdgeEdgeContact:
    def test_sat_min_axis_is_edge_edge(self):
        state = _edge_edge_boxes()
        _, min_axis, _ = _sat_result(state, 0, 1)
        assert int(min_axis) >= 6, f"expected edge-edge axis (>=6), got {int(min_axis)}"

    def test_edge_edge_depth_positive(self):
        state = _edge_edge_boxes()
        _, min_axis, _ = _sat_result(state, 0, 1)
        r_a, r_b, n, d = _edge_edge_contact(state, 0, 1, min_axis)
        assert float(d) > 0.0, f"expected d>0, got {float(d)}"

    def test_edge_edge_normal_perpendicular_to_both_edges(self):
        state = _edge_edge_boxes()
        _, min_axis, _ = _sat_result(state, 0, 1)
        r_a, r_b, n, d = _edge_edge_contact(state, 0, 1, min_axis)
        from spec_driven_xpbd.engine.math import quat_to_mat
        R0 = quat_to_mat(state.q[0]); R1 = quat_to_mat(state.q[1])
        ea = (int(min_axis) - 6) // 3; eb = (int(min_axis) - 6) % 3
        assert abs(float(jnp.dot(n, R0[:, ea]))) < 1e-5
        assert abs(float(jnp.dot(n, R1[:, eb]))) < 1e-5

    def test_edge_edge_normal_points_from_j_toward_i(self):
        state = _edge_edge_boxes()
        _, min_axis, _ = _sat_result(state, 0, 1)
        r_a, r_b, n, d = _edge_edge_contact(state, 0, 1, min_axis)
        d_i_to_j = state.x[1] - state.x[0]
        assert float(jnp.dot(n, d_i_to_j)) < 0.0, "n should point from j toward i"

    def test_collect_contacts_has_active_edge_edge(self):
        state = _edge_edge_boxes()
        contacts = collect_contacts(state, _DT)
        assert int(jnp.sum(contacts.d > 0)) >= 1


# ---------------------------------------------------------------------------
# Face-face flat stack
# ---------------------------------------------------------------------------

class TestFaceFlatStack:
    def test_four_contacts_on_correct_face(self):
        # Two unit cubes stacked flat with 2 mm overlap.
        # Cube 0 center at y=0, cube 1 center at y=0.998 → bottom of 1 at y=0.498 < top of 0 at y=0.5.
        state = _two_boxes(
            x0=[0., 0., 0.], h0=[0.5, 0.5, 0.5], minv0=1.,
            x1=[0., 0.998, 0.], h1=[0.5, 0.5, 0.5], minv1=1.,
        )
        contacts = collect_contacts(state, _DT)
        active = contacts.d > 0
        assert int(jnp.sum(active)) >= 4
        # Both ordered pairs fire: (1,0) gives n=+Y, (0,1) gives n=-Y.
        # All active normals must be in the ±Y direction.
        assert jnp.all(jnp.abs(contacts.n[active, 1]) > 0.99)

    def test_sat_min_axis_is_face(self):
        state = _two_boxes(
            x0=[0., 0., 0.], h0=[0.5, 0.5, 0.5], minv0=1.,
            x1=[0., 0.998, 0.], h1=[0.5, 0.5, 0.5], minv1=1.,
        )
        _, min_axis, _ = _sat_result(state, 0, 1)
        assert int(min_axis) < 6
