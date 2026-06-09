"""Box-box collision detection: AABB broadphase + SAT narrowphase."""

import jax
import jax.numpy as jnp

from .math import quat_to_mat
from .state import SimState, ContactBuffer, empty_contact_buffer

_K = 2.0  # AABB velocity-expansion safety multiplier (paper §3.5)

# Sign combinations for the 8 vertices of a unit box, shape (8, 3).
_VERTEX_SIGNS = jnp.array([
    [-1., -1., -1.], [-1., -1.,  1.], [-1.,  1., -1.], [-1.,  1.,  1.],
    [ 1., -1., -1.], [ 1., -1.,  1.], [ 1.,  1., -1.], [ 1.,  1.,  1.],
])


def broadphase_pairs(state: SimState, dt: float, exclude_pairs: frozenset = frozenset()) -> jax.Array:
    """AABB broadphase for all ordered pairs.

    Returns a bool array of shape (N*(N-1),) — True where the pair is a
    candidate for narrowphase.  Called once per full timestep.

    exclude_pairs: frozenset of (i, j) int tuples to unconditionally exclude
    (e.g. hinge-connected bodies that should never collide).
    """
    N = state.x.shape[0]
    if N < 2:
        return jnp.zeros(0, dtype=bool)
    ii, jj = _pair_indices(N)
    mask = jax.vmap(lambda i, j: _broadphase(state, i, j, dt))(ii, jj)
    if exclude_pairs:
        # Build allow mask from pure-Python indices before any JAX tracing.
        py_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        allow = jnp.array([(i, j) not in exclude_pairs for i, j in py_pairs])
        mask = mask & allow
    return mask


def narrowphase_contacts(state: SimState, pair_mask: jax.Array) -> ContactBuffer:
    """SAT + vertex contacts for broadphase candidates.

    Called once per substep from the current (post-integrate) state so that
    contact normals, positions, and penetration depths are always fresh.
    Returns a fixed-size ContactBuffer of capacity C = 8*N*(N-1).
    """
    N = state.x.shape[0]
    C = 8 * N * (N - 1)
    if N < 2:
        return empty_contact_buffer(C, N)
    ii, jj = _pair_indices(N)

    # SAT is symmetric: compute once per unordered pair, broadcast via lookup table.
    ui, uj = _unordered_pair_indices(N)
    not_sep_vals, min_axis_vals, _ = jax.vmap(
        lambda i, j: _sat_result(state, i, j)
    )(ui, uj)
    sat_not_sep_table = (jnp.zeros((N, N), dtype=bool)
                         .at[ui, uj].set(not_sep_vals)
                         .at[uj, ui].set(not_sep_vals))
    # min_axis only set for ui<uj entries; uj>ui slots default to 0 (harmless —
    # edge-edge generation is gated on i<j so those slots are never read).
    sat_min_axis_table = (jnp.zeros((N, N), dtype=jnp.int32)
                          .at[ui, uj].set(min_axis_vals))

    def process_pair(i, j, active):
        not_sep = sat_not_sep_table[i, j]
        min_axis = sat_min_axis_table[i, j]

        # Vertex-face contacts (8 slots).
        r_a, r_b, n, d = _vertex_contacts(state, i, j)
        d = jnp.where(active & not_sep, d, jnp.full(8, -1.0))

        # Edge-edge contact overwrites slot 0 when i<j and min_axis is an edge axis.
        is_edge_edge = (min_axis >= 6) & (i < j)
        do_ee = active & not_sep & is_edge_edge
        ee_r_a, ee_r_b, ee_n, ee_d = _edge_edge_contact(state, i, j, min_axis)
        ee_d_active = jnp.where(do_ee, ee_d, -1.0)

        r_a = r_a.at[0].set(jnp.where(do_ee, ee_r_a, r_a[0]))
        r_b = r_b.at[0].set(jnp.where(do_ee, ee_r_b, r_b[0]))
        n   = n  .at[0].set(jnp.where(do_ee, ee_n,   n[0]))
        d   = d  .at[0].set(jnp.where(do_ee, ee_d_active, d[0]))

        return ContactBuffer(
            body_a=jnp.broadcast_to(i, (8,)).astype(jnp.int32),
            body_b=jnp.broadcast_to(j, (8,)).astype(jnp.int32),
            r_a=r_a, r_b=r_b, n=n, d=d,
            lambda_n=jnp.zeros(8), lambda_t=jnp.zeros(8),
        )

    per_pair = jax.vmap(process_pair)(ii, jj, pair_mask)
    return ContactBuffer(
        body_a=per_pair.body_a.reshape(C),
        body_b=per_pair.body_b.reshape(C),
        r_a=per_pair.r_a.reshape(C, 3),
        r_b=per_pair.r_b.reshape(C, 3),
        n=per_pair.n.reshape(C, 3),
        d=per_pair.d.reshape(C),
        lambda_n=per_pair.lambda_n.reshape(C),
        lambda_t=per_pair.lambda_t.reshape(C),
    )


def collect_contacts(state: SimState, dt: float) -> ContactBuffer:
    """Broadphase + narrowphase in one call.  Used by tests."""
    return narrowphase_contacts(state, broadphase_pairs(state, dt))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_indices(N: int):
    """Static ordered-pair index arrays (i, j) for i != j, shape (N*(N-1),)."""
    ii = jnp.array([i for i in range(N) for j in range(N) if i != j], dtype=jnp.int32)
    jj = jnp.array([j for i in range(N) for j in range(N) if i != j], dtype=jnp.int32)
    return ii, jj


def _unordered_pair_indices(N: int):
    """Static unordered-pair index arrays (i, j) for i < j, shape (N*(N-1)//2,)."""
    ii = jnp.array([i for i in range(N) for j in range(i + 1, N)], dtype=jnp.int32)
    jj = jnp.array([j for i in range(N) for j in range(i + 1, N)], dtype=jnp.int32)
    return ii, jj


# ---------------------------------------------------------------------------
# Broadphase
# ---------------------------------------------------------------------------


def _broadphase(state: SimState, i, j, dt: float) -> jax.Array:
    """True if the velocity-expanded AABBs of bodies i and j overlap."""
    both_static = (state.m_inv[i] == 0) & (state.m_inv[j] == 0)

    R_i = quat_to_mat(state.q[i])
    hi = jnp.abs(R_i) @ state.half_ext[i] + _K * dt * jnp.abs(state.v[i])

    R_j = quat_to_mat(state.q[j])
    hj = jnp.abs(R_j) @ state.half_ext[j] + _K * dt * jnp.abs(state.v[j])

    overlap = jnp.all(
        (state.x[i] - hi <= state.x[j] + hj) &
        (state.x[j] - hj <= state.x[i] + hi)
    )
    return ~both_static & overlap


# ---------------------------------------------------------------------------
# SAT
# ---------------------------------------------------------------------------


def _sat_result(state: SimState, i, j) -> tuple:
    """SAT over all 15 axes.

    Returns (not_sep, min_axis, min_depth):
      not_sep  : bool   — True when no separating axis found
      min_axis : int32  — index in [0,14] of the minimum-overlap axis
                          0–2 = face normals of i, 3–5 = face normals of j,
                          6–14 = cross products indexed as ea*3+eb
      min_depth: float  — overlap at min_axis (positive = penetrating)
    """
    x_a, x_b = state.x[i], state.x[j]
    R_a = quat_to_mat(state.q[i])
    R_b = quat_to_mat(state.q[j])
    h_a = state.half_ext[i]
    h_b = state.half_ext[j]
    d = x_b - x_a

    def overlap_on(n):
        r_a = h_a @ jnp.abs(R_a.T @ n)
        r_b = h_b @ jnp.abs(R_b.T @ n)
        return r_a + r_b - jnp.abs(jnp.dot(d, n))

    face_overlaps = jax.vmap(overlap_on)(
        jnp.concatenate([R_a.T, R_b.T], axis=0)  # (6, 3)
    )

    cross_overlaps = jnp.stack([
        _edge_overlap(R_a[:, ea], R_b[:, eb], overlap_on)
        for ea in range(3) for eb in range(3)
    ])  # (9,)

    all_overlaps = jnp.concatenate([face_overlaps, cross_overlaps])  # (15,)
    not_sep = ~jnp.any(all_overlaps < 0)
    min_axis = jnp.argmin(all_overlaps).astype(jnp.int32)
    min_depth = all_overlaps[min_axis]
    return not_sep, min_axis, min_depth


def _edge_overlap(ea, eb, overlap_fn) -> jax.Array:
    cross = jnp.cross(ea, eb)
    norm = jnp.linalg.norm(cross)
    n = cross / jnp.where(norm > 1e-6, norm, 1.0)
    return jnp.where(norm > 1e-6, overlap_fn(n), jnp.inf)


# ---------------------------------------------------------------------------
# Contact point generation
# ---------------------------------------------------------------------------


def _edge_edge_contact(state: SimState, i, j, min_axis):
    """Closest-point contact between the edge pair identified by min_axis.

    min_axis ∈ [6,14] encodes ea = (min_axis-6)//3, eb = (min_axis-6)%3.
    Returns (r_a, r_b, n, d) — single contact point, not batched.
    r_a in body i frame, r_b in body j frame.
    n points from j toward i.
    """
    x_i = state.x[i];  x_j = state.x[j]
    R_i = quat_to_mat(state.q[i]);  R_j = quat_to_mat(state.q[j])
    h_i = state.half_ext[i];        h_j = state.half_ext[j]

    ea = (min_axis - 6) // 3
    eb = (min_axis - 6) % 3

    # Edge direction vectors (world frame).
    a_i = R_i[:, ea]   # column ea of R_i
    a_j = R_j[:, eb]

    # Contact normal: cross of the two edge axes, oriented from j toward i.
    cross = jnp.cross(a_i, a_j)
    norm = jnp.linalg.norm(cross)
    n_raw = cross / jnp.where(norm > 1e-6, norm, 1.0)
    n = jnp.where(jnp.dot(n_raw, x_i - x_j) >= 0, n_raw, -n_raw)

    # Select the edge of i closest to j: for each axis k≠ea, pick the half-extent
    # face facing toward j.
    d_ij = x_j - x_i
    s_i = jnp.array([
        jnp.where(ea == 0, 0.0, jnp.sign(jnp.dot(d_ij, R_i[:, 0]))),
        jnp.where(ea == 1, 0.0, jnp.sign(jnp.dot(d_ij, R_i[:, 1]))),
        jnp.where(ea == 2, 0.0, jnp.sign(jnp.dot(d_ij, R_i[:, 2]))),
    ])
    c_i = x_i + R_i @ (s_i * h_i)

    s_j = jnp.array([
        jnp.where(eb == 0, 0.0, jnp.sign(jnp.dot(-d_ij, R_j[:, 0]))),
        jnp.where(eb == 1, 0.0, jnp.sign(jnp.dot(-d_ij, R_j[:, 1]))),
        jnp.where(eb == 2, 0.0, jnp.sign(jnp.dot(-d_ij, R_j[:, 2]))),
    ])
    c_j = x_j + R_j @ (s_j * h_j)

    # Closest point on segment pair (Ericson §5.1.9).
    # Segment i: c_i ± h_i[ea]*a_i  →  half-length L_i = h_i[ea]
    # Segment j: c_j ± h_j[eb]*a_j  →  half-length L_j = h_j[eb]
    L_i = h_i[ea];  L_j = h_j[eb]
    r = c_i - c_j
    b = jnp.dot(a_i, a_j)
    f = jnp.dot(a_j, r)
    c = jnp.dot(a_i, r)
    denom = 1.0 - b * b   # = |cross|^2 / (|a_i||a_j|)^2; 0 when parallel

    t = jnp.where(denom > 1e-6, jnp.clip((b * f - c) / denom, -L_i, L_i), 0.0)
    s = jnp.clip((b * t + f) / jnp.where(denom > 1e-6, 1.0, 1.0), -L_j, L_j)
    # Re-clamp t after s is found.
    t = jnp.clip((b * s - c), -L_i, L_i)

    p_i = c_i + t * a_i
    p_j = c_j + s * a_j
    p   = 0.5 * (p_i + p_j)

    # Penetration depth from SAT overlap formula along n.
    d = (h_i @ jnp.abs(R_i.T @ n)
       + h_j @ jnp.abs(R_j.T @ n)
       - jnp.abs(jnp.dot(x_j - x_i, n)))

    r_a = R_i.T @ (p - x_i)
    r_b = R_j.T @ (p - x_j)
    return r_a, r_b, n, d


def _vertex_contacts(state: SimState, i, j):
    """Test 8 vertices of body i against box j.

    Returns (r_a, r_b, n, d) each shape (8, ...).
    r_a in body i frame, r_b in body j frame (paper §3.3: project into rest state).
    n points from body j toward body i (spec convention).
    d ≤ 0 for vertices outside box j.
    """
    x_a = state.x[i]
    x_b = state.x[j]
    R_a = quat_to_mat(state.q[i])
    R_b = quat_to_mat(state.q[j])
    h_a = state.half_ext[i]
    h_b = state.half_ext[j]

    # Contact face selection via pair-level SAT overlaps for j's 3 face normals.
    # Per-vertex argmin(depths) is degenerate when a corner vertex sits exactly on
    # a face boundary (depth=0), causing it to win over the actual penetrating face.
    # SAT pair overlaps are free of this artifact because they aggregate over the
    # whole body, not a single vertex.
    d_vec = x_b - x_a
    k_pair = jnp.argmin(jnp.stack([
        h_a @ jnp.abs(R_a.T @ R_b[:, k]) + h_b[k] - jnp.abs(jnp.dot(d_vec, R_b[:, k]))
        for k in range(3)
    ]))

    # World-frame vertices of body i: x_a + R_a @ (sign * h_a)
    verts = x_a[None, :] + (_VERTEX_SIGNS * h_a) @ R_a.T  # (8, 3)

    def contact_for_vertex(v):
        # Transform v into box j's local frame.
        v_loc = R_b.T @ (v - x_b)                   # (3,)
        inside = jnp.all(jnp.abs(v_loc) <= h_b)

        depths = h_b - jnp.abs(v_loc)               # positive inside = penetration per axis
        w = jax.nn.one_hot(k_pair, 3)                # (3,) selector for contact face

        # Outward normal of box j's closest face, pointing from j toward i.
        sign = jnp.where(v_loc[k_pair] >= 0, 1.0, -1.0)
        n = sign * (R_b @ w)                         # (3,)

        d = jnp.where(inside, jnp.dot(depths, w), -1.0)
        return R_a.T @ (v - x_a), R_b.T @ (v - x_b), n, d

    return jax.vmap(contact_for_vertex)(verts)
