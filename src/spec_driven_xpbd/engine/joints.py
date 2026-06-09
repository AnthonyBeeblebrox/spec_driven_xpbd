"""
Hinge joint solver — GS and Jacobi variants.

Three sub-constraints per joint, applied in order each position iteration:
  1. Positional: attach r_a (body_a frame) to r_b (body_b frame).
  2. Axis alignment: align axis_a with axis_b (eq. 20).
  3. Angle limit: LimitAngle (Algorithm 3) using ref_a / ref_b.
"""

import dataclasses

import jax
import jax.numpy as jnp

from .math import quat_mul, I_world, rotate
from .state import HingeJoint, SimState


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _rot_v_around_n(n, phi, v):
    """Rodrigues rotation of v around unit axis n by angle phi."""
    c, s = jnp.cos(phi), jnp.sin(phi)
    return c * v + s * jnp.cross(n, v) + (1.0 - c) * jnp.dot(n, v) * n


def _limit_angle_delta_q(n, n1, n2, angle_min, angle_max):
    """Algorithm 3 — compute Δq_limit for one joint.

    n    : hinge axis (unit vector, world frame)
    n1   : ref perp vector of body_a (world frame, unit)
    n2   : ref perp vector of body_b (world frame, unit)

    Returns (delta_q, theta) where delta_q = 0 when inside limits.
    """
    phi = jnp.arcsin(jnp.clip(jnp.dot(jnp.cross(n1, n2), n), -1.0, 1.0))
    phi = jnp.where(jnp.dot(n1, n2) < 0.0, jnp.pi - phi, phi)
    phi = jnp.where(phi > jnp.pi, phi - 2.0 * jnp.pi, phi)
    phi = jnp.where(phi < -jnp.pi, phi + 2.0 * jnp.pi, phi)

    outside = (phi < angle_min) | (phi > angle_max)
    phi_target = jnp.clip(phi, angle_min, angle_max)

    n1_target = _rot_v_around_n(n, phi_target, n1)
    # n2 × n1_target: correction drives n1 toward n1_target (not n1_target toward n2)
    delta_q = jnp.cross(n2, n1_target)
    theta = jnp.linalg.norm(delta_q)

    delta_q = jnp.where(outside, delta_q, jnp.zeros(3))
    theta = jnp.where(outside, theta, 0.0)
    return delta_q, theta


# ---------------------------------------------------------------------------
# Per-constraint correction primitives
# ---------------------------------------------------------------------------


def _pos_correction(q_a, q_b, x_a, x_b, r_a_body, r_b_body,
                    I_inv_a, I_inv_b, m_inv_a, m_inv_b, lam, alpha_tilde):
    """Bilateral positional correction (eq. 2–9, no λ clamping).

    Returns (dx_a, dq_a, dx_b, dq_b, lam_new).
    """
    r_a_w = rotate(q_a, r_a_body)
    r_b_w = rotate(q_b, r_b_body)
    p_a = x_a + r_a_w
    p_b = x_b + r_b_w

    delta = p_a - p_b
    dist = jnp.linalg.norm(delta)
    n = jnp.where(dist > 1e-9, delta / dist, jnp.zeros(3))

    rn_a = jnp.cross(r_a_w, n)
    rn_b = jnp.cross(r_b_w, n)
    w = (m_inv_a + jnp.dot(rn_a, I_inv_a @ rn_a)
       + m_inv_b + jnp.dot(rn_b, I_inv_b @ rn_b))

    dl = jnp.where(w + alpha_tilde > 0.0,
                   (-dist - alpha_tilde * lam) / (w + alpha_tilde), 0.0)
    lam_new = lam + dl
    p = dl * n

    dx_a = m_inv_a * p
    dx_b = -m_inv_b * p
    tau_a = I_inv_a @ jnp.cross(r_a_w, p)
    tau_b = I_inv_b @ jnp.cross(r_b_w, p)
    dq_a = 0.5 * quat_mul(jnp.concatenate([jnp.zeros(1), tau_a]), q_a)
    dq_b = -0.5 * quat_mul(jnp.concatenate([jnp.zeros(1), tau_b]), q_b)
    return dx_a, dq_a, dx_b, dq_b, lam_new


def _ang_correction(q_a, q_b, I_inv_a, I_inv_b, delta_q_vec, theta, lam, alpha_tilde):
    """Angular constraint correction (eq. 11–16).

    delta_q_vec : angular error vector (world frame); magnitude = theta
    Returns (dq_a, dq_b, lam_new).
    """
    n = jnp.where(theta > 1e-9, delta_q_vec / theta, jnp.zeros(3))
    w_a = jnp.dot(n, I_inv_a @ n)
    w_b = jnp.dot(n, I_inv_b @ n)

    dl = jnp.where(w_a + w_b + alpha_tilde > 0.0,
                   (-theta - alpha_tilde * lam) / (w_a + w_b + alpha_tilde), 0.0)
    lam_new = lam + dl
    p = dl * n

    tau_a = I_inv_a @ p
    tau_b = I_inv_b @ p
    dq_a = 0.5 * quat_mul(jnp.concatenate([jnp.zeros(1), tau_a]), q_a)
    dq_b = -0.5 * quat_mul(jnp.concatenate([jnp.zeros(1), tau_b]), q_b)
    return dq_a, dq_b, lam_new


# ---------------------------------------------------------------------------
# State-update helpers (handle sentinel body_b = N safely)
# ---------------------------------------------------------------------------


def _apply_pos(state, a, b, dx_a, dq_a, dx_b, dq_b):
    """Apply positional + rotational correction to state.

    When b = N (sentinel), dx_b = 0 and dq_b = 0 (from zero inertia/mass),
    so the .at[b] operations are no-ops even though JAX clamps the index.
    """
    q_a_new = state.q[a] + dq_a
    q_a_new = q_a_new / jnp.linalg.norm(q_a_new)
    # For sentinel b=N: state.q[b] reads state.q[N-1] (clamped), dq_b=0,
    # so q_b_new = state.q[N-1] and the write is a no-op.
    q_b_new = state.q[b] + dq_b
    q_b_new = q_b_new / jnp.linalg.norm(q_b_new)
    return dataclasses.replace(
        state,
        x=state.x.at[a].add(dx_a).at[b].add(dx_b),
        q=state.q.at[a].set(q_a_new).at[b].set(q_b_new),
    )


def _apply_ang(state, a, b, dq_a, dq_b):
    """Apply rotational correction only."""
    q_a_new = state.q[a] + dq_a
    q_a_new = q_a_new / jnp.linalg.norm(q_a_new)
    q_b_new = state.q[b] + dq_b
    q_b_new = q_b_new / jnp.linalg.norm(q_b_new)
    return dataclasses.replace(
        state,
        q=state.q.at[a].set(q_a_new).at[b].set(q_b_new),
    )


def _q_ext(state):
    """state.q extended with identity quaternion at index N (for sentinel)."""
    return jnp.concatenate([state.q, jnp.array([[1.0, 0.0, 0.0, 0.0]])])


def _x_ext(state):
    return jnp.concatenate([state.x, jnp.zeros((1, 3))])


# ---------------------------------------------------------------------------
# Gauss-Seidel joint solver
# ---------------------------------------------------------------------------


def _solve_joints_gs(state: SimState, joints: HingeJoint, h: float):
    J = joints.body_a.shape[0]
    if J == 0:
        return state, joints

    m_inv_ext = jnp.concatenate([state.m_inv, jnp.zeros(1)])
    I_inv_body_ext = jnp.concatenate([state.I_inv_body, jnp.zeros((1, 3))])

    def body_fn(j, carry):
        state, lp, la, ll = carry
        a = joints.body_a[j]
        b = joints.body_b[j]
        alpha_tilde = joints.compliance[j] / (h * h)

        qe = _q_ext(state)
        xe = _x_ext(state)
        q_a, q_b = qe[a], qe[b]
        x_a, x_b = xe[a], xe[b]
        m_inv_a = m_inv_ext[a]
        m_inv_b = m_inv_ext[b]
        _, I_inv_a = I_world(q_a, I_inv_body_ext[a])
        _, I_inv_b = I_world(q_b, I_inv_body_ext[b])

        # --- Sub-constraint 1: positional ---
        dx_a, dq_a, dx_b, dq_b, lp_new = _pos_correction(
            q_a, q_b, x_a, x_b,
            joints.r_a[j], joints.r_b[j],
            I_inv_a, I_inv_b, m_inv_a, m_inv_b,
            lp[j], alpha_tilde,
        )
        state = _apply_pos(state, a, b, dx_a, dq_a, dx_b, dq_b)

        # Re-read updated orientations for sub-constraint 2
        qe = _q_ext(state)
        q_a, q_b = qe[a], qe[b]
        _, I_inv_a = I_world(q_a, I_inv_body_ext[a])
        _, I_inv_b = I_world(q_b, I_inv_body_ext[b])

        # --- Sub-constraint 2: axis alignment (eq. 20) ---
        a1 = rotate(q_a, joints.axis_a[j])
        a2 = rotate(q_b, joints.axis_b[j])
        delta_q_ang = jnp.cross(a2, a1)
        theta_ang = jnp.linalg.norm(delta_q_ang)
        dq_a, dq_b, la_new = _ang_correction(
            q_a, q_b, I_inv_a, I_inv_b,
            delta_q_ang, theta_ang, la[j], alpha_tilde,
        )
        state = _apply_ang(state, a, b, dq_a, dq_b)

        # Re-read updated orientations for sub-constraint 3
        qe = _q_ext(state)
        q_a, q_b = qe[a], qe[b]
        _, I_inv_a = I_world(q_a, I_inv_body_ext[a])
        _, I_inv_b = I_world(q_b, I_inv_body_ext[b])

        # --- Sub-constraint 3: angle limit (Algorithm 3) ---
        n_hinge = rotate(q_a, joints.axis_a[j])
        b1 = rotate(q_a, joints.ref_a[j])
        b2 = rotate(q_b, joints.ref_b[j])
        delta_q_lim, theta_lim = _limit_angle_delta_q(
            n_hinge, b1, b2, joints.angle_min[j], joints.angle_max[j]
        )
        dq_a, dq_b, ll_new = _ang_correction(
            q_a, q_b, I_inv_a, I_inv_b,
            delta_q_lim, theta_lim, ll[j], alpha_tilde,
        )
        state = _apply_ang(state, a, b, dq_a, dq_b)

        return (state,
                lp.at[j].set(lp_new),
                la.at[j].set(la_new),
                ll.at[j].set(ll_new))

    state, lp, la, ll = jax.lax.fori_loop(
        0, J, body_fn,
        (state, joints.lambda_pos, joints.lambda_ang, joints.lambda_limit),
    )
    return state, dataclasses.replace(
        joints, lambda_pos=lp, lambda_ang=la, lambda_limit=ll
    )


# ---------------------------------------------------------------------------
# Jacobi joint solver — three sequential vmapped passes
# ---------------------------------------------------------------------------


def _solve_joints_jacobi(state: SimState, joints: HingeJoint, h: float):
    J = joints.body_a.shape[0]
    if J == 0:
        return state, joints

    N = state.x.shape[0]
    m_inv_ext = jnp.concatenate([state.m_inv, jnp.zeros(1)])
    I_inv_body_ext = jnp.concatenate([state.I_inv_body, jnp.zeros((1, 3))])
    a = joints.body_a
    b = joints.body_b
    alpha_tilde_vec = joints.compliance / (h * h)

    # Bodies shared by multiple joints receive summed corrections; divide by
    # the number of constraints that touch each body to avoid overcorrection.
    count = jnp.maximum(
        jnp.zeros(N + 1).at[a].add(1).at[b].add(1), 1.0
    )

    # --- Pass 1: positional ---
    qe = _q_ext(state)
    xe = _x_ext(state)

    def pos_one(q_a, q_b, x_a, x_b, I_inv_body_a, I_inv_body_b, m_inv_a, m_inv_b,
                r_a, r_b, lam, at):
        _, I_inv_a = I_world(q_a, I_inv_body_a)
        _, I_inv_b = I_world(q_b, I_inv_body_b)
        return _pos_correction(q_a, q_b, x_a, x_b, r_a, r_b,
                               I_inv_a, I_inv_b, m_inv_a, m_inv_b, lam, at)

    dx_a, dq_a, dx_b, dq_b, lp_new = jax.vmap(pos_one)(
        qe[a], qe[b], xe[a], xe[b],
        I_inv_body_ext[a], I_inv_body_ext[b],
        m_inv_ext[a], m_inv_ext[b],
        joints.r_a, joints.r_b,
        joints.lambda_pos, alpha_tilde_vec,
    )
    delta_x = (jnp.zeros((N + 1, 3)).at[a].add(dx_a).at[b].add(dx_b)) / count[:, None]
    delta_q = (jnp.zeros((N + 1, 4)).at[a].add(dq_a).at[b].add(dq_b)) / count[:, None]
    x_new = state.x + delta_x[:N]
    q_new = state.q + delta_q[:N]
    q_new = q_new / jnp.linalg.norm(q_new, axis=-1, keepdims=True)
    state = dataclasses.replace(state, x=x_new, q=q_new)

    # --- Pass 2: axis alignment ---
    qe = _q_ext(state)

    def ang_one(q_a, q_b, I_inv_body_a, I_inv_body_b, axis_a, axis_b, lam, at):
        _, I_inv_a = I_world(q_a, I_inv_body_a)
        _, I_inv_b = I_world(q_b, I_inv_body_b)
        a1 = rotate(q_a, axis_a)
        a2 = rotate(q_b, axis_b)
        delta_q_vec = jnp.cross(a2, a1)
        theta = jnp.linalg.norm(delta_q_vec)
        dq_a, dq_b, lam_new = _ang_correction(q_a, q_b, I_inv_a, I_inv_b,
                                               delta_q_vec, theta, lam, at)
        return dq_a, dq_b, lam_new

    dq_a, dq_b, la_new = jax.vmap(ang_one)(
        qe[a], qe[b],
        I_inv_body_ext[a], I_inv_body_ext[b],
        joints.axis_a, joints.axis_b,
        joints.lambda_ang, alpha_tilde_vec,
    )
    delta_q = (jnp.zeros((N + 1, 4)).at[a].add(dq_a).at[b].add(dq_b)) / count[:, None]
    q_new = state.q + delta_q[:N]
    q_new = q_new / jnp.linalg.norm(q_new, axis=-1, keepdims=True)
    state = dataclasses.replace(state, q=q_new)

    # --- Pass 3: angle limit ---
    qe = _q_ext(state)

    def lim_one(q_a, q_b, I_inv_body_a, I_inv_body_b, axis_a, ref_a, ref_b,
                angle_min, angle_max, lam, at):
        _, I_inv_a = I_world(q_a, I_inv_body_a)
        _, I_inv_b = I_world(q_b, I_inv_body_b)
        n_hinge = rotate(q_a, axis_a)
        b1 = rotate(q_a, ref_a)
        b2 = rotate(q_b, ref_b)
        delta_q_vec, theta = _limit_angle_delta_q(n_hinge, b1, b2, angle_min, angle_max)
        dq_a, dq_b, lam_new = _ang_correction(q_a, q_b, I_inv_a, I_inv_b,
                                               delta_q_vec, theta, lam, at)
        return dq_a, dq_b, lam_new

    dq_a, dq_b, ll_new = jax.vmap(lim_one)(
        qe[a], qe[b],
        I_inv_body_ext[a], I_inv_body_ext[b],
        joints.axis_a, joints.ref_a, joints.ref_b,
        joints.angle_min, joints.angle_max,
        joints.lambda_limit, alpha_tilde_vec,
    )
    delta_q = (jnp.zeros((N + 1, 4)).at[a].add(dq_a).at[b].add(dq_b)) / count[:, None]
    q_new = state.q + delta_q[:N]
    q_new = q_new / jnp.linalg.norm(q_new, axis=-1, keepdims=True)
    state = dataclasses.replace(state, q=q_new)

    return state, dataclasses.replace(
        joints, lambda_pos=lp_new, lambda_ang=la_new, lambda_limit=ll_new
    )
