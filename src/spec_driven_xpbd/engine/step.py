"""
XPBD rigid body simulation step — Algorithm 2, Müller et al. 2020.
"""

import dataclasses

import jax
import jax.numpy as jnp

from .collision import broadphase_pairs, narrowphase_contacts
from .joints import _solve_joints_gs, _solve_joints_jacobi
from .math import quat_mul, quat_inv, I_world, rotate
from .state import SimState, SimParams, ContactBuffer, HingeJoint, empty_hinge_joints


def xpbd_step(state: SimState, joints: HingeJoint | None = None, *, params: SimParams, exclude_collision_pairs: frozenset = frozenset()) -> SimState:
    """One full simulation frame: num_substeps × (integrate + solve).

    joints: optional HingeJoint pytree (shape (J,)).  Pass None or omit for
    scenes without joints.  Scene builders should use:
        step_fn = jax.jit(partial(xpbd_step, joints=joints, params=params))

    Broadphase (AABB) runs once per frame.  Narrowphase (SAT + vertex contacts)
    runs once per substep on the post-integrate state so that contact normals,
    positions, and penetration depths are always fresh (paper §3.5).
    """
    if joints is None:
        joints = empty_hinge_joints()

    pair_mask = broadphase_pairs(state, params.dt, exclude_collision_pairs)
    h = params.dt / params.num_substeps
    g = jnp.array(params.gravity)

    def substep_fn(_, state):
        x_prev = state.x
        q_prev = state.q
        v_prev = state.v
        omega_prev = state.omega

        state = _integrate(state, h, g)

        # Fresh contacts from current state; lambdas initialised to 0.
        contacts = narrowphase_contacts(state, pair_mask)

        # Reset joint lambdas for this substep.
        joints_sub = dataclasses.replace(
            joints,
            lambda_pos=jnp.zeros_like(joints.lambda_pos),
            lambda_ang=jnp.zeros_like(joints.lambda_ang),
            lambda_limit=jnp.zeros_like(joints.lambda_limit),
        )

        def pos_iter_fn(_, carry):
            state, contacts, js = carry
            # Joints first (higher priority), then contacts.
            if params.solver == 'gs':
                state, js = _solve_joints_gs(state, js, h)
                state, contacts = _solve_positions_gs(state, contacts, h, x_prev, q_prev, params.mu_static)
            else:
                state, js = _solve_joints_jacobi(state, js, h)
                state, contacts = _solve_positions_jacobi(state, contacts, h, x_prev, q_prev, params.mu_static)
            return state, contacts, js

        state, contacts, _ = jax.lax.fori_loop(
            0, params.num_pos_iters, pos_iter_fn, (state, contacts, joints_sub)
        )

        state = _derive_velocities(state, x_prev, q_prev, h)
        state = _solve_velocities(state, contacts, h, params, v_prev, omega_prev, q_prev)
        return state

    return jax.lax.fori_loop(0, params.num_substeps, substep_fn, state)


# ---------------------------------------------------------------------------
# Placeholder
# ---------------------------------------------------------------------------


def _solve_velocities(
    state: SimState, contacts: ContactBuffer, h: float, params: SimParams,
    v_prev: jax.Array, omega_prev: jax.Array, q_prev: jax.Array,
) -> SimState:
    """Velocity solve: dynamic friction + restitution (eq. 29–34).

    Single vectorized pass over all active contacts. No per-body averaging —
    the 1/w scaling in each correction already encodes the correct mass weighting.
    """
    N = state.x.shape[0]
    g_mag = jnp.linalg.norm(jnp.array(params.gravity))

    m_inv = jnp.concatenate([state.m_inv, jnp.zeros(1)])
    I_inv_body = jnp.concatenate([state.I_inv_body, jnp.zeros((1, 3))])
    v_ext       = jnp.concatenate([state.v,   jnp.zeros((1, 3))])
    omega_ext   = jnp.concatenate([state.omega, jnp.zeros((1, 3))])
    vp_ext      = jnp.concatenate([v_prev,    jnp.zeros((1, 3))])
    op_ext      = jnp.concatenate([omega_prev, jnp.zeros((1, 3))])

    a = contacts.body_a
    b = contacts.body_b

    def per_slot(q_a, q_b, qp_a, qp_b, v_a, v_b, omega_a, omega_b,
                 vp_a, vp_b, op_a, op_b,
                 m_inv_a, m_inv_b, I_inv_body_a, I_inv_body_b,
                 r_a, r_b, n, d, lambda_n):
        r_a_w = rotate(q_a, r_a)
        r_b_w = rotate(q_b, r_b)
        _, I_inv_a = I_world(q_a, I_inv_body_a)
        _, I_inv_b = I_world(q_b, I_inv_body_b)

        # Generalized inverse mass along n (eq. 2–3)
        rn_a = jnp.cross(r_a_w, n)
        rn_b = jnp.cross(r_b_w, n)
        w = (m_inv_a + jnp.dot(rn_a, I_inv_a @ rn_a)
           + m_inv_b + jnp.dot(rn_b, I_inv_b @ rn_b))

        # Relative velocity at contact point (eq. 29)
        v_rel = (v_a + jnp.cross(omega_a, r_a_w)) - (v_b + jnp.cross(omega_b, r_b_w))
        v_n = jnp.dot(n, v_rel)          # < 0 = approaching
        v_t = v_rel - v_n * n

        # Dynamic friction (eq. 30)
        f_n = lambda_n / (h * h)
        v_t_mag = jnp.linalg.norm(v_t)
        dv_friction = jnp.where(
            v_t_mag > 0,
            -v_t / v_t_mag * jnp.minimum(h * params.mu_dynamic * jnp.abs(f_n), v_t_mag),
            jnp.zeros(3),
        )

        # Pre-substep normal velocity for restitution target.
        # Use q_prev to rotate r to world frame — consistent with v_prev/omega_prev
        # which are also from the start of the substep.
        r_a_w_prev = rotate(qp_a, r_a)
        r_b_w_prev = rotate(qp_b, r_b)
        v_rel_prev = (vp_a + jnp.cross(op_a, r_a_w_prev)) - (vp_b + jnp.cross(op_b, r_b_w_prev))
        v_n_prev = jnp.dot(n, v_rel_prev)

        # Restitution (eq. 34, adapted for n pointing from b to a)
        # target = max(-e * v_n_prev, 0) — positive (separating) when was approaching
        v_n_target = jnp.where(
            jnp.abs(v_n) > 2.0 * g_mag * h,
            jnp.maximum(-params.restitution * v_n_prev, 0.0),
            v_n,  # below threshold → Δv_restitution = 0
        )
        dv_restitution = n * (v_n_target - v_n)

        # Combined impulse gated on active contact (eq. 33)
        dv = dv_friction + dv_restitution
        p = jnp.where(w > 0, dv / w, jnp.zeros(3))
        p = jnp.where(d > 0, p, jnp.zeros(3))

        dv_a    = m_inv_a * p
        domega_a = I_inv_a @ jnp.cross(r_a_w, p)
        dv_b    = -m_inv_b * p
        domega_b = -I_inv_b @ jnp.cross(r_b_w, p)
        return dv_a, domega_a, dv_b, domega_b

    dv_a, domega_a, dv_b, domega_b = jax.vmap(per_slot)(
        state.q[a], state.q[b],
        q_prev[a], q_prev[b],
        v_ext[a], v_ext[b],
        omega_ext[a], omega_ext[b],
        vp_ext[a], vp_ext[b],
        op_ext[a], op_ext[b],
        m_inv[a], m_inv[b],
        I_inv_body[a], I_inv_body[b],
        contacts.r_a, contacts.r_b,
        contacts.n, contacts.d, contacts.lambda_n,
    )

    delta_v = (jnp.zeros((N + 1, 3))
               .at[a].add(dv_a)
               .at[b].add(dv_b))
    delta_omega = (jnp.zeros((N + 1, 3))
                   .at[a].add(domega_a)
                   .at[b].add(domega_b))

    return dataclasses.replace(
        state,
        v=state.v + delta_v[:N],
        omega=state.omega + delta_omega[:N],
    )


# ---------------------------------------------------------------------------
# Position solvers
# ---------------------------------------------------------------------------


def _contact_correction(q_a, q_b, m_inv_a, m_inv_b, I_inv_body_a, I_inv_body_b,
                        r_a, r_b, n, d, lambda_n):
    """Normal contact correction for one slot (eq. 2–9, unilateral λ_n ≥ 0).

    Returns (dx_a, dq_a, dx_b, dq_b, lambda_n_new).
    Zero corrections when d ≤ 0 or w = 0 (both bodies static / sentinel slots).
    """
    r_a_w = rotate(q_a, r_a)
    r_b_w = rotate(q_b, r_b)
    _, I_inv_a = I_world(q_a, I_inv_body_a)
    _, I_inv_b = I_world(q_b, I_inv_body_b)

    rn_a = jnp.cross(r_a_w, n)
    rn_b = jnp.cross(r_b_w, n)
    w = (m_inv_a + jnp.dot(rn_a, I_inv_a @ rn_a)
       + m_inv_b + jnp.dot(rn_b, I_inv_b @ rn_b))

    # Δλ = d / w  (c = −d, compliance = 0 for contacts); clamp λ_n ≥ 0
    delta_lam = jnp.where(w > 0, d / w, 0.0)
    lam_new = jnp.maximum(lambda_n + delta_lam, 0.0)
    delta_lam = lam_new - lambda_n

    p = delta_lam * n

    dx_a = m_inv_a * p
    dx_b = -m_inv_b * p

    tau_a = I_inv_a @ jnp.cross(r_a_w, p)
    tau_b = I_inv_b @ jnp.cross(r_b_w, p)
    dq_a = 0.5 * quat_mul(jnp.concatenate([jnp.zeros(1), tau_a]), q_a)
    dq_b = -0.5 * quat_mul(jnp.concatenate([jnp.zeros(1), tau_b]), q_b)

    return dx_a, dq_a, dx_b, dq_b, lam_new


def _friction_correction(q_a, q_b, x_a, x_b, x_a_prev, x_b_prev, q_a_prev, q_b_prev,
                          m_inv_a, m_inv_b, I_inv_body_a, I_inv_body_b,
                          r_a, r_b, n, d, lambda_n, lambda_t, mu_s):
    """Static friction correction for one contact slot (paper §3.5, eq. 27–28).

    Resists tangential sliding by treating the tangential displacement since
    substep start as a positional constraint, clamped to the Coulomb cone.

    Returns (dx_a, dq_a, dx_b, dq_b, lambda_t_new).
    Zero corrections when d ≤ 0 (inactive slot).
    """
    r_a_w = rotate(q_a, r_a)
    r_b_w = rotate(q_b, r_b)
    _, I_inv_a = I_world(q_a, I_inv_body_a)
    _, I_inv_b = I_world(q_b, I_inv_body_b)

    p_a = x_a + r_a_w
    p_b = x_b + r_b_w
    p_a_prev = x_a_prev + rotate(q_a_prev, r_a)
    p_b_prev = x_b_prev + rotate(q_b_prev, r_b)
    delta = (p_a - p_b) - (p_a_prev - p_b_prev)
    delta_t = delta - jnp.dot(n, delta) * n
    delta_t_mag = jnp.linalg.norm(delta_t)

    n_t = jnp.where(delta_t_mag > 0.0, delta_t / delta_t_mag, jnp.zeros(3))

    rt_a = jnp.cross(r_a_w, n_t)
    rt_b = jnp.cross(r_b_w, n_t)
    w_t = (m_inv_a + jnp.dot(rt_a, I_inv_a @ rt_a)
         + m_inv_b + jnp.dot(rt_b, I_inv_b @ rt_b))

    delta_lam_t = jnp.where(w_t > 0.0, -delta_t_mag / w_t, 0.0)
    lam_t_new = lambda_t + delta_lam_t
    lam_t_new = jnp.clip(lam_t_new, -mu_s * lambda_n, mu_s * lambda_n)
    delta_lam_t = lam_t_new - lambda_t

    p_t = delta_lam_t * n_t

    active = d > 0.0
    p_t = jnp.where(active, p_t, jnp.zeros(3))
    lam_t_new = jnp.where(active, lam_t_new, lambda_t)

    dx_a = m_inv_a * p_t
    dx_b = -m_inv_b * p_t
    tau_a = I_inv_a @ jnp.cross(r_a_w, p_t)
    tau_b = I_inv_b @ jnp.cross(r_b_w, p_t)
    dq_a = 0.5 * quat_mul(jnp.concatenate([jnp.zeros(1), tau_a]), q_a)
    dq_b = -0.5 * quat_mul(jnp.concatenate([jnp.zeros(1), tau_b]), q_b)

    return dx_a, dq_a, dx_b, dq_b, lam_t_new


def _solve_positions_gs(
    state: SimState, contacts: ContactBuffer, h: float,
    x_prev: jax.Array, q_prev: jax.Array, mu_s: float,
) -> tuple[SimState, ContactBuffer]:
    """Gauss-Seidel: update state immediately after each contact slot.

    Uses jax.lax.fori_loop so the C-iteration loop compiles as a single XLA
    while-loop rather than C unrolled ops.
    """
    C = contacts.d.shape[0]
    if C == 0:
        return state, contacts

    # One extra row at index N so sentinel body indices gather zero mass/inertia.
    m_inv = jnp.concatenate([state.m_inv, jnp.zeros(1)])
    I_inv_body = jnp.concatenate([state.I_inv_body, jnp.zeros((1, 3))])
    x_prev_ext = jnp.concatenate([x_prev, jnp.zeros((1, 3))])
    q_prev_ext = jnp.concatenate([q_prev, jnp.zeros((1, 4))])

    def body_fn(k, carry):
        state, lambda_n, lambda_t = carry
        a = contacts.body_a[k]
        b = contacts.body_b[k]

        # Normal correction
        dx_a, dq_a, dx_b, dq_b, lam_n_new = _contact_correction(
            state.q[a], state.q[b],
            m_inv[a], m_inv[b],
            I_inv_body[a], I_inv_body[b],
            contacts.r_a[k], contacts.r_b[k],
            contacts.n[k], contacts.d[k], lambda_n[k],
        )
        q_a_new = state.q[a] + dq_a
        q_a_new = q_a_new / jnp.linalg.norm(q_a_new)
        q_b_new = state.q[b] + dq_b
        q_b_new = q_b_new / jnp.linalg.norm(q_b_new)
        state = dataclasses.replace(
            state,
            x=state.x.at[a].add(dx_a).at[b].add(dx_b),
            q=state.q.at[a].set(q_a_new).at[b].set(q_b_new),
        )
        lambda_n = lambda_n.at[k].set(lam_n_new)

        # Static friction correction (using post-normal state)
        dx_a_t, dq_a_t, dx_b_t, dq_b_t, lam_t_new = _friction_correction(
            state.q[a], state.q[b],
            state.x[a], state.x[b],
            x_prev_ext[a], x_prev_ext[b],
            q_prev_ext[a], q_prev_ext[b],
            m_inv[a], m_inv[b],
            I_inv_body[a], I_inv_body[b],
            contacts.r_a[k], contacts.r_b[k],
            contacts.n[k], contacts.d[k],
            lam_n_new, lambda_t[k], mu_s,
        )
        q_a_new2 = state.q[a] + dq_a_t
        q_a_new2 = q_a_new2 / jnp.linalg.norm(q_a_new2)
        q_b_new2 = state.q[b] + dq_b_t
        q_b_new2 = q_b_new2 / jnp.linalg.norm(q_b_new2)
        state = dataclasses.replace(
            state,
            x=state.x.at[a].add(dx_a_t).at[b].add(dx_b_t),
            q=state.q.at[a].set(q_a_new2).at[b].set(q_b_new2),
        )
        lambda_t = lambda_t.at[k].set(lam_t_new)

        return state, lambda_n, lambda_t

    state, lambda_n, lambda_t = jax.lax.fori_loop(
        0, C, body_fn, (state, contacts.lambda_n, contacts.lambda_t)
    )
    return state, dataclasses.replace(contacts, lambda_n=lambda_n, lambda_t=lambda_t)


def _solve_positions_jacobi(
    state: SimState, contacts: ContactBuffer, h: float,
    x_prev: jax.Array, q_prev: jax.Array, mu_s: float,
) -> tuple[SimState, ContactBuffer]:
    """Jacobi: compute all corrections simultaneously via vmap, apply once.

    No per-body averaging — the 1/w scaling in each correction already encodes
    the correct mass weighting (spec §10.2).
    """
    N = state.x.shape[0]
    C = contacts.d.shape[0]

    m_inv = jnp.concatenate([state.m_inv, jnp.zeros(1)])
    I_inv_body = jnp.concatenate([state.I_inv_body, jnp.zeros((1, 3))])
    x_prev_ext = jnp.concatenate([x_prev, jnp.zeros((1, 3))])
    q_prev_ext = jnp.concatenate([q_prev, jnp.zeros((1, 4))])

    a = contacts.body_a  # (C,)
    b = contacts.body_b  # (C,)

    # --- Normal pass ---
    dx_a, dq_a, dx_b, dq_b, lam_n_new = jax.vmap(_contact_correction)(
        state.q[a], state.q[b],
        m_inv[a], m_inv[b],
        I_inv_body[a], I_inv_body[b],
        contacts.r_a, contacts.r_b,
        contacts.n, contacts.d, contacts.lambda_n,
    )

    delta_x = jnp.zeros((N + 1, 3)).at[a].add(dx_a).at[b].add(dx_b)
    delta_q = jnp.zeros((N + 1, 4)).at[a].add(dq_a).at[b].add(dq_b)
    x_new = state.x + delta_x[:N]
    q_new = state.q + delta_q[:N]
    q_new = q_new / jnp.linalg.norm(q_new, axis=-1, keepdims=True)
    state = dataclasses.replace(state, x=x_new, q=q_new)

    # --- Friction pass (on post-normal state) ---
    dx_a_t, dq_a_t, dx_b_t, dq_b_t, lam_t_new = jax.vmap(_friction_correction)(
        state.q[a], state.q[b],
        state.x[a], state.x[b],
        x_prev_ext[a], x_prev_ext[b],
        q_prev_ext[a], q_prev_ext[b],
        m_inv[a], m_inv[b],
        I_inv_body[a], I_inv_body[b],
        contacts.r_a, contacts.r_b,
        contacts.n, contacts.d,
        lam_n_new, contacts.lambda_t,
        jnp.full(C, mu_s),
    )

    delta_x_t = jnp.zeros((N + 1, 3)).at[a].add(dx_a_t).at[b].add(dx_b_t)
    delta_q_t = jnp.zeros((N + 1, 4)).at[a].add(dq_a_t).at[b].add(dq_b_t)
    x_new = state.x + delta_x_t[:N]
    q_new = state.q + delta_q_t[:N]
    q_new = q_new / jnp.linalg.norm(q_new, axis=-1, keepdims=True)

    return (dataclasses.replace(state, x=x_new, q=q_new),
            dataclasses.replace(contacts, lambda_n=lam_n_new, lambda_t=lam_t_new))


# ---------------------------------------------------------------------------
# Integration  (explicit Euler, eq. from Algorithm 2)
# ---------------------------------------------------------------------------


def _integrate(state: SimState, h: float, gravity: jax.Array) -> SimState:
    is_dynamic = state.m_inv > 0  # (N,) — static bodies are not integrated

    # --- Linear ---
    v = state.v + h * gravity * is_dynamic[:, None]
    x = state.x + h * v

    # --- Angular (per body via vmap) ---
    def _integrate_rotation(q, omega, I_inv_body, dynamic):
        I, I_inv = I_world(q, I_inv_body)

        gyro = jnp.cross(omega, I @ omega)
        omega_new = omega + h * (I_inv @ -gyro) * dynamic

        # q += h/2 * [0, ω] * q  (pure-quaternion left-multiply)
        omega_q = jnp.array([0.0, omega_new[0], omega_new[1], omega_new[2]])
        q_new = q + (h / 2.0) * quat_mul(omega_q, q)
        q_new = q_new / jnp.linalg.norm(q_new)
        return q_new, omega_new

    q_new, omega_new = jax.vmap(_integrate_rotation)(
        state.q, state.omega, state.I_inv_body, is_dynamic
    )

    return dataclasses.replace(state, x=x, v=v, q=q_new, omega=omega_new)


# ---------------------------------------------------------------------------
# Velocity derivation  (Algorithm 2, post-position-solve)
# ---------------------------------------------------------------------------


def _derive_velocities(
    state: SimState,
    x_prev: jax.Array,
    q_prev: jax.Array,
    h: float,
) -> SimState:
    v = (state.x - x_prev) / h

    def _derive_omega(q, qp):
        dq = quat_mul(q, quat_inv(qp))
        omega = 2.0 * dq[1:] / h  # xyz part of dq
        return jnp.where(dq[0] >= 0, omega, -omega)

    omega = jax.vmap(_derive_omega)(state.q, q_prev)

    return dataclasses.replace(state, v=v, omega=omega)
