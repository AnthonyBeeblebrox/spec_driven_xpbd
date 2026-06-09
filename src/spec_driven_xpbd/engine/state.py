from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class ContactBuffer:
    """Fixed-size padded contact buffer — shape (C,) for all arrays.

    C = 8*N*(N-1) is computed once from the body count N and never changes,
    satisfying JAX JIT's static-shape requirement.  Inactive slots have d ≤ 0;
    solvers gate every correction on (d > 0) so inactive slots contribute zero
    without branching.

    lambda_n / lambda_t are reset to zero at the start of each substep by
    _init_lambdas; the geometry fields (body_a … d) are reused across substeps.
    """

    body_a: jax.Array    # (C,) int32 — index into SimState
    body_b: jax.Array    # (C,) int32 — index into SimState
    r_a: jax.Array       # (C, 3)  contact point relative to body_a CoM (body_a frame)
    r_b: jax.Array       # (C, 3)  contact point relative to body_b CoM (body_b frame)
    n: jax.Array         # (C, 3)  contact normal pointing from b toward a
    d: jax.Array         # (C,)    penetration depth; ≤ 0 → inactive
    lambda_n: jax.Array  # (C,)    normal Lagrange multiplier (reset each substep)
    lambda_t: jax.Array  # (C,)    tangential Lagrange multiplier (reset each substep)


jax.tree_util.register_dataclass(
    ContactBuffer,
    data_fields=["body_a", "body_b", "r_a", "r_b", "n", "d", "lambda_n", "lambda_t"],
    meta_fields=[],
)


def empty_contact_buffer(max_contacts: int, n_bodies: int, dtype=jnp.float32) -> ContactBuffer:
    """All-inactive ContactBuffer of fixed size max_contacts.

    Inactive slots use body_a = body_b = n_bodies as a sentinel (one past the
    last valid index).  Solvers must pad m_inv and I_inv_body with a zero row
    at index n_bodies so that gathers on sentinel slots yield zero contribution.
    """
    C = max_contacts
    return ContactBuffer(
        body_a=jnp.full(C, n_bodies, dtype=jnp.int32),
        body_b=jnp.full(C, n_bodies, dtype=jnp.int32),
        r_a=jnp.zeros((C, 3), dtype=dtype),
        r_b=jnp.zeros((C, 3), dtype=dtype),
        n=jnp.zeros((C, 3), dtype=dtype),
        d=jnp.full(C, -1.0, dtype=dtype),
        lambda_n=jnp.zeros(C, dtype=dtype),
        lambda_t=jnp.zeros(C, dtype=dtype),
    )


@dataclass
class SimState:
    x: jax.Array  # (N, 3)  center-of-mass positions
    v: jax.Array  # (N, 3)  linear velocities
    q: jax.Array  # (N, 4)  orientations [w, x, y, z], body→world
    omega: jax.Array  # (N, 3)  angular velocities (world frame)
    m_inv: jax.Array  # (N,)    inverse mass; 0 = static body
    I_inv_body: jax.Array  # (N, 3)  diagonal of inverse inertia (principal/body frame)
    half_ext: jax.Array  # (N, 3)  box half-extents


jax.tree_util.register_dataclass(
    SimState,
    data_fields=["x", "v", "q", "omega", "m_inv", "I_inv_body", "half_ext"],
    meta_fields=[],
)


@dataclass
class SimParams:
    dt: float = 1.0 / 60.0
    num_substeps: int = 20
    num_pos_iters: int = 1
    gravity: tuple = (0.0, -9.81, 0.0)
    mu_static: float = 0.5
    mu_dynamic: float = 0.3
    restitution: float = 0.3
    solver: str = 'gs'  # 'gs' (Gauss-Seidel) or 'jacobi'


@dataclass
class HingeJoint:
    """Batched hinge joints — all arrays shape (J,) or (J, 3).

    body_b = N (= number of bodies) is the world-anchor sentinel.
    The solver extends state arrays with a zero row at index N and
    an identity quaternion, so sentinel slots yield zero correction.

    When body_b = N, r_b / axis_b / ref_b are given in world frame.
    """
    body_a: jax.Array      # (J,) int32
    body_b: jax.Array      # (J,) int32; N = world sentinel
    r_a: jax.Array         # (J, 3) attachment in body_a frame
    r_b: jax.Array         # (J, 3) attachment in body_b frame (world if body_b=N)
    axis_a: jax.Array      # (J, 3) hinge axis in body_a frame
    axis_b: jax.Array      # (J, 3) hinge axis in body_b frame (world if body_b=N)
    ref_a: jax.Array       # (J, 3) reference perp vector in body_a frame (for LimitAngle)
    ref_b: jax.Array       # (J, 3) reference perp vector in body_b frame (world if body_b=N)
    angle_min: jax.Array   # (J,) lower limit (rad); use -jnp.inf for no limit
    angle_max: jax.Array   # (J,) upper limit (rad); use  jnp.inf for no limit
    compliance: jax.Array  # (J,) α in m/N; 0 = rigid
    lambda_pos: jax.Array  # (J,) positional Lagrange multiplier (reset each substep)
    lambda_ang: jax.Array  # (J,) axis-alignment Lagrange multiplier (reset each substep)
    lambda_limit: jax.Array  # (J,) angle-limit Lagrange multiplier (reset each substep)


jax.tree_util.register_dataclass(
    HingeJoint,
    data_fields=[
        "body_a", "body_b",
        "r_a", "r_b",
        "axis_a", "axis_b",
        "ref_a", "ref_b",
        "angle_min", "angle_max",
        "compliance",
        "lambda_pos", "lambda_ang", "lambda_limit",
    ],
    meta_fields=[],
)


def empty_hinge_joints(dtype=jnp.float32) -> HingeJoint:
    """Zero-joint HingeJoint for scenes without joints."""
    return HingeJoint(
        body_a=jnp.zeros(0, dtype=jnp.int32),
        body_b=jnp.zeros(0, dtype=jnp.int32),
        r_a=jnp.zeros((0, 3), dtype=dtype),
        r_b=jnp.zeros((0, 3), dtype=dtype),
        axis_a=jnp.zeros((0, 3), dtype=dtype),
        axis_b=jnp.zeros((0, 3), dtype=dtype),
        ref_a=jnp.zeros((0, 3), dtype=dtype),
        ref_b=jnp.zeros((0, 3), dtype=dtype),
        angle_min=jnp.zeros(0, dtype=dtype),
        angle_max=jnp.zeros(0, dtype=dtype),
        compliance=jnp.zeros(0, dtype=dtype),
        lambda_pos=jnp.zeros(0, dtype=dtype),
        lambda_ang=jnp.zeros(0, dtype=dtype),
        lambda_limit=jnp.zeros(0, dtype=dtype),
    )
