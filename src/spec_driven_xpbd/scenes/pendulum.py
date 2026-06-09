"""
Single pendulum scene: a 1 kg box hanging from a ceiling anchor via a hinge
that allows rotation around the world X axis, with ±70° angle limits.

The box starts at rest (hanging straight down) and receives an initial
angular velocity of 3 rad/s around X to make it swing and hit the limits.

World anchor is at [0, 3, 0].  Box center at [0, 2.5, 0] (top face at anchor).
"""

from functools import partial

import jax
import jax.numpy as jnp

from spec_driven_xpbd.engine.state import SimState, SimParams, HingeJoint
from spec_driven_xpbd.engine.step import xpbd_step


def make_scene(params: SimParams | None = None) -> tuple[SimState, any]:
    if params is None:
        params = SimParams(num_substeps=40)

    half = 0.5
    m = 1.0
    I_diag = m / 3.0 * (2.0 * half ** 2)  # cube: all three equal

    # Body 0 — hanging box; body 1 = sentinel (world anchor)
    state = SimState(
        x=jnp.array([[0.0, 2.5, 0.0]]),
        v=jnp.zeros((1, 3)),
        q=jnp.array([[1.0, 0.0, 0.0, 0.0]]),
        omega=jnp.array([[3.0, 0.0, 0.0]]),  # initial swing
        m_inv=jnp.array([1.0 / m]),
        I_inv_body=jnp.array([[1.0 / I_diag] * 3]),
        half_ext=jnp.array([[half, half, half]]),
    )

    N = 1  # number of bodies; sentinel = N

    angle_limit = jnp.radians(70.0)

    joints = HingeJoint(
        body_a=jnp.array([0], dtype=jnp.int32),
        body_b=jnp.array([N], dtype=jnp.int32),   # N = world sentinel
        # Attachment: top face of box [0, half, 0] → world anchor [0, 3, 0]
        r_a=jnp.array([[0.0, half, 0.0]]),
        r_b=jnp.array([[0.0, 3.0, 0.0]]),          # world-frame anchor position
        # Hinge axis: world X; in body frame = [1, 0, 0] (identity at rest)
        axis_a=jnp.array([[1.0, 0.0, 0.0]]),
        axis_b=jnp.array([[1.0, 0.0, 0.0]]),       # world-frame axis (fixed)
        # Reference perpendicular: world Y (defines "zero angle" = hanging down)
        ref_a=jnp.array([[0.0, 1.0, 0.0]]),
        ref_b=jnp.array([[0.0, 1.0, 0.0]]),        # world-frame reference
        angle_min=jnp.array([-angle_limit]),
        angle_max=jnp.array([angle_limit]),
        compliance=jnp.array([0.0]),
        lambda_pos=jnp.zeros(1),
        lambda_ang=jnp.zeros(1),
        lambda_limit=jnp.zeros(1),
    )

    step_fn = jax.jit(partial(xpbd_step, joints=joints, params=params))
    return state, step_fn
