"""
Double pendulum: two slender rods connected by hinge joints, both swinging around X.

World anchor at [0, 4, 0].
Rod 1 center at [0, 3.5, 0] (top at anchor), rod 2 center at [0, 2.5, 0] (top at bottom of rod 1).
No angle limits — free rotation produces chaotic motion.
"""

from functools import partial

import jax
import jax.numpy as jnp

from spec_driven_xpbd.engine.state import SimState, SimParams, HingeJoint
from spec_driven_xpbd.engine.step import xpbd_step


def make_scene(params: SimParams | None = None) -> tuple[SimState, any]:
    if params is None:
        params = SimParams(num_substeps=30)

    hx, hy, hz = 0.05, 0.5, 0.05   # slender rod: 1 m long, 0.1 m square cross-section
    m = 1.0
    Ix = m / 3.0 * (hy**2 + hz**2)
    Iy = m / 3.0 * (hx**2 + hz**2)
    Iz = m / 3.0 * (hx**2 + hy**2)
    I_inv = jnp.array([1.0 / Ix, 1.0 / Iy, 1.0 / Iz])

    # Rod 1 starts horizontal (90° around X): local Y → world +Z.
    # R_x(90°) maps Y→Z, so the rod extends in +Z; gravity torque is along -X (hinge axis) ✓
    # q1 = [cos45, sin45, 0, 0]; rotate(q1,[0,hy,0]) = [0,0,hy]
    # center = anchor[0,4,0] - [0,0,hy] = [0,4,-hy]
    # bottom of rod 1 in world = [0,4,-2*hy]; rod 2 hangs from there.
    s = float(jnp.sqrt(0.5))
    state = SimState(
        x=jnp.array([
            [0.0,  4.0,  -hy],         # rod 1 center (horizontal in Z)
            [0.0,  4.0 - hy, -2 * hy], # rod 2 center (hanging from rod 1 tip)
        ]),
        v=jnp.zeros((2, 3)),
        q=jnp.array([
            [s, s, 0.0, 0.0],      # 90° around X — rod 1 horizontal in Z
            [1.0, 0.0, 0.0, 0.0],  # identity — rod 2 hangs down
        ]),
        omega=jnp.zeros((2, 3)),
        m_inv=jnp.array([1.0 / m, 1.0 / m]),
        I_inv_body=jnp.stack([I_inv, I_inv]),
        half_ext=jnp.array([
            [hx, hy, hz],
            [hx, hy, hz],
        ]),
    )

    N = 2  # sentinel index

    joints = HingeJoint(
        body_a=jnp.array([0, 1], dtype=jnp.int32),
        body_b=jnp.array([N, 0], dtype=jnp.int32),   # rod1→world, rod2→rod1
        r_a=jnp.array([
            [0.0,  hy, 0.0],   # top of rod 1
            [0.0,  hy, 0.0],   # top of rod 2
        ]),
        r_b=jnp.array([
            [0.0, 4.0, 0.0],   # world anchor position (world frame)
            [0.0, -hy, 0.0],   # bottom of rod 1 (rod 1 local frame)
        ]),
        # Both joints rotate around X: classic planar double pendulum swinging in YZ.
        axis_a=jnp.array([
            [1.0, 0.0, 0.0],   # rod 1 local X
            [1.0, 0.0, 0.0],   # rod 2 local X
        ]),
        axis_b=jnp.array([
            [1.0, 0.0, 0.0],   # world X (fixed)
            [1.0, 0.0, 0.0],   # rod 1 local X
        ]),
        ref_a=jnp.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]),
        ref_b=jnp.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]),
        angle_min=jnp.array([-jnp.inf, -jnp.inf]),
        angle_max=jnp.array([ jnp.inf,  jnp.inf]),
        compliance=jnp.zeros(2),
        lambda_pos=jnp.zeros(2),
        lambda_ang=jnp.zeros(2),
        lambda_limit=jnp.zeros(2),
    )

    # Bodies 0 and 1 are connected by a hinge — exclude them from collision detection.
    no_collide = frozenset([(0, 1), (1, 0)])
    step_fn = jax.jit(partial(xpbd_step, joints=joints, params=params, exclude_collision_pairs=no_collide))
    return state, step_fn
