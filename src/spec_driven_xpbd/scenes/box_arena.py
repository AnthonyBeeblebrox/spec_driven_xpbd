from functools import partial

import jax
import jax.numpy as jnp

from spec_driven_xpbd.engine.state import SimState, SimParams
from spec_driven_xpbd.engine.step import xpbd_step


def make_scene(params: SimParams | None = None, seed: int = 42, velocity: float = 3.0) -> tuple[SimState, any]:
    """Ground slab + 4 walls + 10 randomly placed dynamic cubes.

    Arena: 4×4 m floor, 1 m tall walls.
    Boxes: half_ext (0.25, 0.25, 0.25), m=1 kg, random positions/orientations/velocities.

    Bodies 0–4  : ground + 4 walls (static)
    Bodies 5–14 : 10 cubes (dynamic)
    """
    hx, hy, hz = 0.25, 0.25, 0.25
    m = 1.0
    Ix = m / 3.0 * (hy**2 + hz**2)
    Iy = m / 3.0 * (hx**2 + hz**2)
    Iz = m / 3.0 * (hx**2 + hy**2)

    N_static = 5
    N_boxes = 10

    # Walls: half_ext_y=1.5 → top at 3.1 m, above max box height of 2.25 m.
    x_static = jnp.array([
        [ 0.0,  0.0,  0.0],  # ground
        [ 0.0,  1.6,  2.1],  # north wall
        [ 0.0,  1.6, -2.1],  # south wall
        [ 2.1,  1.6,  0.0],  # east wall
        [-2.1,  1.6,  0.0],  # west wall
    ])
    half_ext_static = jnp.array([
        [2.0, 0.1, 2.0],  # ground
        [2.0, 1.5, 0.1],  # north wall
        [2.0, 1.5, 0.1],  # south wall
        [0.1, 1.5, 2.0],  # east wall
        [0.1, 1.5, 2.0],  # west wall
    ])

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)

    # Place boxes on a shuffled 4×3 grid (1.0 m spacing).
    # This guarantees Euclidean separation ≥ 1.0 m > 2×circumradius (≈ 0.87 m),
    # preventing SAT overlap at any initial orientation and avoiding the
    # velocity explosion caused by v = Δx/h amplifying large position corrections.
    _gx = jnp.tile(jnp.array([-1.5, -0.5, 0.5, 1.5]), 3)   # (12,) x-slots
    _gz = jnp.repeat(jnp.array([-1.0, 0.0, 1.0]), 4)         # (12,) z-slots
    _idx = jax.random.permutation(k1, 12)[:N_boxes]
    y  = jax.random.uniform(k2, (N_boxes,), minval=0.5, maxval=1.5)
    x_boxes = jnp.stack([_gx[_idx], y, _gz[_idx]], axis=1)

    q_raw   = jax.random.normal(k3, (N_boxes, 4))
    q_boxes = q_raw / jnp.linalg.norm(q_raw, axis=-1, keepdims=True)

    k4, k5 = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), 99))
    vxz   = jax.random.uniform(k4, (N_boxes, 2), minval=-velocity, maxval=velocity)
    v_boxes = jnp.stack([vxz[:, 0], jnp.zeros(N_boxes), vxz[:, 1]], axis=1)
    omega_boxes = jax.random.uniform(k5, (N_boxes, 3), minval=-velocity * 5.0 / 3.0, maxval=velocity * 5.0 / 3.0)

    N = N_static + N_boxes
    x       = jnp.concatenate([x_static, x_boxes])
    q       = jnp.concatenate([jnp.tile(jnp.array([1., 0., 0., 0.]), (N_static, 1)), q_boxes])
    v       = jnp.concatenate([jnp.zeros((N_static, 3)), v_boxes])
    omega   = jnp.concatenate([jnp.zeros((N_static, 3)), omega_boxes])
    m_inv   = jnp.concatenate([jnp.zeros(N_static), jnp.full(N_boxes, 1.0 / m)])
    I_inv_body = jnp.concatenate([
        jnp.zeros((N_static, 3)),
        jnp.tile(jnp.array([1.0/Ix, 1.0/Iy, 1.0/Iz]), (N_boxes, 1)),
    ])
    half_ext = jnp.concatenate([
        half_ext_static,
        jnp.tile(jnp.array([hx, hy, hz]), (N_boxes, 1)),
    ])

    state = SimState(x=x, v=v, q=q, omega=omega,
                     m_inv=m_inv, I_inv_body=I_inv_body, half_ext=half_ext)

    if params is None:
        params = SimParams()
    step_fn = jax.jit(partial(xpbd_step, params=params))
    return state, step_fn
