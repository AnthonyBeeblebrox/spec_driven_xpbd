from functools import partial

import jax
import jax.numpy as jnp

from spec_driven_xpbd.engine.state import SimState, SimParams
from spec_driven_xpbd.engine.step import xpbd_step


def make_scene(params: SimParams | None = None) -> tuple[SimState, any]:
    """1 kg cube (0.5 m side) dropped from y=1.5 m onto a static slab (4×0.2×4 m).

    Body 0 — slab : half_ext (2, 0.1, 2), m_inv=0 (static), center y=0.
    Body 1 — cube : half_ext (0.25, 0.25, 0.25), m=1 kg, center y=1.5.
    Contact expected when cube center reaches y ≈ 0.35 (~0.43 s of free fall).

    Returns:
        (state, step_fn).
    """
    hx, hy, hz = 0.25, 0.25, 0.25
    m = 1.0
    Ix = m / 3.0 * (hy**2 + hz**2)
    Iy = m / 3.0 * (hx**2 + hz**2)
    Iz = m / 3.0 * (hx**2 + hy**2)

    state = SimState(
        x=jnp.array([
            [0.0, 0.0,  0.0],   # slab
            [0.0, 1.5,  0.0],   # cube
        ]),
        v=jnp.zeros((2, 3)),
        q=jnp.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]),
        omega=jnp.zeros((2, 3)),
        m_inv=jnp.array([0.0, 1.0 / m]),
        I_inv_body=jnp.array([
            [0.0,      0.0,      0.0     ],  # static slab
            [1.0 / Ix, 1.0 / Iy, 1.0 / Iz],
        ]),
        half_ext=jnp.array([
            [2.0,  0.1,  2.0 ],  # slab
            [0.25, 0.25, 0.25],  # cube
        ]),
    )

    if params is None:
        params = SimParams()
    step_fn = jax.jit(partial(xpbd_step, params=params))
    return state, step_fn
