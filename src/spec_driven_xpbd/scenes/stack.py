from functools import partial

import jax
import jax.numpy as jnp

from spec_driven_xpbd.engine.state import SimState, SimParams
from spec_driven_xpbd.engine.step import xpbd_step


def make_scene(params: SimParams | None = None) -> tuple[SimState, any]:
    """Three cubes stacked at rest with a 0.05 m horizontal offset per level.

    The offset makes the stack unstable, stressing the contact solver.

    Body 0 : ground slab (static)
    Bodies 1-3 : cubes, bottom → top, offset +0.05 m in x each level
    """
    hx, hy, hz = 0.25, 0.25, 0.25
    m = 1.0
    Ix = m / 3.0 * (hy**2 + hz**2)
    Iy = m / 3.0 * (hx**2 + hz**2)
    Iz = m / 3.0 * (hx**2 + hy**2)

    # ground top at y=0.1; each cube is 0.5 m tall
    y0 = 0.1 + hy           # 0.35 — resting on ground
    y1 = y0 + 2 * hy        # 0.85
    y2 = y1 + 2 * hy        # 1.35

    state = SimState(
        x=jnp.array([
            [0.00, 0.00, 0.0],  # ground
            [0.00,   y0, 0.0],  # bottom cube
            [0.05,   y1, 0.0],  # middle cube
            [0.10,   y2, 0.0],  # top cube
        ]),
        v=jnp.zeros((4, 3)),
        q=jnp.tile(jnp.array([1., 0., 0., 0.]), (4, 1)),
        omega=jnp.zeros((4, 3)),
        m_inv=jnp.array([0.0, 1.0/m, 1.0/m, 1.0/m]),
        I_inv_body=jnp.array([
            [0.0,      0.0,      0.0     ],
            [1.0/Ix, 1.0/Iy, 1.0/Iz],
            [1.0/Ix, 1.0/Iy, 1.0/Iz],
            [1.0/Ix, 1.0/Iy, 1.0/Iz],
        ]),
        half_ext=jnp.array([
            [2.0, 0.1, 2.0],
            [ hx,  hy,  hz],
            [ hx,  hy,  hz],
            [ hx,  hy,  hz],
        ]),
    )

    if params is None:
        params = SimParams()
    step_fn = jax.jit(partial(xpbd_step, params=params))
    return state, step_fn
