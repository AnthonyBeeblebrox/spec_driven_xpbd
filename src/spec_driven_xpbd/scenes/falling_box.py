from functools import partial

import jax
import jax.numpy as jnp

from spec_driven_xpbd.engine.state import SimState, SimParams
from spec_driven_xpbd.engine.step import xpbd_step


def make_scene(params: SimParams | None = None) -> tuple[SimState, any]:
    """Single box (1 kg, 1 m³) dropped from height 3 m.

    Returns:
        (state, step_fn).
    """
    half_ext = (0.5, 0.5, 0.5)
    hx, hy, hz = half_ext
    m = 1.0

    Ix = m / 3.0 * (hy**2 + hz**2)
    Iy = m / 3.0 * (hx**2 + hz**2)
    Iz = m / 3.0 * (hx**2 + hy**2)

    state = SimState(
        x=jnp.array([[0.0, 3.0, 0.0]]),
        v=jnp.zeros((1, 3)),
        q=jnp.array([[1.0, 0.0, 0.0, 0.0]]),
        omega=jnp.zeros((1, 3)),
        m_inv=jnp.array([1.0 / m]),
        I_inv_body=jnp.array([[1.0 / Ix, 1.0 / Iy, 1.0 / Iz]]),
        half_ext=jnp.array([half_ext]),
    )

    if params is None:
        params = SimParams()
    step_fn = jax.jit(partial(xpbd_step, params=params))

    return state, step_fn
