import jax
import jax.numpy as jnp


def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Hamilton product of two quaternions [w, x, y, z]."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return jnp.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_inv(q: jax.Array) -> jax.Array:
    """Conjugate = inverse for unit quaternions."""
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def quat_to_mat(q: jax.Array) -> jax.Array:
    """Rotation matrix (3, 3) from unit quaternion [w, x, y, z]."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotate(q: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate vector v (3,) by quaternion q (4,)."""
    return quat_to_mat(q) @ v


def total_energy(state, gravity=(0.0, -9.81, 0.0)) -> tuple:
    """Returns (ke_lin, ke_rot, pe, total) summed over all dynamic bodies."""
    is_dyn = state.m_inv > 0
    m = jnp.where(is_dyn, 1.0 / jnp.where(is_dyn, state.m_inv, 1.0), 0.0)

    ke_lin = 0.5 * jnp.sum(m * jnp.sum(state.v ** 2, axis=-1))

    R = jax.vmap(quat_to_mat)(state.q)  # (N, 3, 3)
    omega_body = jnp.einsum("nji,nj->ni", R, state.omega)  # R^T @ omega
    I_bd = jnp.where(state.I_inv_body > 0, 1.0 / jnp.where(state.I_inv_body > 0, state.I_inv_body, 1.0), 0.0)
    ke_rot = 0.5 * jnp.sum(omega_body ** 2 * I_bd)

    g = jnp.array(gravity)
    pe = -jnp.sum(m * (state.x @ g))  # PE_i = -m_i * (g · x_i)

    return ke_lin, ke_rot, pe, ke_lin + ke_rot + pe


def I_world(q: jax.Array, I_inv_body: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Returns (I_world, I_inv_world) as (3,3) matrices given orientation q
    and diagonal inverse inertia I_inv_body (3,) in the principal body frame.
    """
    R = quat_to_mat(q)
    # safe inversion: where I_inv_body == 0 (static), I_body = 0 too
    safe = jnp.where(I_inv_body > 0, I_inv_body, 1.0)
    I_body = jnp.where(I_inv_body > 0, 1.0 / safe, 0.0)

    I = R @ jnp.diag(I_body) @ R.T
    I_inv = R @ jnp.diag(I_inv_body) @ R.T
    return I, I_inv
