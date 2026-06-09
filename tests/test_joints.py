"""Tests for HingeJoint — constraint residuals, angle limits, energy, solver parity."""

import dataclasses

import jax.numpy as jnp

_ANCHOR = jnp.array([0.0, 3.0, 0.0])   # world anchor used by the pendulum scene
_R_A = jnp.array([0.0, 0.5, 0.0])      # attachment in body_a frame (top face)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _pendulum(solver="gs", substeps=40, restitution=0.3, mu_s=0.5):
    from spec_driven_xpbd.engine.state import SimParams
    from spec_driven_xpbd.scenes.pendulum import make_scene

    return make_scene(SimParams(solver=solver, num_substeps=substeps,
                                restitution=restitution, mu_static=mu_s))


def _free_pendulum(solver="gs", substeps=40):
    """Hinge pendulum with no angle limits, no friction, no restitution.

    Conservative system: only numerical dissipation from time integration.
    """
    from functools import partial

    import jax

    from spec_driven_xpbd.engine.state import HingeJoint, SimParams
    from spec_driven_xpbd.engine.step import xpbd_step
    from spec_driven_xpbd.scenes.pendulum import make_scene

    state, _ = make_scene()
    joints = HingeJoint(
        body_a=jnp.array([0], dtype=jnp.int32),
        body_b=jnp.array([1], dtype=jnp.int32),   # N=1 sentinel = world
        r_a=jnp.array([[0.0, 0.5, 0.0]]),
        r_b=jnp.array([[0.0, 3.0, 0.0]]),
        axis_a=jnp.array([[1.0, 0.0, 0.0]]),
        axis_b=jnp.array([[1.0, 0.0, 0.0]]),
        ref_a=jnp.array([[0.0, 1.0, 0.0]]),
        ref_b=jnp.array([[0.0, 1.0, 0.0]]),
        angle_min=jnp.array([-jnp.inf]),
        angle_max=jnp.array([jnp.inf]),
        compliance=jnp.array([0.0]),
        lambda_pos=jnp.zeros(1),
        lambda_ang=jnp.zeros(1),
        lambda_limit=jnp.zeros(1),
    )
    params = SimParams(solver=solver, num_substeps=substeps,
                       mu_static=0.0, mu_dynamic=0.0, restitution=0.0)
    fn = jax.jit(partial(xpbd_step, joints=joints, params=params))
    return state, fn


def _swing_angle_deg(state) -> float:
    """Physical swing angle (degrees) around X axis from the box quaternion."""
    return float(2 * jnp.arctan2(state.q[0, 1], state.q[0, 0])) * 180.0 / 3.14159265


# ---------------------------------------------------------------------------
# Smoke / JIT
# ---------------------------------------------------------------------------


class TestHingeSmoke:
    def test_compiles_and_runs(self):
        state, fn = _pendulum()
        s2 = fn(state)
        assert s2.x.shape == (1, 3)
        assert s2.q.shape == (1, 4)

    def test_jit_does_not_retrace(self):
        state, fn = _pendulum()
        fn(state)
        before = fn._cache_size()  # noqa: SLF001
        fn(state)
        assert fn._cache_size() == before  # noqa: SLF001

    def test_existing_scenes_unaffected(self):
        """Scenes that omit the joints argument must still work (backward compat)."""
        from spec_driven_xpbd.scenes.falling_box import make_scene
        state, fn = make_scene()
        s2 = fn(state)
        # Box must have fallen from y=3.
        assert float(s2.x[0, 1]) < 3.0

    def test_free_pendulum_compiles(self):
        state, fn = _free_pendulum()
        s2 = fn(state)
        assert s2.x.shape == (1, 3)


# ---------------------------------------------------------------------------
# Constraint residuals
# ---------------------------------------------------------------------------


class TestHingeConstraintResiduals:
    def test_attachment_error_lt_1mm_gs(self):
        """After 2 s of swinging, attachment point stays within 1 mm of anchor."""
        from spec_driven_xpbd.engine.math import rotate
        state, fn = _pendulum(solver="gs")
        for _ in range(120):
            state = fn(state)
        p = state.x[0] + rotate(state.q[0], _R_A)
        err = float(jnp.linalg.norm(p - _ANCHOR))
        assert err < 1e-3

    def test_attachment_error_lt_1mm_jacobi(self):
        from spec_driven_xpbd.engine.math import rotate
        state, fn = _pendulum(solver="jacobi")
        for _ in range(120):
            state = fn(state)
        p = state.x[0] + rotate(state.q[0], _R_A)
        err = float(jnp.linalg.norm(p - _ANCHOR))
        assert err < 1e-3

    def test_hinge_pure_x_rotation(self):
        """Hinge axis = X: q_y and q_z must remain negligibly small."""
        state, fn = _pendulum()
        for _ in range(120):
            state = fn(state)
        assert float(jnp.abs(state.q[0, 2])) < 1e-4   # q_y
        assert float(jnp.abs(state.q[0, 3])) < 1e-4   # q_z

    def test_axis_alignment_after_off_axis_kick(self):
        """Off-axis angular velocity is corrected by the axis-alignment sub-constraint."""
        from spec_driven_xpbd.engine.math import rotate
        state, fn = _pendulum()
        # Give the box a kick partly outside the hinge axis.
        state = dataclasses.replace(state, omega=jnp.array([[3.0, 2.0, 0.0]]))
        for _ in range(120):
            state = fn(state)
        a1_world = rotate(state.q[0], jnp.array([1.0, 0.0, 0.0]))
        a2_world = jnp.array([1.0, 0.0, 0.0])   # fixed world axis
        alignment = float(jnp.dot(a1_world, a2_world))
        assert alignment > 0.999    # < ~2.6° misalignment


# ---------------------------------------------------------------------------
# Angle limits
# ---------------------------------------------------------------------------


class TestAngleLimits:
    def test_max_limit_not_exceeded(self):
        """omega_0 = +20 rad/s: physical angle must stay ≤ 71°."""
        state, fn = _pendulum()
        state = dataclasses.replace(state, omega=jnp.array([[20.0, 0.0, 0.0]]))
        max_deg = 0.0
        for _ in range(300):
            state = fn(state)
            max_deg = max(max_deg, abs(_swing_angle_deg(state)))
        assert max_deg <= 71.0

    def test_min_limit_not_exceeded(self):
        """omega_0 = −20 rad/s: physical angle must stay ≥ −71°."""
        state, fn = _pendulum()
        state = dataclasses.replace(state, omega=jnp.array([[-20.0, 0.0, 0.0]]))
        min_deg = 0.0
        for _ in range(300):
            state = fn(state)
            min_deg = min(min_deg, _swing_angle_deg(state))
        assert min_deg >= -71.0

    def test_limit_both_sides(self):
        """Repeated swings against both limits are correctly clamped."""
        state, fn = _pendulum()
        state = dataclasses.replace(state, omega=jnp.array([[15.0, 0.0, 0.0]]))
        for _ in range(600):   # 10 s — many bounces
            state = fn(state)
            angle = abs(_swing_angle_deg(state))
            assert angle <= 72.0, f"angle {angle:.1f}° exceeded 72° limit"

    def test_no_limits_swing_past_70deg(self):
        """Without limits, box swings beyond ±70°."""
        state, fn = _free_pendulum()
        state = dataclasses.replace(state, omega=jnp.array([[12.0, 0.0, 0.0]]))
        max_deg = 0.0
        for _ in range(180):
            state = fn(state)
            max_deg = max(max_deg, abs(_swing_angle_deg(state)))
        assert max_deg > 70.0   # would be blocked at 70° if limits were active

    def test_limit_attachment_still_valid(self):
        """Even while hitting angle limits, attachment constraint holds."""
        from spec_driven_xpbd.engine.math import rotate
        state, fn = _pendulum()
        state = dataclasses.replace(state, omega=jnp.array([[20.0, 0.0, 0.0]]))
        for _ in range(300):
            state = fn(state)
        p = state.x[0] + rotate(state.q[0], _R_A)
        err = float(jnp.linalg.norm(p - _ANCHOR))
        assert err < 2e-3   # slightly relaxed: limits add correction load


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------


class TestHingeEnergy:
    def _energy_above_equilibrium(self, state) -> float:
        """KE + PE referenced to the hanging equilibrium (y_eq = 2.5 m)."""
        from spec_driven_xpbd.engine.math import total_energy
        _, _, _, e_total = total_energy(state)
        m, g, y_eq = 1.0, 9.81, 2.5
        return float(e_total) - m * g * y_eq

    def test_energy_does_not_grow(self):
        """Conservative pendulum must not inject energy (≤ 5% over 5 s)."""
        state, fn = _free_pendulum()
        state = fn(state)          # settle v from position change
        e0 = self._energy_above_equilibrium(state)
        for _ in range(300):      # 5 s at 60 fps
            state = fn(state)
        e_end = self._energy_above_equilibrium(state)
        assert e_end <= e0 * 1.05, f"energy grew: {e_end:.4f} > {e0:.4f} * 1.05"

    def test_energy_bounded_dissipation(self):
        """Conservative pendulum must retain ≥ 75% of mechanical energy over 5 s."""
        state, fn = _free_pendulum()
        state = fn(state)
        e0 = self._energy_above_equilibrium(state)
        for _ in range(300):
            state = fn(state)
        e_end = self._energy_above_equilibrium(state)
        assert e_end >= e0 * 0.75, f"excessive dissipation: {e_end:.4f} < {e0:.4f} * 0.75"

    def test_energy_conserved_jacobi(self):
        """Same energy check for the Jacobi solver."""
        state, fn = _free_pendulum(solver="jacobi")
        state = fn(state)
        e0 = self._energy_above_equilibrium(state)
        for _ in range(300):
            state = fn(state)
        e_end = self._energy_above_equilibrium(state)
        assert e0 * 0.75 <= e_end <= e0 * 1.05

    def test_energy_per_substep_profile(self):
        """Energy decays monotonically (or nearly so) — no sudden spikes."""
        from spec_driven_xpbd.engine.math import total_energy
        state, fn = _free_pendulum(substeps=40)
        energies = []
        for _ in range(60):    # 1 s
            state = fn(state)
            _, _, _, e = total_energy(state)
            energies.append(float(e))
        # No single frame should inject more than 1% of the max seen energy.
        e_max = max(energies)
        for i in range(1, len(energies)):
            spike = energies[i] - energies[i - 1]
            assert spike < 0.01 * e_max, f"energy spike of {spike:.4f} at frame {i}"


# ---------------------------------------------------------------------------
# GS / Jacobi parity
# ---------------------------------------------------------------------------


class TestSolverParity:
    def test_gs_jacobi_attachment_both_converge(self):
        """Both solvers keep attachment error < 1 mm after 2 s."""
        from spec_driven_xpbd.engine.math import rotate
        for solver in ("gs", "jacobi"):
            state, fn = _pendulum(solver=solver)
            for _ in range(120):
                state = fn(state)
            p = state.x[0] + rotate(state.q[0], _R_A)
            err = float(jnp.linalg.norm(p - _ANCHOR))
            assert err < 1e-3, f"{solver}: attachment error {err:.2e} ≥ 1e-3"

    def test_gs_jacobi_angle_limits_both_enforced(self):
        """Both solvers enforce ±70° within 1° at omega_0=20 rad/s."""
        for solver in ("gs", "jacobi"):
            state, fn = _pendulum(solver=solver)
            state = dataclasses.replace(state, omega=jnp.array([[20.0, 0.0, 0.0]]))
            max_deg = 0.0
            for _ in range(300):
                state = fn(state)
                max_deg = max(max_deg, abs(_swing_angle_deg(state)))
            assert max_deg <= 71.0, f"{solver}: max angle {max_deg:.1f}° exceeded 71°"
