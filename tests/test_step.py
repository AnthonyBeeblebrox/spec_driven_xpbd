"""Smoke tests for xpbd_step and scene construction."""
import jax.numpy as jnp


class TestFallingBox:
    def test_compiles_and_runs(self):
        from spec_driven_xpbd.scenes.falling_box import make_scene
        state, step_fn = make_scene()
        state2 = step_fn(state)
        assert state2.x.shape == (1, 3)

    def test_box_falls_under_gravity(self):
        from spec_driven_xpbd.scenes.falling_box import make_scene
        state, step_fn = make_scene()
        for _ in range(10):
            state = step_fn(state)
        # Box should have moved downward from y=3.
        assert float(state.x[0, 1]) < 3.0

    def test_jit_does_not_retrace(self):
        # Running step_fn twice with the same-shaped input must not retrace.
        # NOTE: _cache_size() is a private JAX API with no stability guarantee.
        # There is no public replacement as of JAX 0.4.x; accept the fragility.
        from spec_driven_xpbd.scenes.falling_box import make_scene
        state, step_fn = make_scene()
        step_fn(state)
        compiled_count_before = step_fn._cache_size()  # noqa: SLF001
        step_fn(state)
        assert step_fn._cache_size() == compiled_count_before  # noqa: SLF001


class TestBoxOnSlab:
    def test_compiles_and_runs(self):
        from spec_driven_xpbd.scenes.box_on_slab import make_scene
        state, step_fn = make_scene()
        state2 = step_fn(state)
        assert state2.x.shape == (2, 3)

    def test_slab_stays_static(self):
        from spec_driven_xpbd.scenes.box_on_slab import make_scene
        state, step_fn = make_scene()
        for _ in range(30):
            state = step_fn(state)
        # Slab (body 0) must not move.
        assert jnp.allclose(state.x[0], jnp.array([0., 0., 0.]), atol=1e-6)
        assert jnp.allclose(state.v[0], jnp.array([0., 0., 0.]), atol=1e-6)

    def test_cube_falls_under_gravity(self):
        from spec_driven_xpbd.scenes.box_on_slab import make_scene
        state, step_fn = make_scene()
        for _ in range(20):
            state = step_fn(state)
        assert float(state.x[1, 1]) < 1.5

    def test_jit_does_not_retrace(self):
        # N=2 exercises the full collision path (broadphase + SAT + vertex contacts).
        # NOTE: _cache_size() is a private JAX API with no stability guarantee.
        from spec_driven_xpbd.scenes.box_on_slab import make_scene
        state, step_fn = make_scene()
        step_fn(state)
        compiled_count_before = step_fn._cache_size()  # noqa: SLF001
        step_fn(state)
        assert step_fn._cache_size() == compiled_count_before  # noqa: SLF001

    def test_cube_rests_on_slab(self):
        from spec_driven_xpbd.scenes.box_on_slab import make_scene
        state, step_fn = make_scene()
        for _ in range(200):          # ~3.3 s simulation
            state = step_fn(state)
        cube_bottom = float(state.x[1, 1]) - 0.25   # cube half_ext y
        slab_top    = float(state.x[0, 1]) + 0.1    # slab half_ext y
        assert cube_bottom >= slab_top - 1e-3        # resting, not tunnelled

    def test_cube_rests_on_slab_jacobi(self):
        from functools import partial
        import jax
        from spec_driven_xpbd.engine.state import SimState, SimParams
        from spec_driven_xpbd.engine.step import xpbd_step
        from spec_driven_xpbd.scenes.box_on_slab import make_scene
        state, _ = make_scene()
        params = SimParams(solver='jacobi')
        step_fn = jax.jit(partial(xpbd_step, params=params))
        for _ in range(200):
            state = step_fn(state)
        cube_bottom = float(state.x[1, 1]) - 0.25
        slab_top    = float(state.x[0, 1]) + 0.1
        assert cube_bottom >= slab_top - 1e-3
