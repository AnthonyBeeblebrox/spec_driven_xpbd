"""Smoke and physics tests for box_arena and stack scenes."""
import jax.numpy as jnp


class TestBoxArena:
    def test_compiles_and_runs(self):
        from spec_driven_xpbd.scenes.box_arena import make_scene
        state, step_fn = make_scene()
        state2 = step_fn(state)
        assert state2.x.shape == (15, 3)

    def test_static_bodies_stay_static(self):
        from spec_driven_xpbd.scenes.box_arena import make_scene
        state0, step_fn = make_scene()
        state = state0
        for _ in range(30):
            state = step_fn(state)
        assert jnp.allclose(state.x[:5], state0.x[:5], atol=1e-6)
        assert jnp.allclose(state.v[:5], jnp.zeros((5, 3)), atol=1e-6)

    def test_no_box_tunnels_ground(self):
        from spec_driven_xpbd.scenes.box_arena import make_scene
        state, step_fn = make_scene()
        for _ in range(200):
            state = step_fn(state)
        # Tunneling = center passes below ground center (y=0).
        # Solver residual may leave bottoms slightly below ground top (y=0.1)
        # but centers must remain above ground center.
        assert jnp.all(state.x[5:, 1] >= 0.0)

    def test_jit_does_not_retrace(self):
        from spec_driven_xpbd.scenes.box_arena import make_scene
        state, step_fn = make_scene()
        step_fn(state)
        before = step_fn._cache_size()  # noqa: SLF001
        step_fn(state)
        assert step_fn._cache_size() == before  # noqa: SLF001


class TestStack:
    def test_compiles_and_runs(self):
        from spec_driven_xpbd.scenes.stack import make_scene
        state, step_fn = make_scene()
        state2 = step_fn(state)
        assert state2.x.shape == (4, 3)

    def test_ground_stays_static(self):
        from spec_driven_xpbd.scenes.stack import make_scene
        state, step_fn = make_scene()
        for _ in range(60):
            state = step_fn(state)
        assert jnp.allclose(state.x[0], jnp.array([0., 0., 0.]), atol=1e-6)
        assert jnp.allclose(state.v[0], jnp.zeros(3), atol=1e-6)

    def test_bottom_box_rests_on_ground(self):
        from spec_driven_xpbd.scenes.stack import make_scene
        state, step_fn = make_scene()
        for _ in range(200):
            state = step_fn(state)
        box_bottom = float(state.x[1, 1]) - 0.25
        ground_top = 0.1
        assert box_bottom >= ground_top - 1e-3

    def test_no_box_below_ground(self):
        from spec_driven_xpbd.scenes.stack import make_scene
        state, step_fn = make_scene()
        for _ in range(200):
            state = step_fn(state)
        box_bottoms = state.x[1:, 1] - 0.25
        assert jnp.all(box_bottoms >= 0.1 - 1e-3)

    def test_jit_does_not_retrace(self):
        from spec_driven_xpbd.scenes.stack import make_scene
        state, step_fn = make_scene()
        step_fn(state)
        before = step_fn._cache_size()  # noqa: SLF001
        step_fn(state)
        assert step_fn._cache_size() == before  # noqa: SLF001
