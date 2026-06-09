import typer

from spec_driven_xpbd.engine.state import SimParams

app = typer.Typer(help="xpbd rigid body simulator")


def _params(
    solver: str,
    substeps: int,
    restitution: float,
) -> SimParams:
    return SimParams(solver=solver, num_substeps=substeps, restitution=restitution)


@app.command()
def falling_box(
    solver: str = typer.Option("jacobi", help="Position solver: 'gs' or 'jacobi'"),
    substeps: int = typer.Option(10, help="Substeps per frame"),
    restitution: float = typer.Option(0.3, help="Coefficient of restitution"),
) -> None:
    """Single box falling under gravity."""
    from spec_driven_xpbd.scenes.falling_box import make_scene
    from spec_driven_xpbd.viewer import run

    state, step_fn = make_scene(params=_params(solver, substeps, restitution))
    run(state, step_fn)


@app.command()
def box_on_slab(
    solver: str = typer.Option("jacobi", help="Position solver: 'gs' or 'jacobi'"),
    substeps: int = typer.Option(10, help="Substeps per frame"),
    restitution: float = typer.Option(0.3, help="Coefficient of restitution"),
) -> None:
    """Cube dropped onto a static rectangular slab."""
    from spec_driven_xpbd.scenes.box_on_slab import make_scene
    from spec_driven_xpbd.viewer import run

    state, step_fn = make_scene(params=_params(solver, substeps, restitution))
    run(state, step_fn)


@app.command()
def box_arena(
    solver: str = typer.Option("jacobi", help="Position solver: 'gs' or 'jacobi'"),
    substeps: int = typer.Option(10, help="Substeps per frame"),
    restitution: float = typer.Option(0.3, help="Coefficient of restitution"),
    seed: int = typer.Option(42, help="RNG seed for box placement"),
    velocity: float = typer.Option(3.0, help="Max initial linear velocity (m/s)"),
    save_gif: str = typer.Option(None, help="Save a GIF recording to this path"),
    gif_duration: float = typer.Option(5.0, help="Duration of GIF in seconds"),
) -> None:
    """10 random cubes dropped into a walled arena."""
    from spec_driven_xpbd.scenes.box_arena import make_scene
    from spec_driven_xpbd.viewer import run
    from spec_driven_xpbd.engine.math import total_energy

    def _stats(s):
        ke_lin, ke_rot, pe, total = total_energy(s, _params(solver, substeps, restitution).gravity)
        return f"E={float(total):.1f}  KE={float(ke_lin + ke_rot):.1f}  PE={float(pe):.1f}"

    state, step_fn = make_scene(params=_params(solver, substeps, restitution), seed=seed, velocity=velocity)
    run(state, step_fn, stats_fn=_stats, save_gif=save_gif, gif_duration=gif_duration)


@app.command()
def pendulum(
    solver: str = typer.Option("jacobi", help="Position solver: 'gs' or 'jacobi'"),
    substeps: int = typer.Option(10, help="Substeps per frame"),
) -> None:
    """Single box on a hinge joint with ±70° angle limits."""
    from spec_driven_xpbd.scenes.pendulum import make_scene
    from spec_driven_xpbd.viewer import run

    p = SimParams(solver=solver, num_substeps=substeps)
    state, step_fn = make_scene(params=p)
    run(state, step_fn)


@app.command()
def double_pendulum(
    solver: str = typer.Option("jacobi", help="Position solver: 'gs' or 'jacobi'"),
    substeps: int = typer.Option(10, help="Substeps per frame"),
    save_gif: str = typer.Option(None, help="Save a GIF recording to this path"),
    gif_duration: float = typer.Option(5.0, help="Duration of GIF in seconds"),
) -> None:
    """Two slender rods on hinge joints — chaotic double pendulum."""
    from spec_driven_xpbd.scenes.double_pendulum import make_scene
    from spec_driven_xpbd.viewer import run

    from spec_driven_xpbd.engine.math import total_energy

    p = SimParams(solver=solver, num_substeps=substeps)
    state, step_fn = make_scene(params=p)

    def _stats(s):
        ke_lin, ke_rot, pe, total = total_energy(s, p.gravity)
        return f"E={float(total):.2f}  KE={float(ke_lin + ke_rot):.2f}  PE={float(pe):.2f}"

    import math
    run(state, step_fn, stats_fn=_stats,
        colors={0: (1.0, 0.35, 0.35), 1: (0.35, 1.0, 0.35)},
        camera=((8.0, 3.5, -0.5), -math.pi / 2, 0.0),
        save_gif=save_gif, gif_duration=gif_duration)


@app.command()
def stack(
    solver: str = typer.Option("jacobi", help="Position solver: 'gs' or 'jacobi'"),
    substeps: int = typer.Option(10, help="Substeps per frame"),
    restitution: float = typer.Option(0.3, help="Coefficient of restitution"),
) -> None:
    """Three cubes stacked at rest with a small horizontal offset (stability test)."""
    from spec_driven_xpbd.scenes.stack import make_scene
    from spec_driven_xpbd.viewer import run

    state, step_fn = make_scene(params=_params(solver, substeps, restitution))
    run(state, step_fn)
