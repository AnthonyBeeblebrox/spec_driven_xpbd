# SPECIFICATIONS — xpbd

Rigid body simulation library implementing the XPBD algorithm from:
> Müller et al., "Detailed Rigid Body Simulation with Extended Position Based Dynamics", ACM SIGGRAPH 2020.

## 1. Goals

- **Learning project and library**: clean, readable JAX implementation that maps directly to the paper's equations.
- **Fast**: JIT-compiled via `jax.jit`; runs on CPU or GPU without code changes.
- **Configurable precision**: `float32`, `float64`, or `bfloat16` selectable at sim init (JAX dtype).
- **Visualized**: real-time OpenGL/GLFW renderer for passive observation.

## 2. Scope (initial)

| Feature                | Status       |
|------------------------|--------------|
| Box rigid bodies       | in scope     |
| Ground plane contact   | in scope     |
| Box-box contact        | in scope     |
| Hinge joints           | in scope     |
| Gauss-Seidel solver    | in scope     |
| Jacobi solver          | in scope     |
| Float precision choice | in scope     |
| CPU + GPU backends     | in scope     |
| Other shapes           | out of scope |
| Other joint types      | out of scope |
| Interactive renderer   | out of scope |

## 3. Data Structures

All simulation state is stored as a single **`SimState`** pytree (JAX-registered dataclass),
holding N-body batched arrays. Static bodies have `m_inv = 0`.

### 3.1 SimState

```
SimState:
  # Kinematics — shape (N, *)
  x         : (N, 3)   # center-of-mass position
  v         : (N, 3)   # linear velocity
  q         : (N, 4)   # orientation quaternion [w, x, y, z], unit norm
  omega     : (N, 3)   # angular velocity (world frame)

  # Inertial properties — constant
  m_inv     : (N,)     # inverse mass (0 = static/infinite mass)
  I_inv_body: (N, 3)   # diagonal of inverse inertia tensor in principal/body frame

  # Shape
  half_ext  : (N, 3)   # box half-extents
```

Quaternion convention: **[w, x, y, z]**.

`q` encodes the rotation from the body's principal frame (where I is diagonal) to the world frame.
`I_inv_body` is always diagonal because the body frame is chosen to be the principal frame.
For a box, the principal axes coincide with the box's own axes by symmetry.

### 3.2 HingeJoint

Batched: all arrays shape `(J,)` or `(J, 3)`.

**Sentinel convention:** `body_b = N` (one past the last valid body index) means
world-anchor.  The solver extends state arrays with a zero row at index `N`
(`m_inv=0`, `I_inv=0`) and an identity quaternion, so sentinel slots contribute
zero correction automatically — matching the ContactBuffer convention.
When `body_b = N`, `r_b`, `axis_b`, and `ref_b` are given in **world frame**.

```
HingeJoint:                              # all arrays shape (J,) or (J, 3)
  body_a    : (J,) int32  # index into SimState
  body_b    : (J,) int32  # N = world-anchor sentinel
  r_a       : (J, 3)      # attachment point in body_a local frame
  r_b       : (J, 3)      # attachment point in body_b local frame (world if body_b=N)
  axis_a    : (J, 3)      # hinge axis in body_a local frame
  axis_b    : (J, 3)      # hinge axis in body_b local frame (world if body_b=N)
  ref_a     : (J, 3)      # reference perp vector in body_a frame (defines zero angle)
  ref_b     : (J, 3)      # reference perp vector in body_b frame (world if body_b=N)
  angle_min : (J,)        # lower limit (radians); -inf = no limit
  angle_max : (J,)        # upper limit (radians); +inf = no limit
  compliance: (J,)        # α in m/N; 0 = rigid
  lambda_pos  : (J,)      # positional Lagrange multiplier (reset each substep)
  lambda_ang  : (J,)      # axis-alignment Lagrange multiplier (reset each substep)
  lambda_limit: (J,)      # angle-limit Lagrange multiplier (reset each substep)
```

`empty_hinge_joints(dtype)` returns a `HingeJoint` with J=0 for scenes without joints.

### 3.3 ContactBuffer (fixed-size padded array, generated once per full step)

JAX JIT requires static array shapes, so contacts are stored in a fixed-size
buffer of capacity `C` computed once from the body count `N`:

```
C = 8*N*(N-1)      # 8 slots per ordered pair (i,j), i≠j
                   # vertex-face: up to 8 active; edge-edge: 1 active, rest inactive
```

Inactive slots have `d ≤ 0`; solvers gate every correction on `d > 0` so
inactive slots contribute zero without branching.  `lambda_n` and `lambda_t`
are reset to zero at the start of each substep by `_init_lambdas`; the
geometry fields are reused across substeps.

**Sentinel convention:** inactive slots store `body_a = body_b = N` (one past
the last valid body index).  Solvers must extend `m_inv` and `I_inv_body` with
one extra zero row at index `N` before indexing, so that gathering on sentinel
slots yields zero contribution automatically — no per-slot branching needed.

```
ContactBuffer:                    # all arrays shape (C,) or (C, 3)
  body_a    : (C,)  int32   # index into SimState; N = sentinel for inactive slots
  body_b    : (C,)  int32   # index into SimState; N = sentinel for inactive slots
  r_a       : (C, 3)        # contact point relative to body_a CoM (body_a frame)
  r_b       : (C, 3)        # contact point relative to body_b CoM (body_b frame)
  n         : (C, 3)        # contact normal pointing from b toward a
  d         : (C,)          # penetration depth; ≤ 0 → inactive
  lambda_n  : (C,)          # normal Lagrange multiplier (reset each substep)
  lambda_t  : (C,)          # tangential Lagrange multiplier (reset each substep)
```

## 4. Simulation Parameters

```
SimParams:
  dt            : float   # time step (default 1/60 s)
  num_substeps  : int     # substeps per frame (default 20)
  num_pos_iters : int     # position solver iterations per substep (default 1)
  gravity       : (3,)    # default (0, -9.81, 0)
  mu_static     : float   # static friction coefficient (default 0.5)
  mu_dynamic    : float   # dynamic friction coefficient (default 0.3)
  restitution   : float   # coefficient of restitution e ∈ [0, 1] (default 0.3)
  solver        : str     # 'gs' (Gauss-Seidel, default) or 'jacobi'
```

`solver` is resolved at JIT trace time (SimParams is a static closure via `functools.partial`), so no runtime branch overhead.

## 5. Algorithm

Implements **Algorithm 2** from the paper exactly.

```
xpbd_step(state, joints=None, *, params) -> state:
  # joints: HingeJoint pytree (J joints); None or J=0 means no joints.
  # AABB broadphase — once per full step (paper §3.5).
  pair_mask = broadphase_pairs(state, params.dt)   # (N*(N-1),) bool  [done]
  h = params.dt / params.num_substeps

  for _ in range(params.num_substeps):
    x_prev, q_prev = state.x, state.q

    # --- Integrate (explicit Euler) ---                                         [done]
    state.v += h * gravity              # static bodies masked (m_inv == 0)
    state.x += h * state.v
    I, I_inv = I_world(q, I_inv_body)   # rotate inertia tensor to world frame
    state.omega += h * I_inv @ (-omega × (I @ omega))   # gyroscopic term
    state.q += h/2 * quat_mul([0, omega], state.q)
    state.q /= norm(state.q)

    # --- Narrowphase — once per substep on post-integrate state (paper §3.5). ---
    # Fresh n, r_a, r_b, d every substep; lambda_n = lambda_t = 0.             [done]
    contacts = narrowphase_contacts(state, pair_mask)

    # --- Solve positions (joints first, then contacts) ---
    joints_sub = reset_lambdas(joints)   # lambda_pos/ang/limit ← 0 each substep
    for _ in range(params.num_pos_iters):
      state, joints_sub = _solve_joints(state, joints_sub, h)
      state, contacts   = _solve_contacts(state, contacts, h)

    # --- Derive velocities ---                                                  [done]
    state.v = (state.x - x_prev) / h
    dq = quat_mul(state.q, quat_inv(q_prev))
    state.omega = 2 * dq[1:] / h
    state.omega = where(dq[0] >= 0, state.omega, -state.omega)

    # --- Solve velocities (friction + restitution) ---                          [placeholder]
    state = _solve_velocities(state, contacts, h, params)

  return state
```

## 6. Math Primitives (`engine/math.py`)

```
quat_mul(a, b)        Hamilton product, convention [w, x, y, z]
quat_inv(q)           Conjugate (= inverse for unit quaternions)
quat_to_mat(q)        Rotation matrix (3×3) from quaternion
rotate(q, v)          Rotate vector v by quaternion q
I_world(q, I_inv_body) -> (I, I_inv)
                      Rotate diagonal body-frame inertia to world frame:
                        I     = R @ diag(1/I_inv_body) @ Rᵀ
                        I_inv = R @ diag(I_inv_body)   @ Rᵀ
                      Safe: where I_inv_body == 0, I = 0 (static body).
```

## 7. Constraint Projection

### 7.1 Positional constraint (eq. 2–9)

Given bodies a, b with contact/attachment points r_a, r_b (world frame), correction magnitude c, normal n:

```
w_a = m_inv_a + (r_a × n)ᵀ I_inv_a (r_a × n)
w_b = m_inv_b + (r_b × n)ᵀ I_inv_b (r_b × n)
α̃   = compliance / h²
Δλ  = (-c - α̃ λ) / (w_a + w_b + α̃)
λ  += Δλ
p   = Δλ * n

x_a += p * m_inv_a
x_b -= p * m_inv_b
q_a += 0.5 * quat_mul([I_inv_a @ (r_a × p); 0], q_a)
q_b -= 0.5 * quat_mul([I_inv_b @ (r_b × p); 0], q_b)
q_a /= |q_a|;  q_b /= |q_b|
```

### 7.2 Angular constraint (eq. 11–16)

Given correction axis n (world frame) and angle θ:

```
w_a = nᵀ I_inv_a n
w_b = nᵀ I_inv_b n
Δλ  = (-θ - α̃ λ) / (w_a + w_b + α̃)
λ  += Δλ
p   = Δλ * n

q_a += 0.5 * quat_mul([I_inv_a @ p; 0], q_a)
q_b -= 0.5 * quat_mul([I_inv_b @ p; 0], q_b)
```

### 7.3 Hinge joint

Three sub-constraints applied in order each position iteration:

1. **Positional**: bilateral attachment — c = |p_a - p_b|, n = normalize(p_a - p_b). No λ clamping.
2. **Angular axis alignment**: Δq_hinge = axis_b_world × axis_a_world (reversed vs. eq. 20 — required by our world-frame left-multiplication quaternion update convention).
3. **Angle limits** (if finite): Algorithm 3 — LimitAngle(n, n1, n2, α_min, α_max).
   n = axis_a_world (hinge axis), n1 = rotate(q_a, ref_a), n2 = rotate(q_b, ref_b).
   Correction vector: Δq_limit = **n2 × n1_target** (note order — drives n1 toward n1_target).

Joints are solved **before contacts** in every position iteration.

### 7.4 Contact constraint

Applied as a unilateral positional constraint (only if d > 0):

- **Normal**: c = -d, n = contact normal. Lagrange multiplier clamped: λ_n ≥ 0.
- **Tangential (static friction)**: applied after normal correction each position iteration.
  Resists sliding by treating the tangential displacement since substep start as a
  positional constraint, clamped to the Coulomb cone.

```
# World-frame contact points at substep start (x_prev, q_prev captured before _integrate)
p_a_prev = x_a_prev + rotate(q_a_prev, r_a)
p_b_prev = x_b_prev + rotate(q_b_prev, r_b)

# Current world-frame contact points
p_a = x_a + rotate(q_a, r_a)
p_b = x_b + rotate(q_b, r_b)

# Relative tangential displacement since substep start
delta   = (p_a - p_b) - (p_a_prev - p_b_prev)
delta_t = delta - dot(n, delta) * n       # project out normal component

# Tangential constraint (eq. 2–9 with n_t = delta_t / |delta_t|)
# Only applied when |delta_t| > 0
c_t  = |delta_t|
n_t  = delta_t / |delta_t|
Δλ_t = -c_t / (w_a_t + w_b_t)   # w computed along n_t, compliance = 0
λ_t += Δλ_t

# Coulomb cone clamp (static friction)
λ_t  = clamp(λ_t, -μ_s * λ_n, μ_s * λ_n)
Δλ_t = λ_t - λ_t_prev            # effective Δλ after clamp

p_t  = Δλ_t * n_t
# Position and orientation corrections use same formulas as normal constraint.
```

`lambda_t` in `ContactBuffer` is the tangential Lagrange multiplier, accumulated across
position iterations exactly like `lambda_n`. Reset to zero at the start of each substep.
`x_prev` and `q_prev` are already captured at the top of `substep_fn`; they must be passed
to `_solve_positions_gs` / `_solve_positions_jacobi` alongside `contacts`.

## 8. Velocity Solve (eq. 29–34)

For each contact:

r_a, r_b are the contact offsets rotated to world frame using the post-solve orientation q.
n convention: points from b toward a (outward from b).  v_n < 0 = approaching.
v_n_prev: computed from v/omega saved before integration (start of substep).
  Lever arms for v_n_prev are rotated using q_prev (pre-substep orientation) so that
  velocities and orientations are consistent in time.

```
v_rel  = (v_a + omega_a × r_a) - (v_b + omega_b × r_b)
v_n    = n · v_rel                     # < 0 approaching, > 0 separating
v_t    = v_rel - v_n * n
f_n    = lambda_n / h²

# Dynamic friction (eq. 30)
Δv_friction = -v_t / |v_t| * min(h * μ_d * |f_n|, |v_t|)   [0 when |v_t| = 0]

# Pre-substep normal velocity (eq. 29 applied to pre-integration state)
v_n_prev = n · v_rel_before_substep    # < 0 if was approaching

# Restitution (eq. 34, sign-adapted for our n convention)
# Target: v_n_target = max(-e * v_n_prev, 0)  [> 0 = separating when was approaching]
# Applied only when |v_n| > 2|g|h (suppresses jitter at rest)
v_n_target = max(-e * v_n_prev, 0)   if |v_n| > 2|g|h
           = v_n                      otherwise   (→ Δv_restitution = 0)
Δv_restitution = n * (v_n_target - v_n)

# Apply via generalized inverse mass (eq. 33)
p       = (Δv_friction + Δv_restitution) / (w_a + w_b)
v_a     += p * m_inv_a;         v_b     -= p * m_inv_b
omega_a += I_inv_a @ (r_a × p); omega_b -= I_inv_b @ (r_b × p)
```

Applied as a single vectorized pass over all active contacts (d > 0).
Corrections accumulated with scatter-add (no averaging — the w denominator in each
contact already accounts correctly for mass and inertia).

## 9. Collision Detection

### 9.1 Broadphase — AABB

All bodies are SimState entries (no special ground half-space). Static-static
pairs (both m_inv == 0) are skipped.

For each remaining ordered pair (i, j):
- Compute the world-space AABB of each body: `half = |R| @ half_ext + k*dt*|v|`, k=2.
- Overlap on all 3 axes → candidate pair.

Computed once per full time step (not per substep).

### 9.2 Narrowphase — Box vs Box (SAT + full contact generation)

For each broadphase candidate pair (i, j):

**SAT separation test** — computed once per *unordered* pair {i,j} (SAT is symmetric),
then broadcast to both ordered slots (i,j) and (j,i).
15 axes: 3 face normals of i, 3 face normals of j,
9 edge cross-products R_i[:,a] × R_j[:,b].  Overlap along axis n:

```
overlap(n) = (h_i @ |R_i.T n|) + (h_j @ |R_j.T n|) - |dot(x_j - x_i, n)|
```

Any overlap < 0 → separated, all 8 slots inactive.
Degenerate cross-product (|cross| < 1e-6) → skip axis (set overlap = +∞).

**Contact type classification** — the axis with minimum overlap over all 15 determines
the contact feature type, which drives contact point generation:

| Min-axis index | Contact type | Description |
|---|---|---|
| 0–2 | Face-i / Vertex-j | A face of i is the reference; vertices of j are tested against i |
| 3–5 | Face-j / Vertex-i | A face of j is the reference; vertices of i are tested against j |
| 6–14 | Edge-Edge | An edge of i and an edge of j form the contact |

`_sat_result(state, i, j) → (not_sep, min_axis, min_depth)` returns all three values in one
pass over 15 axes.  Axes 0–5 correspond to rows of `R_i.T` then `R_j.T` (i.e. columns of
`R_i` then `R_j`).  Axes 6–14 index `ea*3+eb` with `ea, eb ∈ {0,1,2}` for the 9
cross-products `R_i[:,ea] × R_j[:,eb]`.

**Vertex-face contact generation** (8 slots per ordered pair) — unchanged from original.
For ordered pair (i,j), vertices of i are tested against j's faces.  Reference face k
selected as `argmin(overlap)` over j's 3 face normals.  The symmetric ordered pair (j,i)
covers the case where i's faces are the reference.

For each vertex v of body i:
1. `v_local = R_j.T @ (v - x_j)` — vertex in j's local frame.
2. `inside = all(|v_local| <= h_j)`.
3. `n = sign(v_local[k]) * R_j[:,k]` — outward normal of j's reference face, pointing from j toward i.
4. `d = h_j[k] - |v_local[k]|` if inside, else -1.

**Edge-edge contact generation** (axes 6–14) — overwrites slot 0 when `min_axis >= 6` and
`i < j`.  The i > j ordered pair is left inactive (d = -1) to avoid a duplicate constraint.

Let `ea = (min_axis-6)//3`, `eb = (min_axis-6)%3`.

1. Edge axes: `a_i = R_i[:,ea]`, `a_j = R_j[:,eb]`.
2. `n = normalize(a_i × a_j)`, oriented so `dot(n, x_i - x_j) > 0` (points from j toward i).
3. Select closest edge of i: for each axis k≠ea, `s_k = sign(dot(x_j-x_i, R_i[:,k]))`; set `s_ea = 0`.
   Edge center: `c_i = x_i + R_i @ (s ⊙ h_i)`.
4. Symmetrically select closest edge of j (using `x_i-x_j` as direction).
5. Closest point on segment pair (Ericson §5.1.9): parameterize as
   `P(t) = c_i + t·a_i`, `Q(s) = c_j + s·a_j`, `t ∈ [-h_i[ea], h_i[ea]]`,
   `s ∈ [-h_j[eb], h_j[eb]]`, solve with clamp-and-re-clamp.
6. Contact point `p = (p_i + p_j) / 2`.
7. `d = h_i @ |R_i.T n| + h_j @ |R_j.T n| - |dot(x_j - x_i, n)|` (SAT overlap formula).
8. `r_a = R_i.T @ (p - x_i)`, `r_b = R_j.T @ (p - x_j)` in body frames.

## 10. Solver Variants

### 10.1 Gauss-Seidel

- Python `for` loop over constraints.
- State updated immediately after each constraint (non-linear NPGS).
- Full substep loop JIT-compiled; constraint loop unrolled at trace time (fixed constraint count per scene).

### 10.2 Jacobi

- All constraint corrections computed simultaneously via `jax.vmap`, applied once per iteration.
- Corrections are scatter-added without per-body averaging — the 1/w scaling in each
  correction already encodes the correct mass weighting.

Both solvers expose the same interface: `solve_positions(state, contacts, lambdas, h) -> (state, lambdas)`.

## 11. Module Layout

```
src/xpbd/
  __init__.py        # main() → cli app
  cli.py             # typer CLI, one @app.command() per scene
  viewer.py          # run(state, step_fn, draw_fn=None) — GL loop + renderer
  engine/
    __init__.py      # exports: SimState, SimParams, HingeJoint, empty_hinge_joints, xpbd_step
    state.py         # SimState, SimParams, HingeJoint, empty_hinge_joints
    math.py          # quat_mul, quat_inv, quat_to_mat, rotate, I_world
    step.py          # xpbd_step — Algorithm 2
    joints.py        # _solve_joints_gs, _solve_joints_jacobi
  scenes/
    __init__.py
    falling_box.py   # make_scene() → (state, step_fn)
    pendulum.py      # single pendulum with ±70° angle limits

scripts/             # free-form experiments, not part of the package
```

## 12. Viewer (`viewer.py`)

Single entry point: `run(state, step_fn, draw_fn=None)`.

- `step_fn(state) -> state` called once per frame.
- `draw_fn(state)` optional; if `None`, auto-draws ground + all boxes from state.
- FPS displayed in window title, updated every second.

**Camera**: FPS-style, mouse captured on launch.

| Input         | Action                          |
|---------------|---------------------------------|
| Mouse         | Yaw / pitch (pitch clamped ±89°)|
| W / S         | Forward / backward              |
| A / D         | Strafe left / right             |
| Left Shift    | Move up                         |
| Left Ctrl     | Move down                       |
| Escape        | Close window                    |

**Rendering**:
- Ground: lit quad (normal (0,1,0)) + unlit grid lines.
- Boxes: 6 solid quads with per-face normals, GL_LIGHTING + GL_LIGHT0 (directional, world-space).
  `GL_COLOR_MATERIAL` maps `glColor3f` to material ambient+diffuse.
  Light position set each frame after camera setup so it is fixed in world space.

## 13. Public API

```python
from xpbd.scenes.falling_box import make_scene
from xpbd.viewer import run

state, step_fn = make_scene()
run(state, step_fn)
```

Scenes with joints capture them via `partial`:
```python
step_fn = jax.jit(partial(xpbd_step, joints=joints, params=params))
```

CLI:
```
uv run xpbd falling-box
uv run xpbd pendulum
uv run xpbd <scene-name>       # one command per file in scenes/
```

Each scene exposes `make_scene() -> (SimState, step_fn)` where `step_fn` is already `jax.jit`-compiled via `functools.partial(xpbd_step, joints=..., params=params)`.
Scenes without joints use `partial(xpbd_step, params=params)` (joints defaults to None → empty).

Box inertia (solid box, mass m, half-extents hx/hy/hz):
```
Ix = m/3 * (hy² + hz²)
Iy = m/3 * (hx² + hz²)
Iz = m/3 * (hx² + hy²)
```

## 14. Decisions

- **Box-box contact point generation**: all three contact types (vertex-face, edge-edge) must be handled.  Face-face is a degenerate case of vertex-face (parallel faces) and is adequately covered by vertex enumeration; full Sutherland-Hodgman polygon clipping may be added later if stacking stability requires it.
- **Jacobi convergence**: accumulation strategy determined empirically during implementation.
- **Constraint solver loop**: Python `for` loop unrolled at JIT trace time. `jax.lax.fori_loop` deferred until profiling shows it necessary.
- **Lighting**: fixed-function `GL_LIGHTING` with one directional `GL_LIGHT0`. No shaders.
- **draw_fn**: optional per-scene override; default auto-renders ground + all boxes.
- **Scene initialization — no initial overlaps**: bodies must not overlap at t=0.  XPBD derives velocity from position corrections (`v = Δx/h`) so any initial penetration of depth `d` produces an impulsive velocity `d/h` (e.g. 0.07 m overlap at h=8.3×10⁻⁴ s → 84 m/s), causing immediate explosion.  Scene builders must guarantee pairwise separation ≥ 2×circumradius = 0.87 m for 0.5 m cubes at any orientation.

## 15. Implementation Status

| Component               | Status         |
|-------------------------|----------------|
| SimState / SimParams    | done           |
| Quaternion math         | done           |
| Explicit Euler integrate| done           |
| Velocity derivation     | done           |
| GLFW viewer + camera    | done           |
| GL_LIGHTING renderer    | done           |
| Typer CLI               | done           |
| falling_box scene       | done           |
| Broadphase (AABB)       | done           |
| SAT separation test     | done           |
| Vertex-face contacts    | done           |
| Edge-edge contacts      | done           |
| Positional constraints  | done           |
| Angular constraints     | done           |
| Static friction (pos.)  | done           |
| Velocity solve          | done           |
| Hinge joint             | done           |
| Jacobi solver           | done           |
