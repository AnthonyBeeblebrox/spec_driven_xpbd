"""Generic GLFW/OpenGL viewer. Call run(state, step_fn) from any scene."""

import math
from collections.abc import Callable

import numpy as np
import glfw
from OpenGL.GL import *

# ---------------------------------------------------------------------------
# Camera state
# ---------------------------------------------------------------------------

_cam_pos = [0.0, 12.0, 6.0]
_cam_yaw = 0.0
_cam_pitch = -math.atan(12.0 / 6.0)
_last_mx: float | None = None
_last_my: float | None = None

_SENSITIVITY = 0.002
_SPEED = 5.0

# ---------------------------------------------------------------------------
# Box faces: (normal, quad vertices in unit-cube coords)
# ---------------------------------------------------------------------------

_FACES = [
    ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),  # front
    ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)]),  # back
    ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)]),  # right
    ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),  # left
    ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)]),  # top
    ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),  # bottom
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(state, step_fn: Callable, draw_fn: Callable | None = None, stats_fn: Callable | None = None, colors: dict | None = None, camera: tuple | None = None, save_gif: str | None = None, gif_duration: float = 5.0, gif_fps: int = 20) -> None:
    """
    Start the simulation viewer.

    step_fn(state) -> state   called once per frame
    draw_fn(state)            optional; if None, auto-draws ground + boxes
    stats_fn(state) -> str    optional; result appended to window title each second
    camera                    optional ((x,y,z), yaw, pitch) to override default view
    save_gif                  optional path to save a GIF recording
    gif_duration              duration in seconds to record (default 5.0)
    gif_fps                   frames per second for the GIF (default 20)
    """
    cam_pos, cam_yaw, cam_pitch = camera if camera is not None else (None, None, None)
    _init_camera(cam_pos, cam_yaw, cam_pitch)

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    window = glfw.create_window(800, 600, "xpbd", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # --- OpenGL state ---
    glEnable(GL_DEPTH_TEST)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.15, 0.15, 0.15, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.4, 0.4, 0.4, 1.0])

    # glColor3f drives material ambient+diffuse
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 32.0)

    # Normalize normals after non-uniform scales
    glEnable(GL_NORMALIZE)

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_cursor_pos_callback(window, _on_mouse)

    if draw_fn is not None:
        _draw = draw_fn
    else:
        _draw = lambda s: (_draw_ground(), _draw_boxes(s, colors))

    last_time = glfw.get_time()
    fps_time = last_time
    fps_frames = 0

    gif_frames: list = []
    gif_frame_interval = 1.0 / gif_fps
    gif_next_capture = 0.0

    while not glfw.window_should_close(window):
        now = glfw.get_time()
        dt = min(now - last_time, 0.1)
        last_time = now

        if save_gif and now >= gif_duration:
            break

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        _move(window, dt)
        state = step_fn(state)

        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)
        glClearColor(0.15, 0.15, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        _setup_camera(width, height)
        # Set light position in world space (w=0 → directional light)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 3.0, 2.0, 0.0])

        _draw(state)

        if save_gif and now >= gif_next_capture:
            pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            img = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
            gif_frames.append(img[::-1])  # flip vertically (OpenGL origin is bottom-left)
            gif_next_capture = now + gif_frame_interval

        glfw.swap_buffers(window)
        glfw.poll_events()

        fps_frames += 1
        if now - fps_time >= 1.0:
            title = f"xpbd — {fps_frames} fps"
            if stats_fn is not None:
                title += f"  |  {stats_fn(state)}"
            glfw.set_window_title(window, title)
            fps_frames = 0
            fps_time = now

    glfw.terminate()

    if save_gif and gif_frames:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in gif_frames]
        imgs[0].save(save_gif, save_all=True, append_images=imgs[1:], loop=0, duration=int(1000 / gif_fps))
        print(f"Saved {len(gif_frames)} frames to {save_gif}")


# ---------------------------------------------------------------------------
# Auto-renderer
# ---------------------------------------------------------------------------


def _auto_draw(state) -> None:
    _draw_ground()
    _draw_boxes(state, None)


def _draw_ground() -> None:
    size = 10

    # Lit ground quad
    glNormal3f(0.0, 1.0, 0.0)
    glColor3f(0.3, 0.5, 0.3)
    glBegin(GL_QUADS)
    glVertex3f(-size, 0.0, -size)
    glVertex3f(size, 0.0, -size)
    glVertex3f(size, 0.0, size)
    glVertex3f(-size, 0.0, size)
    glEnd()

    # Grid lines — disable lighting so they stay flat-colored
    glDisable(GL_LIGHTING)
    glColor3f(0.5, 0.7, 0.5)
    glBegin(GL_LINES)
    for i in range(-size, size + 1):
        glVertex3f(i, 0.001, -size)
        glVertex3f(i, 0.001, size)
        glVertex3f(-size, 0.001, i)
        glVertex3f(size, 0.001, i)
    glEnd()
    glEnable(GL_LIGHTING)


def _draw_boxes(state, colors: dict | None) -> None:
    xs = np.array(state.x)
    qs = np.array(state.q)
    half_exts = np.array(state.half_ext)
    m_invs = np.array(state.m_inv)

    for i, (x, q, h, m_inv) in enumerate(zip(xs, qs, half_exts, m_invs)):
        color = colors.get(i) if colors else None
        glPushMatrix()
        glMultMatrixf(_body_matrix(x, q))
        _draw_box_solid(h, dynamic=m_inv > 0, color=color)
        glPopMatrix()


def _body_matrix(x: np.ndarray, q: np.ndarray) -> list:
    """Column-major 4×4 OpenGL matrix from position x and quaternion q=[w,x,y,z]."""
    w, qx, qy, qz = q
    R = np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - w * qz),
                2 * (qx * qz + w * qy),
            ],
            [
                2 * (qx * qy + w * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - w * qx),
            ],
            [
                2 * (qx * qz - w * qy),
                2 * (qy * qz + w * qx),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ]
    )
    return [
        R[0, 0],
        R[1, 0],
        R[2, 0],
        0.0,
        R[0, 1],
        R[1, 1],
        R[2, 1],
        0.0,
        R[0, 2],
        R[1, 2],
        R[2, 2],
        0.0,
        x[0],
        x[1],
        x[2],
        1.0,
    ]


def _draw_box_solid(h: np.ndarray, dynamic: bool = True, color: tuple | None = None) -> None:
    hx, hy, hz = h
    if color is not None:
        glColor3f(*color)
    elif dynamic:
        glColor3f(0.2, 0.45, 0.9)
    else:
        glColor3f(0.9, 0.6, 0.2)
    for normal, verts in _FACES:
        glNormal3fv(normal)
        glBegin(GL_QUADS)
        for vx, vy, vz in verts:
            glVertex3f(vx * hx, vy * hy, vz * hz)
        glEnd()


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def _init_camera(pos=None, yaw=None, pitch=None) -> None:
    global _cam_pos, _cam_yaw, _cam_pitch, _last_mx, _last_my
    _cam_pos = list(pos) if pos is not None else [0.0, 12.0, 6.0]
    _cam_yaw = yaw if yaw is not None else 0.0
    _cam_pitch = pitch if pitch is not None else -math.atan(12.0 / 6.0)
    _last_mx = None
    _last_my = None


def _on_mouse(window, xpos: float, ypos: float) -> None:
    global _cam_yaw, _cam_pitch, _last_mx, _last_my
    if _last_mx is None:
        _last_mx, _last_my = xpos, ypos
        return
    _cam_yaw += (xpos - _last_mx) * _SENSITIVITY
    _cam_pitch -= (ypos - _last_my) * _SENSITIVITY
    _cam_pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, _cam_pitch))
    _last_mx, _last_my = xpos, ypos


def _move(window, dt: float) -> None:
    fwd = _forward()
    right = _right()
    dist = _SPEED * dt

    def pressed(key):
        return glfw.get_key(window, key) == glfw.PRESS

    if pressed(glfw.KEY_W):
        _cam_pos[0] += fwd[0] * dist
        _cam_pos[1] += fwd[1] * dist
        _cam_pos[2] += fwd[2] * dist
    if pressed(glfw.KEY_S):
        _cam_pos[0] -= fwd[0] * dist
        _cam_pos[1] -= fwd[1] * dist
        _cam_pos[2] -= fwd[2] * dist
    if pressed(glfw.KEY_D):
        _cam_pos[0] += right[0] * dist
        _cam_pos[2] += right[2] * dist
    if pressed(glfw.KEY_A):
        _cam_pos[0] -= right[0] * dist
        _cam_pos[2] -= right[2] * dist
    if pressed(glfw.KEY_LEFT_SHIFT):
        _cam_pos[1] += dist
    if pressed(glfw.KEY_LEFT_CONTROL):
        _cam_pos[1] -= dist


def _forward() -> tuple:
    p, y = _cam_pitch, _cam_yaw
    return (math.cos(p) * math.sin(y), math.sin(p), -math.cos(p) * math.cos(y))


def _right() -> tuple:
    return (math.cos(_cam_yaw), 0.0, math.sin(_cam_yaw))


def _setup_camera(width: int, height: int) -> None:
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    t = 0.1 * math.tan(math.radians(60.0) / 2)
    r = t * width / max(height, 1)
    glFrustum(-r, r, -t, t, 0.1, 200.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    fwd = _forward()
    eye = tuple(_cam_pos)
    ctr = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])
    _look_at(eye, ctr, (0.0, 1.0, 0.0))


def _look_at(eye, center, up) -> None:
    ex, ey, ez = eye
    cx, cy, cz = center
    ux, uy, uz = up

    fx, fy, fz = cx - ex, cy - ey, cz - ez
    fl = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / fl, fy / fl, fz / fl

    sx = fy * uz - fz * uy
    sy = fz * ux - fx * uz
    sz = fx * uy - fy * ux
    sl = math.sqrt(sx * sx + sy * sy + sz * sz)
    sx, sy, sz = sx / sl, sy / sl, sz / sl

    rx = sy * fz - sz * fy
    ry = sz * fx - sx * fz
    rz = sx * fy - sy * fx

    glLoadMatrixf(
        [
            sx,
            rx,
            -fx,
            0.0,
            sy,
            ry,
            -fy,
            0.0,
            sz,
            rz,
            -fz,
            0.0,
            -(sx * ex + sy * ey + sz * ez),
            -(rx * ex + ry * ey + rz * ez),
            fx * ex + fy * ey + fz * ez,
            1.0,
        ]
    )
