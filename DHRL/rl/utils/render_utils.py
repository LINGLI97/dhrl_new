"""
Eval GIF rendering utilities for DHRL.
Produces per-epoch eval GIFs with real MuJoCo frames, minimap, and live metrics.
Only active for AntMaze envs; silently disabled for others.
"""
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    _RENDER_DEPS = True
except ImportError:
    _RENDER_DEPS = False

_FONT = None

def _get_font(size=14):
    global _FONT
    if _FONT is None:
        try:
            _FONT = {}
        except Exception:
            pass
    if size not in _FONT:
        try:
            _FONT[size] = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            _FONT[size] = ImageFont.load_default()
    return _FONT[size]


def is_maze_env(env):
    """True if env is an AntMaze EnvWithGoal."""
    return hasattr(env, 'base_env') and hasattr(env.base_env, '_init_torso_x')


def get_maze_components(env):
    """Returns (maze_env, ant_env, torso_x, torso_y, size_scaling, structure)."""
    from envs.antenv.maze_env_utils import construct_maze
    maze_env    = env.base_env
    ant_env     = maze_env.wrapped_env
    torso_x     = maze_env._init_torso_x
    torso_y     = maze_env._init_torso_y
    size_scaling = maze_env.MAZE_SIZE_SCALING
    structure   = construct_maze(env.env_name)
    return maze_env, ant_env, torso_x, torso_y, size_scaling, structure


def render_closeup(ant_env, ant_x, ant_y, width=480, height=480):
    """Angled close-up camera that tracks the ant."""
    viewer = ant_env._get_viewer('rgb_array')
    viewer.cam.distance  = 7
    viewer.cam.elevation = -45
    viewer.cam.azimuth   = 135
    viewer.cam.lookat[0] = ant_x
    viewer.cam.lookat[1] = ant_y
    viewer.cam.lookat[2] = 0.5
    frame = ant_env.sim.render(width=width, height=height,
                               camera_name=None, depth=False, mode='offscreen')
    return frame[::-1]


def render_minimap(x_trail, y_trail, col_trail, ant_x, ant_y,
                   goal_x, goal_y,
                   structure, size_scaling, torso_x, torso_y, size=160):
    """Top-down minimap: maze walls + trajectory + collision dots + goal star."""
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100,
                           facecolor='#111111')
    for i, row in enumerate(structure):
        for j, cell in enumerate(row):
            if cell == 1:
                x = j * size_scaling - torso_x
                y = i * size_scaling - torso_y
                ax.add_patch(plt.Rectangle(
                    (x - size_scaling / 2, y - size_scaling / 2),
                    size_scaling, size_scaling, color='#555555', zorder=0))
    if len(x_trail) > 1:
        ax.plot(x_trail, y_trail, color='#4fa3e0', lw=0.8, alpha=0.6, zorder=1)
    xt = np.array(x_trail); yt = np.array(y_trail)
    ct = np.array(col_trail, dtype=bool)
    if ct.sum() > 0:
        ax.scatter(xt[ct], yt[ct], c='#ff4444', s=10, zorder=3)
    # goal
    ax.scatter([goal_x], [goal_y], c='#ffdd00', s=60, marker='*',
               zorder=6, edgecolors='white', linewidths=0.5)
    # current position
    ax.scatter([ant_x], [ant_y], c='#00ff99', s=40, zorder=5,
               edgecolors='white', linewidths=0.6)
    ax.set_facecolor('#1a1a2e')
    ax.set_aspect('equal'); ax.axis('off')
    xs = [j * size_scaling - torso_x
          for i, row in enumerate(structure) for j, cell in enumerate(row)]
    ys = [i * size_scaling - torso_y
          for i, row in enumerate(structure) for cell in row]
    pad = size_scaling * 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    plt.close(fig)
    return buf


def composite_frame(main_np, minimap_np, metrics: dict, collision: bool):
    """
    Overlay metrics bar + collision flash + minimap onto the main frame.
    metrics: dict of label -> value to display in the status bar.
    """
    img  = Image.fromarray(main_np)
    draw = ImageDraw.Draw(img)
    W, H = img.size
    font_sm = _get_font(13)
    font_lg = _get_font(15)

    # red border flash on collision
    if collision:
        for b in range(10):
            draw.rectangle([b, b, W - 1 - b, H - 1 - b], outline=(255, 40, 40))

    # ── top status bar ────────────────────────────────────────────────────────
    bar_h     = 48
    bar_color = (140, 0, 0) if collision else (15, 15, 35)
    draw.rectangle([0, 0, W, bar_h], fill=bar_color)

    status_tag = "  [COLLISION]" if collision else ""
    # first line: step + collision tag
    t_val = metrics.get('step', '?')
    draw.text((8, 4), f"step={t_val}{status_tag}",
              fill=(255, 80, 80) if collision else (200, 220, 255), font=font_lg)

    # second line: key metrics
    metric_parts = []
    for k, v in metrics.items():
        if k == 'step':
            continue
        if isinstance(v, float):
            metric_parts.append(f"{k}:{v:.3f}")
        else:
            metric_parts.append(f"{k}:{v}")
    draw.text((8, 26), "  ".join(metric_parts),
              fill=(180, 210, 180), font=font_sm)

    # ── minimap inset (bottom-right) ──────────────────────────────────────────
    mm  = Image.fromarray(minimap_np)
    mw, mh = mm.size
    border = 2
    draw.rectangle([W - mw - border * 2 - 4, H - mh - border * 2 - 4,
                    W - 4, H - 4], fill=(30, 30, 50))
    img.paste(mm, (W - mw - border - 4, H - mh - border - 4))
    draw.text((W - mw - border - 4, H - mh - border - 18),
              "MAP", fill=(180, 180, 180), font=font_sm)

    return np.array(img)


def save_eval_gif(frames, save_dir, epoch, use_wandb=False):
    """Save list of numpy frames as a GIF; optionally log to wandb."""
    if not frames:
        return
    gif_dir = os.path.join(save_dir, 'eval_gifs')
    os.makedirs(gif_dir, exist_ok=True)
    gif_path = os.path.join(gif_dir, f'eval_epoch_{epoch:04d}.gif')
    # hold last frame 2 s
    frames = frames + [frames[-1]] * 30
    imageio.mimsave(gif_path, frames, fps=20, loop=0)
    print(f"  eval GIF saved → {gif_path}")

    if use_wandb:
        try:
            import wandb
            wandb.log({'eval/gif': wandb.Video(gif_path, fps=20, format='gif')},
                      step=epoch)
        except Exception as e:
            print(f"  wandb gif log failed: {e}")
