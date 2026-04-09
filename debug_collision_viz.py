"""
Collision visualization debug script.
Runs AntMazeSmall with random actions, produces:
  - collision_viz.png  : static per-episode summary (matplotlib)
  - collision_viz.gif  : real MuJoCo-rendered ant with collision overlay
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ['MUJOCO_PY_MUJOCO_PATH'] = os.path.expanduser('~/.mujoco/mujoco210')
os.environ['LD_LIBRARY_PATH'] = (
    os.path.expanduser('~/.mujoco/mujoco210/bin') + ':' +
    '/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:' +
    os.environ.get('LD_LIBRARY_PATH', '')
)
os.environ.setdefault('MUJOCO_GL', 'egl')   # headless offscreen rendering

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw, ImageFont

import envs
from envs.antenv import EnvWithGoal
from envs.antenv.create_maze_env import create_maze_env
from envs.antenv.maze_env_utils import construct_maze


# ── helpers ──────────────────────────────────────────────────────────────────

def get_all_contact_geoms(maze_env):
    sim = maze_env.wrapped_env.sim
    pairs = []
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1 = sim.model.geom_id2name(c.geom1)
        g2 = sim.model.geom_id2name(c.geom2)
        pairs.append((g1, g2))
    return pairs


def render_closeup(ant_env, ant_x, ant_y, width=480, height=480):
    """Close-up camera tracking the ant: angled view so body is clearly visible."""
    viewer = ant_env._get_viewer('rgb_array')
    viewer.cam.distance  = 7        # close enough to see legs/body
    viewer.cam.elevation = -45      # 45° down – shows 3-D body shape
    viewer.cam.azimuth   = 135      # diagonal – avoids edge-on flatness
    viewer.cam.lookat[0] = ant_x
    viewer.cam.lookat[1] = ant_y
    viewer.cam.lookat[2] = 0.5
    frame = ant_env.sim.render(width=width, height=height,
                               camera_name=None, depth=False, mode='offscreen')
    return frame[::-1]              # mujoco_py returns upside-down


def render_minimap(x_trail, y_trail, col_trail, ant_x, ant_y,
                   structure, size_scaling, torso_x, torso_y, size=160):
    """Top-down overview: maze + trajectory trail + current ant dot."""
    fig, ax = plt.subplots(figsize=(size / 100, size / 100),
                           dpi=100, facecolor='#111111')
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
    xt = np.array(x_trail); yt = np.array(y_trail); ct = np.array(col_trail, dtype=bool)
    if ct.sum() > 0:
        ax.scatter(xt[ct], yt[ct], c='#ff4444', s=10, zorder=3)
    ax.scatter([ant_x], [ant_y], c='#00ff99', s=40, zorder=5,
               edgecolors='white', linewidths=0.6)
    ax.set_facecolor('#1a1a2e')
    ax.set_aspect('equal'); ax.axis('off')
    xs = [j * size_scaling - torso_x
          for i, row in enumerate(structure)
          for j, cell in enumerate(row)]
    ys = [i * size_scaling - torso_y
          for i, row in enumerate(structure)
          for cell in row]
    pad = size_scaling * 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    plt.close(fig)
    return buf


def composite_frame(main_np, minimap_np, collision: bool, t: int, col_total: int):
    """Overlay status bar + collision flash + minimap inset onto main frame."""
    img  = Image.fromarray(main_np)
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # ── red border flash ──────────────────────────────────────────────────────
    if collision:
        for b in range(10):
            draw.rectangle([b, b, W - 1 - b, H - 1 - b],
                           outline=(255, 40, 40))

    # ── top status bar ────────────────────────────────────────────────────────
    bar_h     = 38
    bar_color = (160, 0, 0) if collision else (15, 15, 35)
    draw.rectangle([0, 0, W, bar_h], fill=bar_color)
    col_rate = 100.0 * col_total / max(t + 1, 1)
    status   = "COLLISION" if collision else "ok"
    text     = f"t={t:3d}  [{status}]   collisions: {col_total} ({col_rate:.1f}%)"
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 10), text, fill=(255, 80, 80) if collision else (200, 220, 255),
              font=font)

    # ── minimap inset (bottom-right corner) ───────────────────────────────────
    mm  = Image.fromarray(minimap_np)
    mw, mh = mm.size
    border = 2
    # dark rounded border
    draw.rectangle([W - mw - border * 2 - 4, H - mh - border * 2 - 4,
                    W - 4, H - 4], fill=(30, 30, 50))
    img.paste(mm, (W - mw - border - 4, H - mh - border - 4))
    # label
    draw.text((W - mw - border - 4, H - mh - border - 20),
              "MAP", fill=(180, 180, 180), font=font)

    return np.array(img)


def draw_maze_on(ax, structure, size_scaling, torso_x, torso_y):
    for i, row in enumerate(structure):
        for j, cell in enumerate(row):
            if cell == 1:
                x = j * size_scaling - torso_x
                y = i * size_scaling - torso_y
                ax.add_patch(plt.Rectangle(
                    (x - size_scaling / 2, y - size_scaling / 2),
                    size_scaling, size_scaling,
                    color='#444444', zorder=0))
    ax.set_facecolor('#1a1a2e')


# ── main ─────────────────────────────────────────────────────────────────────

def run(n_episodes=3, max_steps=400, size_scaling=4, seed=42):
    base_env = create_maze_env('AntMazeSmall-v0', seed=seed)
    env = EnvWithGoal(base_env, 'AntMazeSmall-v0')
    env.seed(seed)
    np.random.seed(seed)

    structure = construct_maze('AntMazeSmall-v0')
    maze_env  = base_env
    torso_x   = maze_env._init_torso_x
    torso_y   = maze_env._init_torso_y
    ant_env   = maze_env.wrapped_env   # AntEnv (MujocoEnv)

    all_x, all_y, all_col = [], [], []
    all_frames_ep0 = []          # raw MuJoCo frames for episode 0
    contact_pair_counts = {}

    for ep in range(n_episodes):
        obs = env.reset()
        ob  = obs['observation']
        x_traj, y_traj, col_traj = [], [], []
        col_count = 0

        for t in range(max_steps):
            act = env.action_space.sample()
            obs, _, done, info = env.step(act)
            ob  = obs['observation']
            col = info.get('collision', False)
            if col:
                col_count += 1

            x_traj.append(ob[0])
            y_traj.append(ob[1])
            col_traj.append(col)

            # capture rendered frame for episode 0 (every 2 steps → 200 frames)
            if ep == 0 and t % 2 == 0:
                main_frame = render_closeup(ant_env, ob[0], ob[1])
                minimap    = render_minimap(x_traj, y_traj, col_traj,
                                            ob[0], ob[1],
                                            structure, size_scaling, torso_x, torso_y)
                frame = composite_frame(main_frame, minimap, col, t, col_count)
                all_frames_ep0.append(frame)

            pairs = get_all_contact_geoms(maze_env)
            for p in pairs:
                key = tuple(sorted([str(p[0]), str(p[1])]))
                contact_pair_counts[key] = contact_pair_counts.get(key, 0) + 1

            if done:
                break

        all_x.append(x_traj)
        all_y.append(y_traj)
        all_col.append(col_traj)
        print(f"Episode {ep}: {len(x_traj)} steps, "
              f"collision={col_count} ({100*col_count/len(x_traj):.1f}%)")

    # ── contact pair report ───────────────────────────────────────────────────
    print("\n── Top contact geom pairs ──")
    for pair, cnt in sorted(contact_pair_counts.items(), key=lambda x: -x[1])[:15]:
        wall = any(str(g).startswith('block_') for g in pair)
        tag  = ' *** WALL ***' if wall else ''
        print(f"  {cnt:5d}x  {pair[0]} <-> {pair[1]}{tag}")

    # ── static PNG ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_episodes, figsize=(6 * n_episodes, 6),
                             facecolor='#0d0d1a')
    if n_episodes == 1:
        axes = [axes]
    for ep, ax in enumerate(axes):
        xt = np.array(all_x[ep]); yt = np.array(all_y[ep])
        ct = np.array(all_col[ep])
        draw_maze_on(ax, structure, size_scaling, torso_x, torso_y)
        T = len(xt)
        for t in range(T - 1):
            ax.plot(xt[t:t+2], yt[t:t+2],
                    color=plt.cm.Blues(0.3 + 0.7 * t / T), lw=1.2, zorder=1)
        ax.scatter(xt[ct],  yt[ct],  c='#ff4444', s=25, zorder=3,
                   label=f'collision ({ct.sum()})')
        ax.scatter(xt[~ct], yt[~ct], c='#4fa3e0', s=5,  zorder=2, alpha=0.4)
        ax.plot(xt[0],  yt[0],  '^', color='lime',   ms=9, zorder=5, label='start')
        ax.plot(xt[-1], yt[-1], 's', color='yellow', ms=9, zorder=5, label='end')
        ax.set_title(f'Ep {ep}  |  collision {100*ct.mean():.1f}%',
                     color='white', fontsize=11)
        ax.set_xlabel('x', color='white'); ax.set_ylabel('y', color='white')
        ax.tick_params(colors='white')
        ax.legend(fontsize=8, loc='upper right',
                  facecolor='#222222', labelcolor='white')
        ax.set_aspect('equal')
    fig.suptitle('AntMazeSmall: Trajectory with Wall Collision Detection',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    png_path = os.path.join(os.path.dirname(__file__), 'collision_viz.png')
    plt.savefig(png_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nPNG saved → {png_path}")

    # ── GIF (real MuJoCo rendering, episode 0) ───────────────────────────────
    gif_path = os.path.join(os.path.dirname(__file__), 'collision_viz.gif')
    print(f"Saving GIF ({len(all_frames_ep0)} frames) ...", flush=True)
    # hold last frame 2 s
    all_frames_ep0.extend([all_frames_ep0[-1]] * 40)
    imageio.mimsave(gif_path, all_frames_ep0, fps=20, loop=0)
    print(f"GIF saved  → {gif_path}")


if __name__ == '__main__':
    run(n_episodes=3, max_steps=400, seed=42)
