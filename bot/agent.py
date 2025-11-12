import sys
import enum
import math
from collections import defaultdict
from typing import Optional, NamedTuple, Tuple, List, Dict, Set

import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Basic types (judge-compatible)
# ────────────────────────────────────────────────────────────────────────────────

class CellType(enum.Enum):
    GOAL = 100
    START = 1
    WALL = -1
    UNKNOWN = 2
    EMPTY = 0
    NOT_VISIBLE = 3

class Player(NamedTuple):
    x: int
    y: int
    vel_x: int
    vel_y: int
    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=int)
    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vel_x, self.vel_y], dtype=int)

class Circuit(NamedTuple):
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int

class State(NamedTuple):
    circuit: Circuit
    visible_track: Optional[np.ndarray]   # safety map: NOT_VISIBLE -> WALL
    visible_raw: Optional[np.ndarray]     # raw window with NOT_VISIBLE intact
    players: List[Player]
    agent: Optional[Player]

# ────────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────────

def read_initial_observation() -> Circuit:
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)

def read_observation(old_state: State) -> Optional[State]:
    line = input()
    if line == '~~~END~~~':
        return None

    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)
    circuit = old_state.circuit

    players: List[Player] = []
    for _ in range(circuit.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    H, W = circuit.track_shape
    R = circuit.visibility_radius
    visible_raw = np.full((H, W), CellType.NOT_VISIBLE.value, dtype=int)
    visible_track = np.full((H, W), CellType.WALL.value, dtype=int)

    for i in range(2 * R + 1):
        row_vals = [int(a) for a in input().split()]
        x = posx - R + i
        if 0 <= x < H:
            y_start = posy - R
            y_end   = posy + R + 1
            loc = row_vals
            ys = y_start
            if y_start < 0:
                loc = loc[-y_start:]
                ys = 0
            if y_end > W:
                loc = loc[:-(y_end - W)]
            ye = ys + len(loc)
            if ys < ye:
                visible_raw[x, ys:ye] = loc
                safety = [CellType.WALL.value if v == CellType.NOT_VISIBLE.value else v for v in loc]
                visible_track[x, ys:ye] = safety

    return old_state._replace(visible_track=visible_track,
                              visible_raw=visible_raw,
                              players=players, agent=agent)

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────

def tri(n: int) -> int:
    return n*(n+1)//2

def brakingOk(vx: int, vy: int, rSafe: int) -> bool:
    # each axis must be stoppable within safe radius
    return (tri(abs(vx)) <= rSafe) and (tri(abs(vy)) <= rSafe)

def validLineLocal(state: State, p1: np.ndarray, p2: np.ndarray) -> bool:
    # collision check only within the CURRENT visible window (safe-by-visibility)
    track = state.visible_track
    if track is None: return False
    H, W = track.shape
    if (np.any(p1 < 0) or np.any(p2 < 0) or
        p1[0] >= H or p1[1] >= W or p2[0] >= H or p2[1] >= W):
        return False
    diff = p2 - p1
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))
        for i in range(abs(diff[0]) + 1):
            x = int(p1[0] + i*d)
            y = p1[1] + i*slope*d
            yC = int(np.ceil(y)); yF = int(np.floor(y))
            if (track[x, yC] < 0 and track[x, yF] < 0): return False
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(diff[1]) + 1):
            x = p1[0] + i*slope*d
            y = int(p1[1] + i*d)
            xC = int(np.ceil(x)); xF = int(np.floor(x))
            if (track[xC, y] < 0 and track[xF, y] < 0): return False
    return True

# ────────────────────────────────────────────────────────────────────────────────
# World model (simple; plus ASCII dumping)
# ────────────────────────────────────────────────────────────────────────────────

def is_traversable_val(v: int) -> bool:
    # Treat UNKNOWN as not traversable for planning; it becomes known once seen
    return (v >= 0) and (v != CellType.UNKNOWN.value)

class WorldModel:
    def __init__(self, shape: tuple[int, int]) -> None:
        H, W = shape
        self.shape = shape
        self.known_map = np.full((H, W), CellType.UNKNOWN.value, dtype=int)
        self.visited_count = np.zeros((H, W), dtype=int)
        self.last_pos: Optional[Tuple[int,int]] = None

        # ascii dump bookkeeping
        self.turn = 0
        self.dump_file = "map_dump.txt"
        self._dump_initialized = False

    def updateWithObservation(self, st: State) -> None:
        if st.visible_raw is None: return
        raw = st.visible_raw
        seen = (raw != CellType.NOT_VISIBLE.value)
        self.known_map[seen] = raw[seen]

    def traversable(self, x: int, y: int) -> bool:
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W): return False
        return is_traversable_val(self.known_map[x, y])

# ────────────────────────────────────────────────────────────────────────────────
# Left-hand wall follower (no hand-flipping, deterministic)
# ────────────────────────────────────────────────────────────────────────────────

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W in grid coords

def left_of(d: Tuple[int,int]) -> Tuple[int,int]:
    dx, dy = d
    return (-dy, dx)

def right_of(d: Tuple[int,int]) -> Tuple[int,int]:
    dx, dy = d
    return (dy, -dx)

def back_of(d: Tuple[int,int]) -> Tuple[int,int]:
    dx, dy = d
    return (-dx, -dy)

class LeftWallPolicy:
    """
    Pure left-hand rule:
        - If LEFT is free → turn LEFT.
        - else if FRONT is free → go STRAIGHT.
        - else if RIGHT is free → turn RIGHT.
        - else → turn BACK.

    Corridor rule:
        - If walls on BOTH sides (left & right), prefer STRAIGHT and aim to accelerate.
    """
    def __init__(self, world: WorldModel) -> None:
        self.world = world
        self.heading: Tuple[int,int] = (0, 1)  # default EAST

    # local sensing helpers (use visible_* so it reflects current frame)
    def _is_wall_local(self, state: State, x: int, y: int) -> bool:
        if state.visible_raw is None: return False
        H, W = state.visible_raw.shape
        if not (0 <= x < H and 0 <= y < W): return False
        return state.visible_raw[x, y] == CellType.WALL.value

    def _is_free_local(self, state: State, x: int, y: int) -> bool:
        if state.visible_track is None: return False
        H, W = state.visible_track.shape
        if not (0 <= x < H and 0 <= y < W): return False
        return state.visible_track[x, y] >= 0

    def _ensure_heading(self, state: State) -> None:
        # If we have velocity, align heading with its dominant axis; else keep current.
        if state.agent is None: return
        vx, vy = int(state.agent.vel_x), int(state.agent.vel_y)
        if vx == 0 and vy == 0:
            return
        if abs(vx) >= abs(vy):
            self.heading = (int(np.sign(vx)), 0)
        else:
            self.heading = (0, int(np.sign(vy)))

    def next_grid_target(self, state: State) -> Tuple[Tuple[int,int], str]:
        assert state.agent is not None
        ax, ay = int(state.agent.x), int(state.agent.y)
        self._ensure_heading(state)

        dF = self.heading
        dL = left_of(dF)
        dR = right_of(dF)
        dB = back_of(dF)

        lx, ly = ax + dL[0], ay + dL[1]
        fx, fy = ax + dF[0], ay + dF[1]
        rx, ry = ax + dR[0], ay + dR[1]
        bx, by = ax + dB[0], ay + dB[1]

        left_wall  = self._is_wall_local(state, lx, ly)
        right_wall = self._is_wall_local(state, rx, ry)

        left_free   = self._is_free_local(state, lx, ly)
        front_free  = self._is_free_local(state, fx, fy)
        right_free  = self._is_free_local(state, rx, ry)
        back_free   = self._is_free_local(state, bx, by)

        # Corridor: walls on both sides → go straight if possible, and speed up.
        if left_wall and right_wall and front_free:
            # keep heading
            return (fx, fy), "corridor"

        # Left-hand rule strict
        if left_free:
            self.heading = dL
            return (lx, ly), "turn_left"
        if front_free:
            # keep heading
            return (fx, fy), "forward"
        if right_free:
            self.heading = dR
            return (rx, ry), "turn_right"
        if back_free:
            self.heading = dB
            return (bx, by), "turn_back"

        # No move known safe in local window — stay (should be rare with validLineLocal)
        return (ax, ay), "stuck"

# ────────────────────────────────────────────────────────────────────────────────
# Low-level driver: choose acceleration toward a grid target
# ────────────────────────────────────────────────────────────────────────────────

def choose_accel_toward_cell(state: State,
                             world: WorldModel,
                             policy: LeftWallPolicy,
                             target_cell: Tuple[int,int],
                             mode: str) -> Tuple[int,int]:
    assert state.agent is not None and state.visible_track is not None
    rSafe = max(0, state.circuit.visibility_radius - 1)

    p = state.agent.pos
    v = state.agent.vel
    vx, vy = int(v[0]), int(v[1])

    # Target speeds: push harder in corridors to stay centered by momentum
    max_safe = max(1.0, math.sqrt(2 * max(0, rSafe)))
    if mode == "corridor":
        target_speed = max_safe
    elif mode == "turn_back":
        target_speed = max(1.0, 0.5 * max_safe)
    else:
        target_speed = max(1.5, 0.7 * max_safe)

    to_cell = np.array([target_cell[0], target_cell[1]], dtype=float) - p.astype(float)
    n_to = float(np.linalg.norm(to_cell)) or 1.0
    desired_dir = to_cell / n_to

    best = None  # (score, (ax,ay))

    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            nvx, nvy = vx + ax, vy + ay
            if not brakingOk(nvx, nvy, rSafe):
                continue
            next_pos = p + v + np.array([ax, ay], dtype=int)
            nx, ny = int(next_pos[0]), int(next_pos[1])

            if not validLineLocal(state, p, next_pos):
                continue
            if any(np.all(next_pos == q.pos) for q in state.players):
                continue

            dist_cell = float(np.linalg.norm(next_pos.astype(float) - np.array(target_cell, dtype=float)))
            speed_next = float(math.hypot(nvx, nvy))
            speed_pen = abs(speed_next - target_speed)

            heading_pen = 0.0
            if speed_next > 0.0:
                vel_dir = np.array([nvx, nvy], dtype=float) / max(speed_next, 1e-9)
                heading_pen = (1.0 - float(np.dot(vel_dir, desired_dir))) * 0.8

            visit_pen = 0.0
            if 0 <= nx < world.shape[0] and 0 <= ny < world.shape[1]:
                visit_pen = 0.15 * float(world.visited_count[nx, ny])

            score = (2.0 * dist_cell) + (0.9 * speed_pen) + heading_pen + visit_pen
            cand = (score, (ax, ay))
            if (best is None) or (cand[0] < best[0]):
                best = cand

    if best is not None:
        return best[1]

    # Fallbacks that preserve motion preference:
    # Try to keep moving in current heading if safe.
    hx, hy = policy.heading
    for ax, ay in ((np.sign(hx), np.sign(hy)), (0, 0)):
        nvx, nvy = vx + ax, vy + ay
        nxt = p + v + np.array([ax, ay], dtype=int)
        if brakingOk(nvx, nvy, rSafe) and validLineLocal(state, p, nxt):
            if not any(np.all(nxt == q.pos) for q in state.players):
                return int(ax), int(ay)

    return (0, 0)  # last resort

# ────────────────────────────────────────────────────────────────────────────────
# ASCII dump (writes to file map_dump.txt every turn)
# ────────────────────────────────────────────────────────────────────────────────

def dump_ascii(world: WorldModel, policy: LeftWallPolicy, state: State, mode: str) -> None:
    if state.agent is None:
        return
    H, W = world.shape
    km = world.known_map
    vis = world.visited_count

    grid = [['?' for _ in range(W)] for _ in range(H)]
    for x in range(H):
        for y in range(W):
            v = km[x, y]
            if v == CellType.WALL.value:
                grid[x][y] = '#'
            elif v == CellType.GOAL.value:
                grid[x][y] = 'G'
            elif v == CellType.START.value:
                grid[x][y] = 'S'
            elif v == CellType.EMPTY.value:
                grid[x][y] = '.'
            elif v == CellType.UNKNOWN.value:
                grid[x][y] = '?'

    for x in range(H):
        for y in range(W):
            if vis[x, y] > 0 and grid[x][y] not in ('#', 'G', 'S'):
                grid[x][y] = 'v'

    for p in state.players:
        if 0 <= p.x < H and 0 <= p.y < W:
            grid[p.x][p.y] = 'O'

    ax, ay = int(state.agent.x), int(state.agent.y)
    if 0 <= ax < H and 0 <= ay < W:
        grid[ax][ay] = 'A'

    hdr = []
    hdr.append(
        f"TURN {world.turn}  pos=({ax},{ay}) vel=({int(state.agent.vel_x)},{int(state.agent.vel_y)}) "
        f"mode={mode} heading={policy.heading}"
    )
    hdr.append("LEGEND: #=WALL  ?=UNKNOWN  .=EMPTY  G=GOAL  S=START  v=visited  O=other  A=agent")
    lines = ["\n".join(hdr)]
    for x in range(H):
        lines.append("".join(grid[x]))
    lines.append("")

    mode_flag = "a"
    if not world._dump_initialized:
        mode_flag = "w"
        world._dump_initialized = True
    with open(world.dump_file, mode_flag, encoding="utf-8") as f:
        f.write("\n".join(lines))

# ────────────────────────────────────────────────────────────────────────────────
# Decision loop
# ────────────────────────────────────────────────────────────────────────────────

def calculateMove(world: WorldModel, policy: LeftWallPolicy, state: State) -> Tuple[int,int]:
    assert state.agent is not None and state.visible_raw is not None
    world.updateWithObservation(state)

    ax, ay = int(state.agent.x), int(state.agent.y)
    world.visited_count[ax, ay] += 1

    target_cell, mode = policy.next_grid_target(state)
    ax_cmd, ay_cmd = choose_accel_toward_cell(state, world, policy, target_cell, mode)

    world.last_pos = (ax, ay)
    dump_ascii(world, policy, state, mode)
    world.turn += 1

    return ax_cmd, ay_cmd

# ────────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    print("READY", flush=True)
    circuit = read_initial_observation()
    state: Optional[State] = State(circuit, None, None, [], None)

    world = WorldModel(circuit.track_shape)
    policy = LeftWallPolicy(world)

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return

        ax, ay = calculateMove(world, policy, state)

        # clamp to judge-legal range
        ax = -1 if ax < -1 else (1 if ax > 1 else int(ax))
        ay = -1 if ay < -1 else (1 if ay > 1 else int(ay))
        print(f"{ax} {ay}", flush=True)

if __name__ == "__main__":
    main()
