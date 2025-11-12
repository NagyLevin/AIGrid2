import sys
import enum
import math
from collections import deque
from typing import Optional, NamedTuple, Tuple, List

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
    return (tri(abs(vx)) <= rSafe) and (tri(abs(vy)) <= rSafe)

def validLineLocal(state: State, p1: np.ndarray, p2: np.ndarray) -> bool:
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
# World model (TTL soft-blocking + corridor / wall-hug subgoals + ASCII)
# ────────────────────────────────────────────────────────────────────────────────

def is_traversable_val(v: int) -> bool:
    return (v >= 0) and (v != CellType.UNKNOWN.value)

class WorldModel:
    def __init__(self, shape: tuple[int, int]) -> None:
        H, W = shape
        self.shape = shape
        self.known_map = np.full((H, W), CellType.UNKNOWN.value, dtype=int)
        self.visited_count = np.zeros((H, W), dtype=int)

        self.block_ttl = np.zeros((H, W), dtype=np.int16)  # TTL > 0 means "soft-blocked"
        self.BLOCK_TTL_DEFAULT = 40

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

    def is_blocked(self, x: int, y: int) -> bool:
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W): return True
        return self.block_ttl[x, y] > 0

    def decay_blocks(self) -> None:
        bt = self.block_ttl
        pos = bt > 0
        bt[pos] -= 1

    # ---- judge-wall queries (DO NOT count soft-blocks as walls) ----
    def _is_judge_wall(self, x: int, y: int) -> bool:
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W): return True
        return self.known_map[x, y] == CellType.WALL.value

    def _free_known(self, x: int, y: int) -> bool:
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W): return False
        v = self.known_map[x, y]
        return v == CellType.EMPTY.value or v == CellType.START.value or v == CellType.GOAL.value

    def is_corridor_cell(self, x: int, y: int) -> bool:
        if not self._free_known(x, y):
            return False
        N = self._is_judge_wall(x-1, y)
        S = self._is_judge_wall(x+1, y)
        W = self._is_judge_wall(x, y-1)
        E = self._is_judge_wall(x, y+1)
        ns_corr = N and S and (not W) and (not E)
        we_corr = W and E and (not N) and (not S)
        return ns_corr or we_corr

    def is_wall_adjacent(self, x: int, y: int) -> bool:
        return (self._is_judge_wall(x-1, y) or self._is_judge_wall(x+1, y) or
                self._is_judge_wall(x, y-1) or self._is_judge_wall(x, y+1))

    def bfs_to_nearest_corridor(self, start: Tuple[int,int], forbid_first: Optional[Tuple[int,int]]) -> Optional[List[Tuple[int,int]]]:
        sx, sy = start
        H, W = self.shape
        if not (0 <= sx < H and 0 <= sy < W): return None
        if not self.traversable(sx, sy): return None

        q = deque([(sx, sy)])
        prev = {(sx, sy): None}
        while q:
            x, y = q.popleft()
            if self.is_corridor_cell(x, y):
                path = []
                cur = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                if forbid_first and len(path) >= 2 and path[1] == forbid_first:
                    pass
                else:
                    return path
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = x+dx, y+dy
                if not (0 <= nx < H and 0 <= ny < W): continue
                if (nx, ny) in prev: continue
                if not self.traversable(nx, ny): continue
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
        return None

    def bfs_to_nearest_wallhug(self, start: Tuple[int,int], forbid_first: Optional[Tuple[int,int]]) -> Optional[List[Tuple[int,int]]]:
        sx, sy = start
        H, W = self.shape
        if not (0 <= sx < H and 0 <= sy < W): return None
        if not self.traversable(sx, sy): return None

        def search(allow_blocked: bool) -> Optional[List[Tuple[int,int]]]:
            q = deque([(sx, sy)])
            prev = {(sx, sy): None}
            while q:
                x, y = q.popleft()
                if self.is_wall_adjacent(x, y):
                    path = []
                    cur = (x, y)
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    path.reverse()
                    if forbid_first and len(path) >= 2 and path[1] == forbid_first:
                        pass
                    else:
                        return path
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx, ny = x+dx, y+dy
                    if not (0 <= nx < H and 0 <= ny < W): continue
                    if (nx, ny) in prev: continue
                    if not self.traversable(nx, ny): continue
                    if (not allow_blocked) and self.is_blocked(nx, ny):
                        continue
                    prev[(nx, ny)] = (x, y)
                    q.append((nx, ny))
            return None

        path = search(allow_blocked=False)
        if path is None:
            path = search(allow_blocked=True)
        return path

# ────────────────────────────────────────────────────────────────────────────────
# Left/right adaptive wall follower with corridor & wall-hug subgoals + anti-backstep
# ────────────────────────────────────────────────────────────────────────────────

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W

def left_of(d: Tuple[int,int]) -> Tuple[int,int]:
    dx, dy = d
    return (-dy, dx)

def right_of(d: Tuple[int,int]) -> Tuple[int,int]:
    dx, dy = d
    return (dy, -dx)

def back_of(d: Tuple[int,int]) -> Tuple[int,int]:
    dx, dy = d
    return (-dx, -dy)

class WallPolicy:
    """
    Priorities:
      1) Subgoal: go to a **corridor** (judge walls on both sides).
      2) Subgoal: if none, go to the **nearest judge-wall-adjacent** cell (to start/keep wall-follow).
      3) Local wall-follow: prefer **LEFT**; if no left wall but **RIGHT** wall exists, hug **RIGHT**.
      4) Never step **back** to last_pos unless dead end. Respect soft-blocks; use them only when stuck.

    We ignore other players entirely.
    """
    def __init__(self, world: WorldModel) -> None:
        self.world = world
        self.heading: Tuple[int,int] = (0, 1)  # default EAST
        self.last_mode: str = "init"

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
        if state.agent is None: return
        vx, vy = int(state.agent.vel_x), int(state.agent.vel_y)
        if vx == 0 and vy == 0:
            return
        if abs(vx) >= abs(vy):
            self.heading = (int(np.sign(vx)), 0)
        else:
            self.heading = (0, int(np.sign(vy)))

    def _choose_by_hand(self, state: State) -> Tuple[Tuple[int,int], str]:
        assert state.agent is not None
        ax, ay = int(state.agent.x), int(state.agent.y)
        last = self.world.last_pos

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

        w = self.world
        left_unblocked  = left_free  and (not w.is_blocked(lx, ly))
        front_unblocked = front_free and (not w.is_blocked(fx, fy))
        right_unblocked = right_free and (not w.is_blocked(rx, ry))
        back_unblocked  = back_free  and (not w.is_blocked(bx, by))

        def not_back(cell):
            return (last is None) or (cell != last)

        left_ok   = left_unblocked  and not_back((lx, ly))
        front_ok  = front_unblocked and not_back((fx, fy))
        right_ok  = right_unblocked and not_back((rx, ry))

        dead_end = (not left_ok) and (not front_ok) and (not right_ok) and back_unblocked

        # Corridor straight rule (judge walls on both sides)
        if left_wall and right_wall and front_free and front_unblocked and not_back((fx, fy)):
            return (fx, fy), "corridor"

        # Adaptive: prefer left; if no left wall but right wall exists, hug right
        if (not left_wall) and right_wall:
            if right_ok:
                self.heading = dR; return (rx, ry), "turn_right_adapt"
            if front_ok:
                return (fx, fy), "forward_adapt"
            if left_ok:
                self.heading = dL; return (lx, ly), "turn_left_adapt"
        else:
            if left_ok:
                self.heading = dL; return (lx, ly), "turn_left"
            if front_ok:
                return (fx, fy), "forward"
            if right_ok:
                self.heading = dR; return (rx, ry), "turn_right"

        # Allow back only in true dead ends
        if dead_end:
            self.heading = dB
            return (bx, by), "dead_end_back"

        # Forced (blocked) but avoid back if possible
        left_forced  = left_free  and not_back((lx, ly))
        front_forced = front_free and not_back((fx, fy))
        right_forced = right_free and not_back((rx, ry))

        if left_forced:
            self.heading = dL; return (lx, ly), "forced_left"
        if front_forced:
            return (fx, fy), "forced_forward"
        if right_forced:
            self.heading = dR; return (rx, ry), "forced_right"

        if back_free:
            self.heading = dB; return (bx, by), "forced_back"

        return (ax, ay), "stuck"

    def next_target(self, state: State, world: WorldModel) -> Tuple[Tuple[int,int], str]:
        self._ensure_heading(state)
        ax, ay = int(state.agent.x), int(state.agent.y)

        # 1) Corridor subgoal (primary)
        path = world.bfs_to_nearest_corridor((ax, ay), forbid_first=world.last_pos)
        if path and len(path) >= 2:
            self.last_mode = "subgoal_corridor"
            return path[1], "subgoal_corridor"

        # 2) Wall-hug subgoal (secondary): seek any judge-wall-adjacent cell
        path2 = world.bfs_to_nearest_wallhug((ax, ay), forbid_first=world.last_pos)
        if path2 and len(path2) >= 2:
            self.last_mode = "subgoal_wallhug"
            return path2[1], "subgoal_wallhug"

        # 3) Local wall-follow with anti-backstep & soft-block preference
        tgt, mode = self._choose_by_hand(state)
        self.last_mode = mode
        return tgt, mode

# ────────────────────────────────────────────────────────────────────────────────
# Low-level driver (players ignored) with hard speed cap and corridor no-accel
# ────────────────────────────────────────────────────────────────────────────────

def choose_accel_toward_cell(state: State,
                             world: WorldModel,
                             policy: WallPolicy,
                             target_cell: Tuple[int,int],
                             mode: str) -> Tuple[int,int]:
    assert state.agent is not None and state.visible_track is not None
    R = state.circuit.visibility_radius
    rSafe = max(0, R - 1)

    p = state.agent.pos
    v = state.agent.vel
    vx, vy = int(v[0]), int(v[1])
    cur_speed = float(math.hypot(vx, vy))

    # ---- HARD SPEED CAP: never exceed half the render distance
    half_R = max(1.0, 0.5 * float(R))

    # target speed per mode, but never above half_R
    if mode in ("corridor", "subgoal_corridor"):
        # do NOT accelerate in corridors: keep or reduce, but cap at half_R
        target_speed = min(cur_speed, half_R)
    elif "back" in mode:
        target_speed = min(half_R, max(1.0, 0.5 * half_R))
    else:
        # normal/forced turns: free to adjust, but still capped by half_R
        target_speed = min(half_R, max(1.5, 0.7 * half_R))

    to_cell = np.array([target_cell[0], target_cell[1]], dtype=float) - p.astype(float)
    n_to = float(np.linalg.norm(to_cell)) or 1.0
    desired_dir = to_cell / n_to

    best = None  # (score, (ax,ay))
    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            nvx, nvy = vx + ax, vy + ay

            # braking safety
            if not brakingOk(nvx, nvy, rSafe):
                continue

            next_pos = p + v + np.array([ax, ay], dtype=int)
            nx, ny = int(next_pos[0]), int(next_pos[1])

            if not validLineLocal(state, p, next_pos):
                continue

            next_speed = float(math.hypot(nvx, nvy))

            # HARD CAP: reject candidates exceeding R/2
            if next_speed > half_R + 1e-9:
                continue

            # In corridors, reject any acceleration (next_speed > cur_speed)
            if mode in ("corridor", "subgoal_corridor") and next_speed > cur_speed + 1e-9:
                continue

            dist_cell = float(np.linalg.norm(next_pos.astype(float) - np.array(target_cell, dtype=float)))
            speed_pen = abs(next_speed - target_speed)

            heading_pen = 0.0
            if next_speed > 0.0:
                vel_dir = np.array([nvx, nvy], dtype=float) / max(next_speed, 1e-9)
                heading_pen = (1.0 - float(np.dot(vel_dir, desired_dir))) * 0.8

            visit_pen = 0.0
            if 0 <= nx < world.shape[0] and 0 <= ny < world.shape[1]:
                visit_pen = 0.15 * float(world.visited_count[nx, ny])

            score = (2.0 * dist_cell) + (0.9 * speed_pen) + heading_pen + visit_pen
            if (best is None) or (score < best[0]):
                best = (score, (ax, ay))

    if best is not None:
        return best[1]

    # Fallbacks preserving motion: enforce caps here too
    hx, hy = policy.heading
    for ax, ay in ((int(np.sign(hx)), int(np.sign(hy))), (0, 0)):
        nvx, nvy = vx + ax, vy + ay
        next_speed = float(math.hypot(nvx, nvy))
        nxt = p + v + np.array([ax, ay], dtype=int)
        if (brakingOk(nvx, nvy, rSafe)
            and validLineLocal(state, p, nxt)
            and next_speed <= half_R + 1e-9
            and not (mode in ("corridor", "subgoal_corridor") and next_speed > cur_speed + 1e-9)):
            return int(ax), int(ay)

    return (0, 0)

# ────────────────────────────────────────────────────────────────────────────────
# ASCII dump
# ────────────────────────────────────────────────────────────────────────────────

def dump_ascii(world: WorldModel, policy: WallPolicy, state: State, mode: str) -> None:
    if state.agent is None:
        return
    H, W = world.shape
    km = world.known_map
    vis = world.visited_count
    blk = world.block_ttl > 0

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

    bx, by = np.where(blk)
    for i in range(len(bx)):
        x, y = int(bx[i]), int(by[i])
        if grid[x][y] not in ('#', 'G', 'S'):
            grid[x][y] = 'X'

    for p in state.players:
        if 0 <= p.x < H and 0 <= p.y < W:
            grid[p.x][p.y] = 'O'

    ax, ay = int(state.agent.x), int(state.agent.y)
    if 0 <= ax < H and 0 <= ay < W:
        grid[ax][ay] = 'A'

    hdr = []
    hdr.append(
        f"TURN {world.turn}  pos=({ax},{ay}) vel=({int(state.agent.vel_x)},{int(state.agent.vel_y)}) "
        f"mode={mode} heading={policy.heading} last_pos={world.last_pos}"
    )
    hdr.append("LEGEND: #=WALL  ?=UNKNOWN  .=EMPTY  G=GOAL  S=START  v=visited  X=soft-block  O=other  A=agent  (speed cap=R/2)")
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

def calculateMove(world: WorldModel, policy: WallPolicy, state: State) -> Tuple[int,int]:
    assert state.agent is not None and state.visible_raw is not None

    world.decay_blocks()
    world.updateWithObservation(state)

    curr = (int(state.agent.x), int(state.agent.y))
    if world.last_pos is not None and world.last_pos != curr:
        px, py = world.last_pos
        world.block_ttl[px, py] = world.BLOCK_TTL_DEFAULT

    ax, ay = curr
    world.visited_count[ax, ay] += 1

    target_cell, mode = policy.next_target(state, world)
    ax_cmd, ay_cmd = choose_accel_toward_cell(state, world, policy, target_cell, mode)

    world.last_pos = curr
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
    policy = WallPolicy(world)

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return

        ax, ay = calculateMove(world, policy, state)

        ax = -1 if ax < -1 else (1 if ax > 1 else int(ax))
        ay = -1 if ay < -1 else (1 if ay > 1 else int(ay))
        print(f"{ax} {ay}", flush=True)

if __name__ == "__main__":
    main()
