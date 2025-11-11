import sys
import enum
import heapq
import numpy as np
from collections import deque, defaultdict
from typing import Optional, NamedTuple, Tuple, List, Dict, Set

# ================= Types =================

class CellType(enum.Enum):
    NOT_VISIBLE = 3
    WALL = -1
    EMPTY = 0
    START = 1
    GOAL = 100

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
    visible_track: Optional[np.ndarray]
    players: list[Player]
    agent: Player

# ================ I/O ====================

def read_initial_observation() -> Circuit:
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)

# ============== Memory ===================

class Memory:
    def __init__(self):
        self.initialised = False
        self.world: Optional[np.ndarray] = None      # stitched raw map
        self.safety: Optional[np.ndarray] = None     # unknown -> WALL for planning/LOS
        self.visit_count: Dict[Tuple[int,int], int] = defaultdict(int)

        # Subgoal/route
        self.subgoal: Optional[Tuple[int,int]] = None
        self.last_pos: Optional[Tuple[int,int]] = None

        # Heuristic table for real-time search (learned costs)
        self.h: Dict[Tuple[int,int], float] = {}

    def ensure(self, circuit: Circuit):
        if not self.initialised:
            H, W = circuit.track_shape
            self.world = np.full((H, W), CellType.NOT_VISIBLE.value, dtype=int)
            self.safety = np.full((H, W), CellType.WALL.value, dtype=int)
            self.initialised = True

    def overlay_window(self, posx: int, posy: int, R: int, window_rows: List[List[int]]):
        assert self.world is not None and self.safety is not None
        H, W = self.world.shape
        for i in range(2*R + 1):
            gx = posx - R + i
            if not (0 <= gx < H): 
                continue
            row = window_rows[i]
            gy_start = posy - R
            for j, val in enumerate(row):
                gy = gy_start + j
                if 0 <= gy < W:
                    self.world[gx, gy] = val
                    # Unknown is treated unsafe for movement until seen:
                    self.safety[gx, gy] = (CellType.WALL.value
                                           if val == CellType.NOT_VISIBLE.value
                                           else val)

    def mark_visit(self, p: Tuple[int,int]):
        self.visit_count[p] += 1

MEM = Memory()

def read_observation(old_state: State) -> Optional[State]:
    line = input()
    if line == '~~~END~~~':
        return None
    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)

    players: List[Player] = []
    cd = old_state.circuit
    for _ in range(cd.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    window_rows: List[List[int]] = []
    for _ in range(2 * cd.visibility_radius + 1):
        window_rows.append([int(a) for a in input().split()])

    MEM.overlay_window(posx, posy, cd.visibility_radius, window_rows)

    # Full-size "visible" (for compatibility/debug)
    H, W = cd.track_shape
    R = cd.visibility_radius
    visible_track = np.full((H, W), CellType.NOT_VISIBLE.value, dtype=int)
    for i in range(2*R + 1):
        gx = posx - R + i
        if 0 <= gx < H:
            row = window_rows[i]
            gy_start = posy - R
            for j, val in enumerate(row):
                gy = gy_start + j
                if 0 <= gy < W:
                    visible_track[gx, gy] = val

    return old_state._replace(visible_track=visible_track, players=players, agent=agent)

# ============= Geometry / LOS ==============

def clamp_idx(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v

def valid_line_on_map(track_int: np.ndarray, pos1: np.ndarray, pos2: np.ndarray) -> bool:
    # Same robust LOS used before (blocks if the straight segment crosses a solid wall band).
    if (np.any(pos1 < 0) or np.any(pos2 < 0)
        or np.any(pos1 >= track_int.shape) or np.any(pos2 >= track_int.shape)):
        return False
    diff = pos2 - pos1
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))
        for i in range(abs(int(diff[0])) + 1):
            x = pos1[0] + i*d
            y = pos1[1] + i*slope*d
            y_ceil  = clamp_idx(int(np.ceil(y)),  0, track_int.shape[1]-1)
            y_floor = clamp_idx(int(np.floor(y)), 0, track_int.shape[1]-1)
            if track_int[x, y_ceil] < 0 and track_int[x, y_floor] < 0:
                return False
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(int(diff[1])) + 1):
            x = pos1[0] + i*slope*d
            y = pos1[1] + i*d
            x_ceil  = clamp_idx(int(np.ceil(x)),  0, track_int.shape[0]-1)
            x_floor = clamp_idx(int(np.floor(x)), 0, track_int.shape[0]-1)
            if track_int[x_ceil, y] < 0 and track_int[x_floor, y] < 0:
                return False
    return True

# ============ Grid helpers =================

def in_bounds(H: int, W: int, p: Tuple[int,int]) -> bool:
    x, y = p
    return 0 <= x < H and 0 <= y < W

def neighbors4(H: int, W: int, p: Tuple[int,int]):
    x, y = p
    if x > 0:         yield (x-1, y)
    if x+1 < H:       yield (x+1, y)
    if y > 0:         yield (x, y-1)
    if y+1 < W:       yield (x, y+1)

def free_cell(x: int, y: int) -> bool:
    assert MEM.safety is not None
    H, W = MEM.safety.shape
    if not (0 <= x < H and 0 <= y < W): return False
    return MEM.safety[x, y] >= 0

# ========= Subgoals: GOAL or Frontier ========

def frontier_cells() -> Set[Tuple[int,int]]:
    """Frontiers = free cells that border at least one NOT_VISIBLE neighbor."""
    assert MEM.world is not None and MEM.safety is not None
    H, W = MEM.world.shape
    out: Set[Tuple[int,int]] = set()
    for x in range(H):
        for y in range(W):
            if MEM.safety[x, y] < 0: 
                continue
            for nx, ny in neighbors4(H, W, (x, y)):
                if MEM.world[nx, ny] == CellType.NOT_VISIBLE.value:
                    out.add((x, y))
                    break
    return out

def choose_subgoal(cur: Tuple[int,int]) -> Optional[Tuple[int,int]]:
    """Closest GOAL if known; else nearest frontier (Manhattan)."""
    assert MEM.world is not None and MEM.safety is not None
    goals = list(map(tuple, np.argwhere(MEM.world == CellType.GOAL.value)))
    if goals:
        # nearest goal by manhattan
        return min(goals, key=lambda g: abs(g[0]-cur[0]) + abs(g[1]-cur[1]))
    F = frontier_cells()
    if not F:
        return None
    return min(F, key=lambda f: abs(f[0]-cur[0]) + abs(f[1]-cur[1]))

# ======== A* utilities (bounded lookahead) ========

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def bounded_astar(start: Tuple[int,int], goal: Tuple[int,int], max_expand: int) -> Tuple[Optional[List[Tuple[int,int]]], Set[Tuple[int,int]], Dict[Tuple[int,int], int], Dict[Tuple[int,int], Tuple[int,int]]]:
    """A* from start to single goal, with expansion cap. Returns (path or None, closed, g, parent)."""
    assert MEM.safety is not None
    H, W = MEM.safety.shape
    g: Dict[Tuple[int,int], int] = {start: 0}
    parent: Dict[Tuple[int,int], Tuple[int,int]] = {}
    closed: Set[Tuple[int,int]] = set()
    tie = 0
    pq: List[Tuple[int,int,Tuple[int,int]]] = []
    heapq.heappush(pq, (manhattan(start, goal), tie, start))
    expanded = 0
    while pq and expanded < max_expand:
        _, _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == goal:
            # reconstruct
            path = [u]
            while u in parent:
                u = parent[u]
                path.append(u)
            path.reverse()
            return path, closed, g, parent
        ux, uy = u
        for v in neighbors4(H, W, u):
            vx, vy = v
            if not in_bounds(H, W, v): continue
            if MEM.safety[vx, vy] < 0: continue
            ng = g[u] + 1
            if v not in g or ng < g[v]:
                g[v] = ng
                parent[v] = u
                tie += 1
                heapq.heappush(pq, (ng + manhattan(v, goal), tie, v))
    return None, closed, g, parent

# ======== LSS-LRTA*-style step (Koenig & Sun / RTAA* family) ========
# We do: bounded A* lookahead toward the current subgoal.
# If goal not found, update local heuristics for CLOSED states and move greedily to neighbor with min (1 + h).

def ensure_h(cur_goal: Tuple[int,int]):
    """Initialize heuristic lazily: Manhattan to the current subgoal when missing."""
    if MEM.h.get(cur_goal) is None:
        MEM.h[cur_goal] = 0.0

def local_heuristic(p: Tuple[int,int], goal: Tuple[int,int]) -> float:
    # Learned h if available; else admissible base = Manhattan
    return MEM.h.get(p, float(manhattan(p, goal)))

def lss_lrta_step(cur: Tuple[int,int], goal: Tuple[int,int], lookahead: int = 600) -> Tuple[Tuple[int,int], bool]:
    """
    Returns (next_cell, goal_reached_flag).
    If the bounded A* finds the full path, we follow its first step.
    Otherwise, update heuristics on CLOSED and choose best neighbor by (1 + h).
    """
    ensure_h(goal)
    path, closed, gvals, parent = bounded_astar(cur, goal, lookahead)

    if path and len(path) >= 2:
        return path[1], (path[-1] == goal)

    # Update heuristics on CLOSED (one-step backup toward the best neighbor),
    # classic LRTA* backup over neighbors; this is simple and robust.
    assert MEM.safety is not None
    H, W = MEM.safety.shape
    for s in closed:
        sx, sy = s
        best = float('inf')
        for v in neighbors4(H, W, s):
            vx, vy = v
            if not in_bounds(H, W, v): continue
            if MEM.safety[vx, vy] < 0: continue
            cand = 1.0 + local_heuristic(v, goal)
            if cand < best: best = cand
        if best < float('inf'):
            old = MEM.h.get(s, float(manhattan(s, goal)))
            # Heuristic never decreases; learn upwards to avoid ping-pong
            MEM.h[s] = max(old, best)

    # Choose neighbor of current that minimizes (1 + h(nei)) (tie-break by Manhattan)
    best_n = None
    best_key = (float('inf'), float('inf'))
    for v in neighbors4(H, W, cur):
        if not free_cell(v[0], v[1]): continue
        cost = 1.0 + local_heuristic(v, goal)
        key = (cost, manhattan(v, goal))
        # small tabu to avoid immediate reversal unless strictly better
        if MEM.last_pos is not None and v == MEM.last_pos:
            key = (key[0] + 0.01, key[1] + 1)  # slight penalty
        if key < best_key:
            best_key = key
            best_n = v

    if best_n is None:
        # No move? stay put; controller will brake.
        return cur, False
    return best_n, False

# ========= Unit-step motion (|vel|≤1) =========

ACCELS = [(ax, ay) for ax in (-1,0,1) for ay in (-1,0,1)]

def unitstep_allowed(state: State, new_vel: np.ndarray) -> bool:
    assert MEM.safety is not None
    H, W = MEM.safety.shape
    self_pos = state.agent.pos.astype(int)
    new_pos = self_pos + new_vel
    if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= H or new_pos[1] >= W:
        return False
    if MEM.safety[new_pos[0], new_pos[1]] < 0:
        return False
    if not valid_line_on_map(MEM.safety, self_pos, new_pos):
        return False
    for p in state.players:
        if np.array_equal(new_pos, p.pos):
            return False
    return True

def accel_to_unit(state: State, desired_step: np.ndarray) -> Tuple[int,int]:
    cur_vel = state.agent.vel.astype(int)
    # If overspeed, brake first
    if abs(cur_vel[0]) > 1 or abs(cur_vel[1]) > 1:
        b = -np.sign(cur_vel)
        return int(np.clip(b[0], -1, 1)), int(np.clip(b[1], -1, 1))
    desired = np.clip(desired_step, -1, 1).astype(int)
    a = np.clip(desired - cur_vel, -1, 1).astype(int)
    nv = np.clip(cur_vel + a, -1, 1)
    if unitstep_allowed(state, nv):
        return int(a[0]), int(a[1])
    # try braking
    b = -np.sign(cur_vel)
    nv2 = np.clip(cur_vel + b, -1, 1)
    if unitstep_allowed(state, nv2):
        return int(b[0]), int(b[1])
    # try any safe unit accel
    for ax, ay in ACCELS:
        nv3 = np.clip(cur_vel + np.array([ax, ay], int), -1, 1)
        if unitstep_allowed(state, nv3):
            return ax, ay
    return 0, 0

# =============== High level =================

def decide_next_cell(state: State) -> Tuple[int,int]:
    """Pick or refresh a subgoal (GOAL else frontier) and move one LSS-LRTA* step toward it."""
    cur = tuple(state.agent.pos.tolist())
    # Refresh subgoal if missing or reached/unreachable
    sg = MEM.subgoal
    if sg is None or sg == cur:
        sg = choose_subgoal(cur)
        MEM.subgoal = sg
        # Reset learned heuristic map on big subgoal change if desired
        # (we keep learning across steps; admissibility isn't required here)
    if sg is None:
        # No subgoals anymore; stay
        return cur

    nxt, reached = lss_lrta_step(cur, sg, lookahead=800)
    if reached or nxt == sg:
        # subgoal reached or within one step; clear so we pick a new one next tick
        MEM.subgoal = None
    return nxt

def calculate_move(state: State) -> tuple[int, int]:
    cur = tuple(state.agent.pos.tolist())
    MEM.mark_visit(cur)

    # Decide target cell
    target_cell = decide_next_cell(state)
    cur_pos = state.agent.pos.astype(int)
    diff = np.array(target_cell, int) - cur_pos

    # Enforce single 4-neighbor step
    if abs(diff[0]) + abs(diff[1]) > 1:
        if abs(diff[0]) >= abs(diff[1]): diff = np.array([np.sign(diff[0]), 0], int)
        else: diff = np.array([0, np.sign(diff[1])], int)

    ax, ay = accel_to_unit(state, diff)

    # Predict next cell to keep a small tabu against instant reversal
    new_vel = np.clip(state.agent.vel.astype(int) + np.array([ax, ay], int), -1, 1)
    new_pos = tuple((cur_pos + new_vel).tolist())
    MEM.last_pos = cur  # remember where we came from

    return int(ax), int(ay)

# ================ Main =====================

def main():
    print('READY', flush=True)
    circuit = read_initial_observation()
    MEM.ensure(circuit)
    state: Optional[State] = State(circuit, None, [], None)  # type: ignore
    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return
        dx, dy = calculate_move(state)
        print(f'{dx} {dy}', flush=True)

if __name__ == "__main__":
    main()
