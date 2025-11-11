import sys
import enum
import numpy as np
from collections import deque, defaultdict
from typing import Optional, NamedTuple, Tuple, List, Dict, Set

# ---------------- Types ----------------

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

# -------------- I/O --------------------

def read_initial_observation() -> Circuit:
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)

# -------------- Memory -----------------

class Memory:
    def __init__(self):
        self.initialised = False
        self.world: Optional[np.ndarray] = None     # stitched map (raw values)
        self.safety: Optional[np.ndarray] = None    # unknown->walls
        self.visited: Set[Tuple[int,int]] = set()   # visited cells (blocked for forward)
        self.visit_count: Dict[Tuple[int,int], int] = defaultdict(int)
        self.speed_cap_axis: int = 1                # per-axis speed cap
        self.path: List[Tuple[int,int]] = []        # current planned path (grid cells)

    def ensure(self, circuit: Circuit):
        if not self.initialised:
            H, W = circuit.track_shape
            self.world = np.full((H, W), CellType.NOT_VISIBLE.value, dtype=int)
            self.safety = np.full((H, W), CellType.WALL.value, dtype=int)
            self.speed_cap_axis = max(1, circuit.visibility_radius // 2)
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
                    self.safety[gx, gy] = (CellType.WALL.value
                                           if val == CellType.NOT_VISIBLE.value
                                           else val)

    def mark_visit(self, p: Tuple[int,int]):
        self.visited.add(p)
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

    # build full-size visible (compat only)
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

# -------------- Geometry ----------------

def clamp_idx(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v

def valid_line_on_map(track_int: np.ndarray, pos1: np.ndarray, pos2: np.ndarray) -> bool:
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
            y_ceil = clamp_idx(int(np.ceil(y)), 0, track_int.shape[1]-1)
            y_floor = clamp_idx(int(np.floor(y)), 0, track_int.shape[1]-1)
            if track_int[x, y_ceil] < 0 and track_int[x, y_floor] < 0:
                return False
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(int(diff[1])) + 1):
            x = pos1[0] + i*slope*d
            y = pos1[1] + i*d
            x_ceil = clamp_idx(int(np.ceil(x)), 0, track_int.shape[0]-1)
            x_floor = clamp_idx(int(np.floor(x)), 0, track_int.shape[0]-1)
            if track_int[x_ceil, y] < 0 and track_int[x_floor, y] < 0:
                return False
    return True

# -------------- Grid utilities ----------------

def neighbors4(H: int, W: int, p: Tuple[int,int]):
    x, y = p
    if x > 0:         yield (x-1, y)
    if x+1 < H:       yield (x+1, y)
    if y > 0:         yield (x, y-1)
    if y+1 < W:       yield (x, y+1)

def forward_neighbors(cur: Tuple[int,int]) -> List[Tuple[int,int]]:
    """Unvisited, traversable immediate neighbors (visited-as-walls)."""
    assert MEM.world is not None and MEM.safety is not None
    H, W = MEM.world.shape
    res = []
    for n in neighbors4(H, W, cur):
        x, y = n
        if 0 <= x < H and 0 <= y < W and MEM.safety[x, y] >= 0 and (x, y) not in MEM.visited:
            res.append(n)
    # prefer neighbors that border unknown, then deterministic tie-break
    def unk_score(p):
        s = 0
        x, y = p
        for nx, ny in neighbors4(H, W, p):
            if MEM.world[nx, ny] == CellType.NOT_VISIBLE.value:
                s += 1
        return s
    res.sort(key=lambda p: (-unk_score(p), p[0], p[1]))
    return res

def has_unvisited_neighbor(p: Tuple[int,int]) -> bool:
    for n in forward_neighbors(p):
        return True
    return False

def bfs_path(start: Tuple[int,int], is_goal) -> Optional[List[Tuple[int,int]]]:
    """BFS on known traversable cells (visited ALLOWED) to nearest cell matching is_goal."""
    assert MEM.safety is not None
    H, W = MEM.safety.shape
    if MEM.safety[start[0], start[1]] < 0:
        return None
    q = deque([start])
    parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start: None}
    while q:
        u = q.popleft()
        if is_goal(u):
            # rebuild
            path = [u]
            v = u
            while parent[v] is not None:
                v = parent[v]
                path.append(v)
            path.reverse()
            return path
        ux, uy = u
        for v in neighbors4(H, W, u):
            vx, vy = v
            if not (0 <= vx < H and 0 <= vy < W):
                continue
            if MEM.safety[vx, vy] < 0:
                continue
            if v not in parent:
                parent[v] = u
                q.append(v)
    return None

def nearest_frontier_path(cur: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    """Find shortest path to the closest cell that has at least one unvisited traversable neighbor."""
    def is_frontier(u: Tuple[int,int]) -> bool:
        # Don't require u itself to be unvisited; only that from u we can step to some NEW cell.
        for n in forward_neighbors(u):
            return True
        return False
    return bfs_path(cur, is_frontier)

def path_to_goal(cur: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    assert MEM.world is not None
    goals = set(map(tuple, np.argwhere(MEM.world == CellType.GOAL.value)))
    if not goals:
        return None
    def is_goal(u: Tuple[int,int]) -> bool:
        return u in goals
    return bfs_path(cur, is_goal)

# -------------- Acceleration towards a single next cell ----------------

ACCELS = [(ax, ay) for ax in (-1,0,1) for ay in (-1,0,1)]

def choose_accel_to_cell(state: State, target_cell: Tuple[int,int]) -> Tuple[int,int]:
    assert MEM.safety is not None
    safety = MEM.safety
    H, W = safety.shape
    self_pos = state.agent.pos.astype(int)
    self_vel = state.agent.vel.astype(int)
    max_ax = MEM.speed_cap_axis
    others = [p.pos for p in state.players if not np.array_equal(p.pos, self_pos)]

    def allowed(a: Tuple[int,int]) -> bool:
        ax, ay = a
        if abs(ax) > 1 or abs(ay) > 1: return False
        new_vel = self_vel + np.array([ax, ay], dtype=int)
        if abs(new_vel[0]) > max_ax or abs(new_vel[1]) > max_ax:
            return False
        new_pos = self_pos + new_vel
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= H or new_pos[1] >= W:
            return False
        if not valid_line_on_map(safety, self_pos, new_pos):
            return False
        if any(np.array_equal(new_pos, p) for p in others):
            return False
        return True

    # try exact landing
    tgt = np.array(target_cell, dtype=int)
    desired_v = tgt - self_pos
    base_a = tuple(np.clip((desired_v - self_vel), -1, 1).astype(int).tolist())

    cands: List[Tuple[int,int]] = []
    for dx, dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
        a = (int(np.clip(base_a[0]+dx, -1, 1)), int(np.clip(base_a[1]+dy, -1, 1)))
        if a not in cands: cands.append(a)
    if (0,0) in cands:
        cands.remove((0,0))
        cands.append((0,0))

    for a in cands:
        if allowed(a):
            new_pos = self_pos + (self_vel + np.array(a, int))
            if int(new_pos[0]) == target_cell[0] and int(new_pos[1]) == target_cell[1]:
                return a

    # otherwise choose the allowed accel that gets closest (L1) with small speed penalty
    best_a, best_score = None, 1e9
    for a in ACCELS:
        if not allowed(a): continue
        new_vel = self_vel + np.array(a, int)
        new_pos = self_pos + new_vel
        dist = abs(int(new_pos[0])-target_cell[0]) + abs(int(new_pos[1])-target_cell[1])
        score = dist + 0.2*(abs(int(new_vel[0])) + abs(int(new_vel[1])))
        if score < best_score:
            best_score, best_a = score, a
    if best_a is not None:
        return best_a

    # braking fallback
    brake = tuple((-np.sign(self_vel)).astype(int).tolist())
    return brake if allowed(brake) else (0,0)

def brake_or_wait(state: State) -> Tuple[int,int]:
    self_vel = state.agent.vel.astype(int)
    return tuple((-np.sign(self_vel)).astype(int).tolist())

# -------------- High-level decision ----------------

def plan_path(state: State) -> List[Tuple[int,int]]:
    """Build a path [cur, next, ...]; we will only use the next step."""
    assert MEM.world is not None and MEM.safety is not None
    cur = tuple(state.agent.pos.tolist())

    # 1) If goal is known, shortest path to goal
    pg = path_to_goal(cur)
    if pg and len(pg) >= 2:
        return pg

    # 2) If we can go forward to an unvisited neighbor, take that immediate step (strict DFS forward)
    fwd = forward_neighbors(cur)
    if fwd:
        return [cur, fwd[0]]

    # 3) Otherwise, BFS to nearest frontier (cell that has some unvisited traversable neighbor)
    pf = nearest_frontier_path(cur)
    if pf and len(pf) >= 2:
        return pf

    # 4) Nowhere to go (fully explored known area, no goal seen) -> stop/brake
    return [cur]

def calculate_move(state: State) -> tuple[int, int]:
    MEM.mark_visit(tuple(state.agent.pos.tolist()))

    # keep path fresh and trimmed
    cur = tuple(state.agent.pos.tolist())
    if not MEM.path or cur not in MEM.path:
        MEM.path = plan_path(state)
    else:
        i = MEM.path.index(cur)
        MEM.path = MEM.path[i:]

    if len(MEM.path) <= 1:
        ax, ay = brake_or_wait(state)
        return (int(ax), int(ay))

    next_cell = MEM.path[1]
    ax, ay = choose_accel_to_cell(state, next_cell)
    return (int(ax), int(ay))

# -------------- Main -----------------------

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
