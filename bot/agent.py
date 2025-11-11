import sys
import enum
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
        self.world: Optional[np.ndarray] = None      # raw stitched map
        self.safety: Optional[np.ndarray] = None     # unknown -> WALL for LOS
        self.visited: Set[Tuple[int,int]] = set()    # visited cells (forward walls)
        self.dead_end: Set[Tuple[int,int]] = set()   # permanently avoided cells
        self.visit_count: Dict[Tuple[int,int], int] = defaultdict(int)

        self.spine: List[Tuple[int,int]] = []        # start..current path (DFS spine)
        self.subgoal: Optional[Tuple[int,int]] = None
        self.route: List[Tuple[int,int]] = []        # planned path to subgoal (cells)

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

    # keep a full-size compatible visible array (not used for planning)
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

# ============ Grid helpers =================

def neighbors4(H: int, W: int, p: Tuple[int,int]):
    x, y = p
    if x > 0:         yield (x-1, y)
    if x+1 < H:       yield (x+1, y)
    if y > 0:         yield (x, y-1)
    if y+1 < W:       yield (x, y+1)

def ensure_spine_sync(cur: Tuple[int,int]):
    if not MEM.spine:
        MEM.spine.append(cur)
        return
    if MEM.spine[-1] == cur:
        return
    if cur in MEM.spine:
        i = MEM.spine.index(cur)
        MEM.spine[:] = MEM.spine[:i+1]
    else:
        MEM.spine.append(cur)

# ======= Planning grids & frontiers =======

def build_forward_passable() -> np.ndarray:
    """Forward choices: safety>=0 and not visited/dead_end."""
    assert MEM.safety is not None
    passable = (MEM.safety >= 0).astype(np.int8)
    for (x, y) in MEM.visited:
        passable[x, y] = -1
    for (x, y) in MEM.dead_end:
        passable[x, y] = -1
    return passable

def build_passable_with_spine(cur: Tuple[int,int]) -> np.ndarray:
    """
    A* planning grid: safety>=0. Block visited/dead_end EXCEPT along DFS spine.
    Current cell is always free.
    """
    assert MEM.safety is not None
    passable = (MEM.safety >= 0).astype(np.int8)
    for (x, y) in MEM.visited:
        passable[x, y] = -1
    for (x, y) in MEM.dead_end:
        passable[x, y] = -1
    for v in MEM.spine:
        passable[v[0], v[1]] = 1
    passable[cur[0], cur[1]] = 1
    return passable

def frontier_cells(passable: np.ndarray) -> Set[Tuple[int,int]]:
    """Cells in 'passable' that border at least one NOT_VISIBLE tile."""
    assert MEM.world is not None
    H, W = passable.shape
    out: Set[Tuple[int,int]] = set()
    for x in range(H):
        for y in range(W):
            if passable[x, y] < 0:
                continue
            for nx, ny in neighbors4(H, W, (x, y)):
                if MEM.world[nx, ny] == CellType.NOT_VISIBLE.value:
                    out.add((x, y))
                    break
    return out

# ================== A* =====================

def a_star(start: Tuple[int,int], goals: Set[Tuple[int,int]], passable: np.ndarray,
           step_cost: Dict[Tuple[int,int], int] | None = None) -> Optional[List[Tuple[int,int]]]:
    if not goals:
        return None
    H, W = passable.shape
    sx, sy = start
    if not (0 <= sx < H and 0 <= sy < W): return None
    if passable[sx, sy] < 0: return None

    def h(p: Tuple[int,int]) -> int:
        # Manhattan to the nearest goal (sample up to 64)
        x, y = p
        dmin = 10**9
        cnt = 0
        for gx, gy in goals:
            d = abs(gx - x) + abs(gy - y)
            if d < dmin: dmin = d
            cnt += 1
            if cnt >= 64 and dmin <= 1: break
        return dmin

    import heapq
    g: Dict[Tuple[int,int], int] = {start: 0}
    parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start: None}
    heap: List[Tuple[int,int,Tuple[int,int]]] = []
    tie = 0
    heapq.heappush(heap, (h(start), tie, start))
    while heap:
        _, _, u = heapq.heappop(heap)
        if u in goals:
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
            if not (0 <= vx < H and 0 <= vy < W): continue
            if passable[vx, vy] < 0: continue
            base = 1
            if step_cost is not None:
                base = step_cost.get(v, base)
            ng = g[u] + base
            if v not in g or ng < g[v]:
                g[v] = ng
                parent[v] = u
                tie += 1
                heapq.heappush(heap, (ng + h(v), tie, v))
    return None

# ============== Candidate sets ==============

def forward_unvisited_neighbors(cur: Tuple[int,int]) -> List[Tuple[int,int]]:
    assert MEM.world is not None and MEM.safety is not None
    H, W = MEM.world.shape
    ret = []
    pfwd = build_forward_passable()
    for nx, ny in neighbors4(H, W, cur):
        if 0 <= nx < H and 0 <= ny < W and pfwd[nx, ny] >= 0:
            ret.append((nx, ny))
    # prefer neighbors that border lots of unknown
    def unk_score(p):
        s = 0
        x, y = p
        for ax, ay in neighbors4(H, W, p):
            if MEM.world[ax, ay] == CellType.NOT_VISIBLE.value:
                s += 1
        return s
    ret.sort(key=lambda p: (-unk_score(p), p[0], p[1]))
    return ret

# ========= Unit-step motion (|vel|<=1) =========

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

def accel_to_unit(state: State, desired_vel: np.ndarray) -> Tuple[int,int]:
    cur_vel = state.agent.vel.astype(int)
    # if overspeed, brake toward 0
    if abs(cur_vel[0]) > 1 or abs(cur_vel[1]) > 1:
        b = -np.sign(cur_vel)
        return int(np.clip(b[0], -1, 1)), int(np.clip(b[1], -1, 1))
    desired = np.clip(desired_vel, -1, 1).astype(int)
    a = np.clip(desired - cur_vel, -1, 1).astype(int)
    nv = np.clip(cur_vel + a, -1, 1)
    if unitstep_allowed(state, nv):
        return int(a[0]), int(a[1])
    # try braking
    b = -np.sign(cur_vel)
    nv2 = np.clip(cur_vel + b, -1, 1)
    if unitstep_allowed(state, nv2):
        return int(b[0]), int(b[1])
    # try anything safe within unit bounds
    for ax, ay in ACCELS:
        nv3 = np.clip(cur_vel + np.array([ax, ay], int), -1, 1)
        if unitstep_allowed(state, nv3):
            return ax, ay
    return 0, 0

# ============== Subgoal manager ==============

def ensure_spine_and_mark(state: State):
    cur = tuple(state.agent.pos.tolist())
    ensure_spine_sync(cur)
    MEM.mark_visit(cur)

def recompute_subgoal_and_route(state: State):
    """Pick a stable subgoal (GOAL else best frontier) and compute route to it with A*."""
    assert MEM.world is not None and MEM.safety is not None
    cur = tuple(state.agent.pos.tolist())

    # 1) GOAL known? Use GOAL with standard safety passability (visited allowed).
    goals = set(map(tuple, np.argwhere(MEM.world == CellType.GOAL.value)))
    if goals:
        passable = (MEM.safety >= 0).astype(np.int8)
        path = a_star(cur, goals, passable)
        if path and len(path) >= 2:
            MEM.subgoal = path[-1]
            MEM.route = path
            return

    # 2) Otherwise choose the best frontier under spine-only backtracking.
    passable = build_passable_with_spine(cur)
    frontiers = frontier_cells(passable)
    if not frontiers:
        # no frontiers visible → if we still have spine, step toward parent by planning
        path = a_star(cur, set(MEM.spine[:-1]), passable)
        if path and len(path) >= 2:
            MEM.subgoal = path[-1]
            MEM.route = path
        else:
            MEM.subgoal, MEM.route = None, []
        return

    # Rank frontiers by path length; tie-break by local unknown count (info gain)
    # Limit candidate evaluation for speed: keep the K closest by heuristic
    def h_est(p: Tuple[int,int]) -> int:
        return abs(p[0]-cur[0]) + abs(p[1]-cur[1])
    K = 64
    cand = sorted(list(frontiers), key=h_est)[:K]

    # Penalize using the spine (we allow it but make it slightly "expensive").
    step_cost: Dict[Tuple[int,int], int] = {}
    for v in MEM.spine:
        step_cost[v] = 3  # higher cost than unexplored (1)

    best = None
    best_key = (10**9, -1, 10**9)
    for tgt in cand:
        path = a_star(cur, {tgt}, passable, step_cost=step_cost)
        if not path: 
            continue
        plen = len(path)
        # info gain around target
        unk = 0
        for nx, ny in neighbors4(*MEM.world.shape, tgt):
            if MEM.world[nx, ny] == CellType.NOT_VISIBLE.value:
                unk += 1
        key = (plen, -unk, h_est(tgt))
        if key < best_key:
            best_key = key
            best = (tgt, path)

    if best is None:
        # fall back: any reachable cell (spine allowed) to keep moving
        all_ok = {(x,y) for x in range(MEM.safety.shape[0]) for y in range(MEM.safety.shape[1]) if passable[x,y]>=0}
        path = a_star(cur, all_ok, passable)
        if path and len(path) >= 2:
            MEM.subgoal = path[-1]
            MEM.route = path
        else:
            MEM.subgoal, MEM.route = None, []
        return

    MEM.subgoal, MEM.route = best[0], best[1]

def need_new_subgoal(state: State) -> bool:
    if MEM.subgoal is None or not MEM.route:
        return True
    cur = tuple(state.agent.pos.tolist())
    # If current not on route or subgoal reached → refresh
    if cur not in MEM.route:
        return True
    if cur == MEM.subgoal:
        return True
    return False

def next_step_from_route(state: State) -> Optional[Tuple[int,int]]:
    """Return the next cell toward the current subgoal; repair route if needed."""
    cur = tuple(state.agent.pos.tolist())
    if need_new_subgoal(state):
        recompute_subgoal_and_route(state)
    if not MEM.route or cur not in MEM.route:
        return None
    i = MEM.route.index(cur)
    if i+1 >= len(MEM.route):
        return None
    return MEM.route[i+1]

# =============== High level =================

def after_move_spine_bookkeeping(old: Tuple[int,int], new: Tuple[int,int]):
    """Maintain DFS spine and dead_end marking for 'only backtrack when necessary'."""
    # Forward step into an unvisited -> push
    if new not in MEM.spine:
        MEM.spine.append(new)
        return
    # If we stepped back to parent and old had no forward options, mark it dead_end and pop.
    if len(MEM.spine) >= 2 and new == MEM.spine[-2]:
        # Check forward options from 'old' respecting visited/dead_end as walls
        pfwd = build_forward_passable()
        has_new = False
        H, W = pfwd.shape
        x, y = old
        for nx, ny in neighbors4(H, W, (x, y)):
            if 0 <= nx < H and 0 <= ny < W and pfwd[nx, ny] >= 0:
                has_new = True
                break
        if not has_new:
            MEM.dead_end.add(old)
        MEM.spine.pop()  # drop old; new is now top

def calculate_move(state: State) -> tuple[int, int]:
    cur = tuple(state.agent.pos.tolist())
    ensure_spine_and_mark(state)

    # Clear stale subgoal if reached
    if MEM.subgoal is not None and cur == MEM.subgoal:
        MEM.subgoal, MEM.route = None, []

    # Decide the next cell from the subgoal route
    nxt = next_step_from_route(state)

    # If still no plan, try simple forward neighbor, else controlled one-step back
    if nxt is None:
        fwd = forward_unvisited_neighbors(cur)
        if fwd:
            nxt = fwd[0]
            # create a trivial local route
            MEM.route = [cur, nxt]
            MEM.subgoal = nxt
        elif len(MEM.spine) >= 2:
            parent = MEM.spine[-2]
            nxt = parent
            MEM.route = [cur, parent]
            MEM.subgoal = parent
        else:
            # brake to zero
            cv = state.agent.vel.astype(int)
            b = -np.sign(cv)
            return int(np.clip(b[0], -1, 1)), int(np.clip(b[1], -1, 1))

    # Convert next cell into a unit-step velocity
    cur_pos = state.agent.pos.astype(int)
    target = np.array(nxt, dtype=int)
    diff = target - cur_pos
    if abs(diff[0]) + abs(diff[1]) > 1:
        # route is always 4-neighbor, but guard anyway
        if abs(diff[0]) >= abs(diff[1]): diff = np.array([np.sign(diff[0]), 0], int)
        else: diff = np.array([0, np.sign(diff[1])], int)

    ax, ay = accel_to_unit(state, diff)

    # Predict next cell if accel is applied (unit-speed model)
    new_vel = np.clip(state.agent.vel.astype(int) + np.array([ax, ay], int), -1, 1)
    new_pos = tuple((cur_pos + new_vel).tolist())
    if new_pos != cur:
        after_move_spine_bookkeeping(cur, new_pos)

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
