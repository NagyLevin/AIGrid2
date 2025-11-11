import sys
import enum
import math
import heapq
from collections import deque, defaultdict
import numpy as np
from typing import Optional, NamedTuple, Tuple, List, Dict

# ---- Judge-compatible basic types ----

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

# ---- Input (strict judge protocol; stdout stays clean) ----

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

    # Build both projections:
    #  - visible_raw keeps NOT_VISIBLE
    #  - visible_track turns NOT_VISIBLE into WALL for local safety
    visible_raw = np.full((H, W), CellType.NOT_VISIBLE.value, dtype=int)
    visible_track = np.full((H, W), CellType.WALL.value, dtype=int)

    for i in range(2 * R + 1):
        row_vals = [int(a) for a in input().split()]   # exactly 2R+1 ints
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
                # safety: NOT_VISIBLE -> WALL
                safety = [CellType.WALL.value if v == CellType.NOT_VISIBLE.value else v
                          for v in loc]
                visible_track[x, ys:ye] = safety

    return old_state._replace(visible_track=visible_track,
                              visible_raw=visible_raw,
                              players=players, agent=agent)

# ---- Persistent World Model (UNKNOWN is not plannable) ----

class WorldModel:
    def __init__(self, shape: tuple[int, int]) -> None:
        H, W = shape
        self.shape = shape
        self.known_map = np.full((H, W), CellType.UNKNOWN.value, dtype=int)
        self.visited_count = np.zeros((H, W), dtype=int)
        # Directed edge traversal counts: key = (x,y,nx,ny)
        self.edge_visits: defaultdict[tuple[int,int,int,int], int] = defaultdict(int)

        # --- Anti-backtrack memory
        self.backtrail: deque[Tuple[int,int]] = deque(maxlen=80)  # recent path cells
        self.prev_pos: Optional[Tuple[int,int]] = None
        self.last_dir: np.ndarray = np.array([0, 0], dtype=int)

        # Committed exploration target (hysteresis)
        self.commit_target: Optional[Tuple[int,int]] = None
        self.commit_ttl: int = 0
        self.stalled_steps: int = 0
        self._COMMIT_TTL_DEFAULT = 40
        self._STALL_RESET = 8
        self.no_backtrack_lock: int = 0  # countdown

    def updateWithObservation(self, st: State) -> None:
        raw = st.visible_raw
        if raw is None:
            return
        seen = (raw != CellType.NOT_VISIBLE.value)
        self.known_map[seen] = raw[seen]

    # ---- Frontier detection ----
    def _trav(self, v: int) -> bool:
        return (v >= 0) and (v != CellType.UNKNOWN.value)

    def _n4(self, x: int, y: int):
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.shape[0] and 0 <= ny < self.shape[1]:
                yield nx, ny

    def has_unknown_neighbor(self, x: int, y: int) -> bool:
        km = self.known_map
        for nx, ny in self._n4(x, y):
            if km[nx, ny] == CellType.UNKNOWN.value:
                return True
        return False

    def frontierCells(self) -> List[Tuple[int,int]]:
        km = self.known_map
        H, W = self.shape
        out = []
        for x in range(H):
            for y in range(W):
                if not self._trav(km[x,y]):
                    continue
                for nx, ny in self._n4(x,y):
                    if km[nx,ny] == CellType.UNKNOWN.value:
                        out.append((x,y)); break
        return out

    def nearestFrontierFrom(self, start: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
        sx, sy = start
        if not (0 <= sx < self.shape[0] and 0 <= sy < self.shape[1]): return None
        if not self._trav(self.known_map[sx,sy]): return None

        goals = set(self.frontierCells())
        if not goals: return None

        q = deque([(sx,sy)])
        prev = {(sx,sy): None}
        while q:
            x,y = q.popleft()
            if (x,y) in goals:
                path = []
                cur = (x,y)
                while cur is not None:
                    path.append(cur); cur = prev[cur]
                path.reverse(); return path
            for nx,ny in self._n4(x,y):
                if (nx,ny) in prev: continue
                if self._trav(self.known_map[nx,ny]):
                    prev[(nx,ny)] = (x,y); q.append((nx,ny))
        return None

    # Edge helpers
    def edge_count(self, x: int, y: int, nx: int, ny: int) -> int:
        return self.edge_visits[(x, y, nx, ny)]

    def touch_edge(self, x: int, y: int, nx: int, ny: int) -> None:
        self.edge_visits[(x, y, nx, ny)] += 1

# ---- Geometry & safety ----

def isTraversableForPlanning(v: int) -> bool:
    return (v >= 0) and (v != CellType.UNKNOWN.value)

def validLineOnMap(world: WorldModel, p1: np.ndarray, p2: np.ndarray) -> bool:
    km = world.known_map
    H, W = km.shape
    if (p1[0] < 0 or p1[1] < 0 or p2[0] < 0 or p2[1] < 0 or
        p1[0] >= H or p1[1] >= W or p2[0] >= H or p2[1] >= W):
        return False

    diff = p2 - p1
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))
        for i in range(abs(diff[0]) + 1):
            x = int(p1[0] + i*d)
            y = p1[1] + i*slope*d
            yCeil = int(np.ceil(y)); yFloor = int(np.floor(y))
            if not isTraversableForPlanning(km[x, yCeil]) and \
               not isTraversableForPlanning(km[x, yFloor]):
                return False
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(diff[1]) + 1):
            x = p1[0] + i*slope*d
            y = int(p1[1] + i*d)
            xCeil = int(np.ceil(x)); xFloor = int(np.floor(x))
            if not isTraversableForPlanning(km[xCeil, y]) and \
               not isTraversableForPlanning(km[xFloor, y]):
                return False
    return True

def traversable_local(v: int) -> bool:
    return v >= 0

def validLineLocal(state: State, p1: np.ndarray, p2: np.ndarray) -> bool:
    track = state.visible_track
    if (np.any(p1 < 0) or np.any(p2 < 0) or
        p1[0] >= track.shape[0] or p1[1] >= track.shape[1] or
        p2[0] >= track.shape[0] or p2[1] >= track.shape[1]):
        return False

    diff = p2 - p1
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))
        for i in range(abs(diff[0]) + 1):
            x = int(p1[0] + i*d)
            y = p1[1] + i*slope*d
            yCeil = int(np.ceil(y)); yFloor = int(np.floor(y))
            if (not traversable_local(track[x, yCeil]) and
                not traversable_local(track[x, yFloor])): return False
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(diff[1]) + 1):
            x = p1[0] + i*slope*d
            y = int(p1[1] + i*d)
            xCeil = int(np.ceil(x)); xFloor = int(np.floor(x))
            if (not traversable_local(track[xCeil, y]) and
                not traversable_local(track[xFloor, y])): return False
    return True

# ---- Braking invariant helpers ----

def tri(n: int) -> int:
    return n*(n+1)//2

def brakingOk(vx: int, vy: int, rSafe: int) -> bool:
    return (tri(abs(vx)) <= rSafe) and (tri(abs(vy)) <= rSafe)

# ---- Coarse planner (positions only) with tabu backtrail ----

class CoarsePlanner2D:
    def __init__(self, world: WorldModel):
        self.world = world
        self.cost_unknown = 5.0
        self.cost_empty = 1.0
        self.cost_goal = 1.0
        self.cost_recent_back = 50.0
        self.cost_old_back = 6.0

    def _neighbors8(self, x: int, y: int):
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.world.shape[0] and 0 <= ny < self.world.shape[1]:
                yield nx, ny, (1.41421356 if dx and dy else 1.0)

    def _cell_cost(self, v: int, cell: Tuple[int,int]) -> float:
        if v == CellType.WALL.value:
            return float("inf")
        base = self.cost_goal if v == CellType.GOAL.value else \
               (self.cost_unknown if v == CellType.UNKNOWN.value else self.cost_empty)

        if self.world.backtrail:
            try:
                idx = len(self.world.backtrail) - 1 - list(self.world.backtrail)[::-1].index(cell)
                age = len(self.world.backtrail) - 1 - idx
                if age <= 12:   base += self.cost_recent_back
                elif age <= 35: base += self.cost_old_back
            except ValueError:
                pass
        return base

    def plan_path(self, start: Tuple[int,int], target: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
        sx, sy = start
        tx, ty = target
        km = self.world.known_map
        if not (0 <= sx < km.shape[0] and 0 <= sy < km.shape[1]): return None
        if not (0 <= tx < km.shape[0] and 0 <= ty < km.shape[1]): return None
        if km[tx,ty] == CellType.WALL.value:
            return None

        def h(x: int, y: int) -> float:
            return math.hypot(x-tx, y-ty)

        g: Dict[Tuple[int,int], float] = {(sx,sy): 0.0}
        parent: Dict[Tuple[int,int], Tuple[int,int]] = {}
        pq: List[Tuple[float, int, Tuple[int,int]]] = []
        rid = 0
        heapq.heappush(pq, (h(sx,sy), rid, (sx,sy))); rid += 1
        seen = set()

        while pq:
            _, _, (x,y) = heapq.heappop(pq)
            if (x,y) in seen:
                continue
            seen.add((x,y))

            if (x,y) == (tx,ty) or km[x,y] == CellType.GOAL.value:
                path = []
                cur = (x,y)
                while cur != (sx,sy):
                    path.append(cur); cur = parent[cur]
                path.append((sx,sy))
                path.reverse()
                return path

            for nx, ny, step_len in self._neighbors8(x,y):
                c = self._cell_cost(km[nx,ny], (nx,ny))
                if c == float("inf"): 
                    continue
                cand = g[(x,y)] + step_len + c
                if cand < g.get((nx,ny), float("inf")):
                    g[(nx,ny)] = cand
                    parent[(nx,ny)] = (x,y)
                    f = cand + h(nx,ny)
                    heapq.heappush(pq, (f, rid, (nx,ny))); rid += 1
        return None

# ---- A* planner on (x,y,vx,vy) with anti-oscillation costs (fallback) ----

class AStarPlanner:
    def __init__(self, world: WorldModel, vMax: int, rSafe: int, maxNodes: int = 20000):
        self.world = world
        self.v_max = vMax
        self.R_safe = max(0, rSafe)
        self.max_nodes = maxNodes
        self.turn_pen_back = 6.0
        self.turn_pen_half = 2.0
        self.turn_pen_ortho = 0.5
        self.w_backtrack = 6.0
        self.w_edge_repeat = 0.9
        self.w_visit = 0.15

    def heuristicSteps(self, pos: Tuple[int,int], target: Tuple[int,int]) -> float:
        dx = pos[0] - target[0]
        dy = pos[1] - target[1]
        dist = math.hypot(dx, dy)
        return dist / (self.v_max + 1)

    def _clampV(self, v: int) -> int:
        return max(-self.v_max, min(self.v_max, v))

    def _turnPenalty(self, vx: int, vy: int, nvx: int, nvy: int) -> float:
        a = math.hypot(vx, vy); b = math.hypot(nvx, nvy)
        if a == 0 or b == 0: return 0.0
        cos = (vx*nvx + vy*nvy) / (a*b)
        if cos < -0.5: return self.turn_pen_back
        if cos < 0.0:  return self.turn_pen_half
        if abs(cos) < 1e-9: return self.turn_pen_ortho
        return 0.0

    def plan(self, startPos: Tuple[int,int], startVel: Tuple[int,int], targetPos: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
        sx, sy = startPos
        svx, svy = startVel
        tx, ty = targetPos

        startState = (sx, sy, svx, svy)
        gCost = { startState: 0.0 }
        parent: dict[Tuple[int,int,int,int], Tuple[Tuple[int,int,int,int], Tuple[int,int]]] = {}
        pq: List[Tuple[float,int,Tuple[int,int,int,int]]] = []
        counter = 0
        heapq.heappush(pq, (self.heuristicSteps((sx,sy),(tx,ty)), counter, startState))
        counter += 1
        nodesPopped = 0
        km = self.world.known_map

        def isGoalCell(x: int, y: int) -> bool:
            return km[x,y] == CellType.GOAL.value or (x == tx and y == ty)

        while pq:
            _, _, (x, y, vx, vy) = heapq.heappop(pq)
            nodesPopped += 1
            if nodesPopped > self.max_nodes:
                return None

            if isGoalCell(x, y):
                actions: List[Tuple[int,int]] = []
                cur = (x, y, vx, vy)
                while cur != startState:
                    prev, act = parent[cur]
                    actions.append(act); cur = prev
                actions.reverse()
                return actions

            prev_x, prev_y = (x - vx, y - vy)

            for ax in (-1,0,1):
                for ay in (-1,0,1):
                    nvx = self._clampV(vx + ax)
                    nvy = self._clampV(vy + ay)
                    if not brakingOk(nvx, nvy, self.R_safe):
                        continue

                    nx = x + nvx
                    ny = y + nvy
                    if not (0 <= nx < km.shape[0] and 0 <= ny < km.shape[1]):
                        continue
                    if not validLineOnMap(self.world, np.array([x,y]), np.array([nx,ny])):
                        continue

                    base = 1.0
                    turnp = self._turnPenalty(vx,vy,nvx,nvy)

                    backtrack_pen =  self.w_backtrack if (nx == prev_x and ny == prev_y) else 0.0
                    edge_rep_pen =  self.w_edge_repeat * float(self.world.edge_count(x, y, nx, ny))
                    visit_pen    =  self.w_visit * float(self.world.visited_count[nx, ny])

                    stepCost = base + turnp + backtrack_pen + edge_rep_pen + visit_pen

                    ns = (nx, ny, nvx, nvy)
                    tentative = gCost[(x,y,vx,vy)] + stepCost
                    if tentative < gCost.get(ns, float("inf")):
                        gCost[ns] = tentative
                        parent[ns] = ((x,y,vx,vy), (ax,ay))
                        f = tentative + self.heuristicSteps((nx,ny),(tx,ty))
                        heapq.heappush(pq, (f, counter, ns))
                        counter += 1
        return None

# ---- Target selection with hysteresis ----

def _choose_committed_target(world: WorldModel, agentXY: Tuple[int,int]) -> Optional[Tuple[int,int]]:
    km = world.known_map
    goals = np.argwhere(km == CellType.GOAL.value)
    if goals.size > 0:
        dists = [ (abs(int(x)-agentXY[0]) + abs(int(y)-agentXY[1]), (int(x),int(y))) for (x,y) in goals ]
        dists.sort()
        world.commit_target = dists[0][1]
        world.commit_ttl = world._COMMIT_TTL_DEFAULT
        return world.commit_target

    def valid_frontier_cell(c: Tuple[int,int]) -> bool:
        x, y = c
        if not (0 <= x < km.shape[0] and 0 <= y < km.shape[1]): return False
        if not isTraversableForPlanning(km[x,y]): return False
        return world.has_unknown_neighbor(x, y)

    if world.commit_target and world.commit_ttl > 0 and valid_frontier_cell(world.commit_target):
        world.commit_ttl -= 1
        return world.commit_target

    fpath = world.nearestFrontierFrom(agentXY)
    if not fpath:
        world.commit_target = None
        world.commit_ttl = 0
        return None

    frontier = fpath[-1]
    candidates = world.frontierCells()
    if candidates:
        vdir = world.last_dir.astype(float)
        def score(c):
            dist = abs(c[0]-agentXY[0]) + abs(c[1]-agentXY[1])
            if np.all(vdir == 0):
                turn = 0.0
            else:
                to = np.array([c[0]-agentXY[0], c[1]-agentXY[1]], dtype=float)
                n_to = np.linalg.norm(to) or 1.0
                cos = float(np.dot(vdir, to) / (np.linalg.norm(vdir) * n_to))
                turn = (1.0 - cos) * 4.0  # prefer forward
            tabu = 20.0 if c in world.backtrail else 0.0
            return dist + turn + tabu
        candidates.sort(key=score)
        frontier = candidates[0]

    world.commit_target = frontier
    world.commit_ttl = world._COMMIT_TTL_DEFAULT
    return frontier

# ---- Local fallback (visibility-safe + forward/novelty bias) ----

def fallbackMoveWithBrakeAndBias(state: State, world: WorldModel, rSafe: int) -> Tuple[int,int]:
    assert state.agent is not None
    selfPos = state.agent.pos
    v = state.agent.vel
    vx, vy = int(v[0]), int(v[1])

    last_pos = selfPos - v  # exact previous cell by kinematics

    newCenter = selfPos + v
    candidates: List[Tuple[Tuple[int,int], float, Tuple[int,int]]] = []  # ((ax,ay), rank, (nx,ny))

    for ax in (-1,0,1):
        for ay in (-1,0,1):
            nvx, nvy = vx + ax, vy + ay
            if not brakingOk(nvx, nvy, rSafe):
                continue
            nextMove = newCenter + np.array([ax, ay])
            nx, ny = int(nextMove[0]), int(nextMove[1])
            if not validLineLocal(state, selfPos, nextMove):
                continue

            a = math.hypot(vx, vy); b = math.hypot(nvx, nvy)
            rank = 1.0
            if a > 0 and b > 0:
                cos = (vx*nvx + vy*nvy) / (a*b)
                if   cos > 0: rank -= 0.4
                elif cos == 0: rank += 0.2
                else: rank += 1.5
            elif a == 0 and b > 0:
                rank -= 0.1
            elif b == 0:
                rank += 0.3

            if nx == int(last_pos[0]) and ny == int(last_pos[1]):
                rank += 5.0  # avoid immediate backtrack unless necessary

            rank += 0.9 * float(world.edge_count(int(selfPos[0]), int(selfPos[1]), nx, ny))
            rank += 0.18 * float(world.visited_count[nx, ny])

            if world.has_unknown_neighbor(nx, ny):
                rank -= 0.6

            candidates.append(((ax, ay), rank, (nx, ny)))

    if not candidates:
        ax = -1 if vx > 0 else (1 if vx < 0 else 0)
        ay = -1 if vy > 0 else (1 if vy < 0 else 0)
        if brakingOk(vx+ax, vy+ay, rSafe):
            return (ax, ay)
        return (0, 0)

    non_back = [c for c in candidates if c[2] != (int(last_pos[0]), int(last_pos[1]))]
    pool = non_back if non_back else candidates

    pool.sort(key=lambda it: it[1])
    best_axay = pool[0][0]
    return best_axay

# ---- Pure-Pursuit Waypoint driver (fast default) ----

def _furthest_visible_on_path(world: WorldModel, start_xy: Tuple[int,int], path: List[Tuple[int,int]], max_jump: int = 40) -> Tuple[int, Tuple[int,int]]:
    if not path:
        return 0, start_xy
    p0 = np.array(start_xy, dtype=int)
    last_idx = 0
    last_cell = path[0]
    upto = min(len(path)-1, max_jump)
    for i in range(1, upto+1):
        c = path[i]
        if validLineOnMap(world, p0, np.array(c, dtype=int)):
            last_idx = i
            last_cell = c
        else:
            break
    return last_idx, last_cell

def _path_curvature(path: List[Tuple[int,int]], i: int) -> float:
    if i <= 0 or i >= len(path)-1:
        return 0.0
    p0 = np.array(path[i-1]); p1 = np.array(path[i]); p2 = np.array(path[i+1])
    v1 = p1 - p0; v2 = p2 - p1
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    cos = np.clip(float(np.dot(v1, v2) / (n1*n2)), -1.0, 1.0)
    ang = math.acos(cos)  # [0..pi]
    return min(1.0, ang / math.pi)

def _target_speed_from_context(distance_left: float, curvature: float, rSafe: int) -> float:
    """
    Pick a target |v'|. Go faster on straights and when far; slow down near bends or close to goal.
    HARD CHANGE HERE: cap from vision radius is HALVED.
    """
    base = 3.5 if curvature < 0.15 else (2.5 if curvature < 0.4 else 1.5)
    far_boost = 2.0 if distance_left > 15 else (1.0 if distance_left > 8 else 0.0)
    tgt = base + far_boost
    # Half of the old cap derived from rSafe (vision radius).
    cap = max(1.0, 0.5 * math.sqrt(2 * max(0, rSafe)))
    return float(min(tgt, cap))

def _score_accel(ax: int, ay: int,
                 state: State, world: WorldModel, rSafe: int,
                 waypoint: Tuple[int,int], target_speed: float) -> Optional[Tuple[float, Tuple[int,int]]]:
    assert state.agent is not None
    p = state.agent.pos
    v = state.agent.vel
    vx, vy = int(v[0]), int(v[1])
    nvx, nvy = vx + ax, vy + ay
    if not brakingOk(nvx, nvy, rSafe):
        return None

    next_pos = p + v + np.array([ax, ay], dtype=int)
    nx, ny = int(next_pos[0]), int(next_pos[1])

    if not validLineLocal(state, p, next_pos):
        return None
    if any(np.array_equal(next_pos, q.pos) for q in state.players):
        return None

    prev_cell = None
    if world.prev_pos is not None:
        prev_cell = tuple(world.prev_pos)
    if prev_cell is not None and (nx, ny) == prev_cell and world.no_backtrack_lock > 0:
        base_pen = 1e6  # strongly avoid if any other option exists
    else:
        base_pen = 0.0

    wp = np.array(waypoint, dtype=float)
    dist_to_wp = float(np.linalg.norm(wp - next_pos))
    speed_next = float(math.hypot(nvx, nvy))
    speed_pen = abs(speed_next - target_speed)

    heading_pen = 0.0
    if speed_next > 0.0:
        to_wp = wp - next_pos
        n_to = float(np.linalg.norm(to_wp))
        if n_to > 0:
            cos = float((nvx*to_wp[0] + nvy*to_wp[1]) / (speed_next * n_to))
            heading_pen = (1.0 - cos) * 0.6

    node_pen = 0.22 * float(world.visited_count[nx, ny])
    edge_pen = 0.8 * float(world.edge_count(int(p[0]), int(p[1]), nx, ny))
    explore_bonus = -0.8 if world.has_unknown_neighbor(nx, ny) else 0.0
    stop_bias = 0.3 if speed_next == 0.0 else 0.0

    score = base_pen + (2.0 * dist_to_wp) + (0.8 * speed_pen) + heading_pen + node_pen + edge_pen + stop_bias + explore_bonus
    return score, (nx, ny)

def pure_pursuit_move(state: State, world: WorldModel,
                      coarse: CoarsePlanner2D,
                      rSafe: int) -> Optional[Tuple[int,int]]:
    assert state.agent is not None
    agent_xy = (int(state.agent.x), int(state.agent.y))

    # Choose / keep a committed target
    target = _choose_committed_target(world, agent_xy)
    if target is None:
        return None

    # fast 2D path with tabu backtrail
    path = coarse.plan_path(agent_xy, target)
    if not path or len(path) <= 1:
        return None

    # choose furthest visible waypoint
    far_idx, wp = _furthest_visible_on_path(world, agent_xy, path, max_jump=40)

    # curvature and path distance for speed target
    curv = _path_curvature(path, max(1, min(far_idx, len(path)-2)))
    dist_left = float(len(path) - far_idx)
    target_speed = _target_speed_from_context(dist_left, curv, rSafe)

    scored: List[Tuple[float, Tuple[int,int], Tuple[int,int]]] = []  # (score, (nx,ny), (ax,ay))
    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            res = _score_accel(ax, ay, state, world, rSafe, wp, target_speed)
            if res is None:
                continue
            sc, nxny = res
            scored.append((sc, nxny, (ax, ay)))

    if not scored:
        return None

    # DFS-style rule: if there exists any candidate not stepping onto the previous cell, drop backtracking ones
    non_back = []
    prev_cell = tuple(world.prev_pos) if world.prev_pos is not None else None
    for sc, nxny, axay in scored:
        if prev_cell is not None and nxny == prev_cell:
            continue
        non_back.append((sc, nxny, axay))

    if non_back:
        scored = non_back
        world.no_backtrack_lock = 6  # for next few steps, keep resisting backtracking
    else:
        world.no_backtrack_lock = max(0, world.no_backtrack_lock - 1)

    scored.sort(key=lambda t: t[0])
    best_axay = scored[0][2]
    return best_axay

# ---- Decision (receding horizon) ----

def calculateMove(world: WorldModel, planner: AStarPlanner,
                  rng: np.random.Generator, state: State,
                  coarse: CoarsePlanner2D) -> Tuple[int,int]:
    assert state.agent is not None
    assert state.visible_raw is not None

    # Map update
    world.updateWithObservation(state)

    R = state.circuit.visibility_radius
    SAFETY_MARGIN = 1
    rSafe = max(0, R - SAFETY_MARGIN)

    agentXY = (int(state.agent.x), int(state.agent.y))
    agentV  = (int(state.agent.vel_x), int(state.agent.vel_y))

    if world.prev_pos is not None:
        world.last_dir = np.array(agentXY) - np.array(world.prev_pos)
    world.backtrail.append(agentXY)

    fast_axay = pure_pursuit_move(state, world, coarse, rSafe)

    if fast_axay is None:
        mode_target = _choose_committed_target(world, agentXY)
        if mode_target is not None:
            actions = planner.plan(agentXY, agentV, mode_target)
            if actions:
                ax, ay = actions[0]
            else:
                ax, ay = fallbackMoveWithBrakeAndBias(state, world, rSafe)
        else:
            ax, ay = fallbackMoveWithBrakeAndBias(state, world, rSafe)
    else:
        ax, ay = fast_axay

    nextPos = state.agent.pos + state.agent.vel + np.array([ax, ay])
    if any(np.all(nextPos == p.pos) for p in state.players):
        ax, ay = 0, 0
        nextPos = state.agent.pos + state.agent.vel + np.array([ax, ay])

    world.visited_count[agentXY[0], agentXY[1]] += 1
    nx, ny = int(nextPos[0]), int(nextPos[1])
    world.touch_edge(agentXY[0], agentXY[1], nx, ny)

    if world.commit_target is not None:
        before = abs(agentXY[0]-world.commit_target[0]) + abs(agentXY[1]-world.commit_target[1])
        after  = abs(nx-world.commit_target[0]) + abs(ny-world.commit_target[1])
        if after < before:
            world.stalled_steps = 0
        else:
            world.stalled_steps += 1
            if world.stalled_steps >= world._STALL_RESET:
                world.commit_ttl = 0
                world.commit_target = None
                world.stalled_steps = 0

    world.prev_pos = agentXY
    return ax, ay

# ---- main: strict I/O (only READY + moves on stdout) ----

def main():
    print("READY", flush=True)  # mandatory handshake
    circuit = read_initial_observation()
    state: Optional[State] = State(circuit, None, None, [], None)

    world = WorldModel(circuit.track_shape)

    # braking guard drives speed; v_max can be lenient (4D planner is fallback)
    R = circuit.visibility_radius
    SAFETY_MARGIN = 1
    rSafe = max(0, R - SAFETY_MARGIN)

    planner = AStarPlanner(world, vMax=7, rSafe=rSafe, maxNodes=25000)
    coarse = CoarsePlanner2D(world)
    rng = np.random.default_rng(seed=1)

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return
        ax, ay = calculateMove(world, planner, rng, state, coarse)
        ax = -1 if ax < -1 else (1 if ax > 1 else int(ax))
        ay = -1 if ay < -1 else (1 if ay > 1 else int(ay))
        print(f"{ax} {ay}", flush=True)

if __name__ == "__main__":
    main()
#large1.png 120
#arrows.png 181
