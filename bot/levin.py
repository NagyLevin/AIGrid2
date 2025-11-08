import sys
import enum
import math
import heapq
from collections import deque, defaultdict
import numpy as np
from typing import Optional, NamedTuple, Tuple, List

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

        # --- DFS memory ---
        self.dfs_stack: List[Tuple[int,int]] = []              # junction checkpoints
        self.dfs_expanded_edges: set[tuple[int,int,int,int]] = set()  # expanded directed edges

    def updateWithObservation(self, st: State) -> None:
        raw = st.visible_raw
        if raw is None:
            return
        seen = (raw != CellType.NOT_VISIBLE.value)
        self.known_map[seen] = raw[seen]

    # ---- Basic traversability helpers ----
    def _trav(self, v: int) -> bool:
        return (v >= 0) and (v != CellType.UNKNOWN.value)

    def _n4(self, x: int, y: int):
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.shape[0] and 0 <= ny < self.shape[1]:
                yield nx, ny

    def traversable_neighbors(self, x: int, y: int) -> List[Tuple[int,int]]:
        km = self.known_map
        out = []
        for nx, ny in self._n4(x, y):
            if self._trav(km[nx, ny]):
                out.append((nx, ny))
        return out

    def degree(self, x: int, y: int) -> int:
        return len(self.traversable_neighbors(x, y))

    def is_frontier_cell(self, x: int, y: int) -> bool:
        # Traversable cell that touches UNKNOWN
        km = self.known_map
        if not self._trav(km[x,y]):
            return False
        for nx, ny in self._n4(x, y):
            if km[nx, ny] == CellType.UNKNOWN.value:
                return True
        return False

    # ---- Frontier detection (BFS fallback) ----
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

    # ---- Edge helpers (anti-oscillation & DFS) ----
    def edge_count(self, x: int, y: int, nx: int, ny: int) -> int:
        return self.edge_visits[(x, y, nx, ny)]

    def touch_edge(self, x: int, y: int, nx: int, ny: int) -> None:
        self.edge_visits[(x, y, nx, ny)] += 1

    # --- DFS helpers ---
    def dfs_edge_expanded(self, x: int, y: int, nx: int, ny: int) -> bool:
        return (x, y, nx, ny) in self.dfs_expanded_edges

    def record_dfs_traverse(self, x: int, y: int, nx: int, ny: int) -> None:
        # record the directed edge we actually traversed this turn
        self.dfs_expanded_edges.add((x, y, nx, ny))

    def has_unexpanded_from(self, x: int, y: int) -> bool:
        for nx, ny in self.traversable_neighbors(x, y):
            if not self.dfs_edge_expanded(x, y, nx, ny):
                return True
        return False

    def prune_dfs_stack(self) -> None:
        # Pop exhausted junctions
        while self.dfs_stack and not self.has_unexpanded_from(*self.dfs_stack[-1]):
            self.dfs_stack.pop()

    def update_dfs_stack_at(self, cur: Tuple[int,int]) -> None:
        cx, cy = cur
        # push junctions (deg>=3) as checkpoints if not already top
        if self.degree(cx, cy) >= 3:
            if not self.dfs_stack or self.dfs_stack[-1] != (cx, cy):
                # avoid duplicates deeper in stack (loop closure)
                if (cx, cy) in self.dfs_stack:
                    # keep the earlier ones up to this occurrence
                    idx = self.dfs_stack.index((cx, cy))
                    self.dfs_stack = self.dfs_stack[:idx+1]
                else:
                    self.dfs_stack.append((cx, cy))
        # prune any exhausted junctions on top
        self.prune_dfs_stack()

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

# ---- A* planner on (x,y,vx,vy) with anti-oscillation costs ----

class AStarPlanner:
    def __init__(self, world: WorldModel, vMax: int, rSafe: int, maxNodes: int = 20000):
        self.world = world
        self.v_max = vMax
        self.R_safe = max(0, rSafe)
        self.max_nodes = maxNodes
        # turning penalties
        self.turn_pen_back = 6.0
        self.turn_pen_half = 2.0
        self.turn_pen_ortho = 0.5
        # anti-oscillation weights
        self.w_backtrack = 4.0     # stepping back to previous cell
        self.w_edge_repeat = 0.7   # repeating the same directed edge
        self.w_visit = 0.05        # revisiting frequently seen cells

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

            # previous cell (exact kinematics identity): p_{t-1} = p_t - v_t
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

                    # --- Anti-oscillation extras ---
                    backtrack_pen = self.w_backtrack if (nx == prev_x and ny == prev_y) else 0.0
                    edge_rep_pen = self.w_edge_repeat * float(self.world.edge_count(x, y, nx, ny))
                    visit_pen = self.w_visit * float(self.world.visited_count[nx, ny])

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

# ---- DFS-first Target selection ----

def chooseTargetDFS(world: WorldModel, agentXY: Tuple[int,int]) -> Tuple[str, Optional[Tuple[int,int]]]:
    """
    DFS policy on the grid graph (known traversable cells).
    - Extend along an unexpanded neighbor (frontier-biased) if any.
    - Else backtrack to nearest junction with remaining branches.
    - Else None.
    """
    km = world.known_map

    # 1) If GOAL is known anywhere, let higher layer handle it outside.
    # (We check goals outside this function.)

    # 2) Maintain stack with current position context.
    world.update_dfs_stack_at(agentXY)

    x, y = agentXY
    if not (0 <= x < km.shape[0] and 0 <= y < km.shape[1]) or not world._trav(km[x,y]):
        return "idle", None

    # 3) Try to extend DFS from current cell: prefer unexpanded edges.
    neighbors = world.traversable_neighbors(x, y)

    # Partition neighbors by frontier bias first
    unexpanded = [(nx, ny) for (nx, ny) in neighbors if not world.dfs_edge_expanded(x, y, nx, ny)]
    if unexpanded:
        frontier_first = [(nx, ny) for (nx, ny) in unexpanded if world.is_frontier_cell(nx, ny)]
        pool = frontier_first if frontier_first else unexpanded
        # Choose least visited to reduce loops
        pool.sort(key=lambda p: (world.visited_count[p[0], p[1]]))
        return "dfs_extend", pool[0]

    # 4) No local unexpanded edges -> backtrack to nearest junction with remaining branches
    world.prune_dfs_stack()
    if world.dfs_stack:
        return "dfs_backtrack", world.dfs_stack[-1]

    # 5) No DFS target -> None (let caller try BFS frontier or fallback)
    return "idle", None

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

            # base rank and forward bias
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

            # --- Anti-oscillation extras (local) ---
            if nx == int(last_pos[0]) and ny == int(last_pos[1]):
                rank += 3.0  # avoid immediate backtrack unless necessary

            rank += 0.6 * float(world.edge_count(int(selfPos[0]), int(selfPos[1]), nx, ny))
            rank += 0.05 * float(world.visited_count[nx, ny])

            candidates.append(((ax, ay), rank, (nx, ny)))

    if not candidates:
        # gentle brake towards zero velocity
        ax = -1 if vx > 0 else (1 if vx < 0 else 0)
        ay = -1 if vy > 0 else (1 if vy < 0 else 0)
        if brakingOk(vx+ax, vy+ay, rSafe):
            return (ax, ay)
        return (0, 0)

    # Prefer those that are not immediate backtracks when possible
    non_back = [c for c in candidates if c[2] != (int(last_pos[0]), int(last_pos[1]))]
    pool = non_back if non_back else candidates

    pool.sort(key=lambda it: it[1])
    best_axay = pool[0][0]
    return best_axay

# ---- Decision (receding horizon) ----

def calculateMove(world: WorldModel, planner: AStarPlanner,
                  rng: np.random.Generator, state: State) -> Tuple[int,int]:
    assert state.agent is not None
    assert state.visible_raw is not None

    world.updateWithObservation(state)

    R = state.circuit.visibility_radius
    SAFETY_MARGIN = 1
    rSafe = max(0, R - SAFETY_MARGIN)

    agentXY = (int(state.agent.x), int(state.agent.y))
    agentV  = (int(state.agent.vel_x), int(state.agent.vel_y))

    # --- Priority 1: if any GOAL is known, go for it
    km = world.known_map
    goals = np.argwhere(km == CellType.GOAL.value)
    if goals.size > 0:
        sx, sy = agentXY
        dists = [ (abs(int(x)-sx)+abs(int(y)-sy), (int(x),int(y))) for (x,y) in goals ]
        dists.sort()
        target = dists[0][1]
        actions = planner.plan(agentXY, agentV, target)
        if actions:
            ax, ay = actions[0]
        else:
            ax, ay = fallbackMoveWithBrakeAndBias(state, world, rSafe)
    else:
        # --- DFS-first target selection
        mode, target = chooseTargetDFS(world, agentXY)

        if target is None or mode == "idle":
            # Try BFS to nearest frontier as a graceful fallback
            path = world.nearestFrontierFrom(agentXY)
            if path and len(path) > 0:
                target = path[-1]
                actions = planner.plan(agentXY, agentV, target)
                if actions:
                    ax, ay = actions[0]
                else:
                    ax, ay = fallbackMoveWithBrakeAndBias(state, world, rSafe)
            else:
                ax, ay = fallbackMoveWithBrakeAndBias(state, world, rSafe)
        else:
            # Plan to DFS target (neighbor or backtrack junction)
            actions = planner.plan(agentXY, agentV, target)
            if actions:
                ax, ay = actions[0]
            else:
                # If planner fails, try safe local move
                ax, ay = fallbackMoveWithBrakeAndBias(state, world, rSafe)

    # avoid stepping onto another player's cell
    nextPos = state.agent.pos + state.agent.vel + np.array([ax, ay])
    if any(np.all(nextPos == p.pos) for p in state.players):
        ax, ay = 0, 0
        nextPos = state.agent.pos + state.agent.vel + np.array([ax, ay])

    # update visit counts (node + edge) to discourage retracing
    world.visited_count[agentXY[0], agentXY[1]] += 1
    nx, ny = int(nextPos[0]), int(nextPos[1])
    world.touch_edge(agentXY[0], agentXY[1], nx, ny)

    # --- Record DFS traversal on the actual directed edge we took this turn
    world.record_dfs_traverse(agentXY[0], agentXY[1], nx, ny)

    return ax, ay

# ---- main: strict I/O (only READY + moves on stdout) ----

def main():
    print("READY", flush=True)  # mandatory handshake
    circuit = read_initial_observation()
    state: Optional[State] = State(circuit, None, None, [], None)

    world = WorldModel(circuit.track_shape)

    # braking guard drives speed; v_max can be lenient
    R = circuit.visibility_radius
    SAFETY_MARGIN = 1
    rSafe = max(0, R - SAFETY_MARGIN)

    planner = AStarPlanner(world, vMax=7, rSafe=rSafe, maxNodes=30000)
    rng = np.random.default_rng(seed=1)

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return
        ax, ay = calculateMove(world, planner, rng, state)
        # clamp to {-1,0,1} for the judge delta
        ax = -1 if ax < -1 else (1 if ax > 1 else int(ax))
        ay = -1 if ay < -1 else (1 if ay > 1 else int(ay))
        print(f"{ax} {ay}", flush=True)

if __name__ == "__main__":
    main()
