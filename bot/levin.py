import sys
import enum
import math
import heapq
from collections import deque
import numpy as np
from typing import Optional, NamedTuple, Tuple, List

# --- Logging setup -------------------------------------------------------------

LOG_FILE = "levin.log"

def log_line(prefix: str, message: str) -> None:
    """Append one line: '<prefix>: <message>' to LOG_FILE (never crash on error)."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{prefix}: {message}\n")
            f.flush()
    except Exception as e:
        print(f"[logging error] {e}", file=sys.stderr)

def log_judge(raw: str) -> None:
    log_line("JUDGE", raw)

def log_agent(raw: str) -> None:
    log_line("Agent", raw)

def log_info(msg: str) -> None:
    log_line("INFO", msg)

def send_agent_move(ax: int, ay: int) -> None:
    """Send acceleration to judge + log it."""
    out = f"{ax} {ay}"
    print(out)
    log_agent(out)

# --- Environment types (mirror server) ----------------------------------------

class CellType(enum.Enum):
    GOAL = 100
    START = 1
    WALL = -1
    UNKNOWN = 2        # fontos: a saját világmodellben ez NEM bejárható
    EMPTY = 0
    NOT_VISIBLE = 3    # szerver-ablak csak; saját térképre nem mentjük el így

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
    visible_track: Optional[np.ndarray]   # safety map (NOT_VISIBLE->WALL)
    visible_raw: Optional[np.ndarray]     # raw (ahol nem láttunk: NOT_VISIBLE)
    players: List[Player]
    agent: Optional[Player]

# --- IO with the judge ---------------------------------------------------------

def read_initial_observation() -> Circuit:
    """
    Reads: H W num_players visibility_radius
    """
    raw = input()
    log_judge(raw)
    H, W, num_players, visibility_radius = map(int, raw.split())
    return Circuit((H, W), num_players, visibility_radius)

def read_observation(old_state: State) -> Optional[State]:
    """
    Reads one turn:
      <posx posy velx vely> or '~~~END~~~'
      <num_players lines: pposx pposy>
      <(2R+1) lines of the local window values>
    Builds:
      visible_raw  : full-map array (NOT_VISIBLE where unknown this turn)
      visible_track: full-map safety array (NOT_VISIBLE treated as WALL)
    """
    line = input()
    log_judge(line)
    if line == '~~~END~~~':
        return None

    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)
    circuit_data = old_state.circuit

    # Other players (positions only per spec here)
    players: List[Player] = []
    for _ in range(circuit_data.num_players):
        pline = input()
        log_judge(pline)
        pposx, pposy = map(int, pline.split())
        players.append(Player(pposx, pposy, 0, 0))

    H, W = circuit_data.track_shape
    R = circuit_data.visibility_radius

    # Prepare full-size buffers
    visible_raw = np.full((H, W), CellType.NOT_VISIBLE.value, dtype=int)
    visible_track = np.full((H, W), CellType.WALL.value, dtype=int)  # safety: unseen=WALL

    # Read exactly (2R+1) lines; place them into the correct strip of the map
    for i in range(2 * R + 1):
        row_raw = input()
        log_judge(row_raw)
        line_vals = [int(a) for a in row_raw.split()]

        x = posx - R + i
        if x < 0 or x >= H:
            continue
        y_start = posy - R
        y_end = y_start + (2 * R + 1)

        # Trim horizontally to [0, W)
        slice_vals = line_vals[:]
        y0 = y_start
        if y_start < 0:
            slice_vals = slice_vals[-y_start:]
            y0 = 0
        if y_end > W:
            slice_vals = slice_vals[:-(y_end - W)]
        y1 = y0 + len(slice_vals)
        if y0 < y1:
            # raw: keep NOT_VISIBLE as is
            visible_raw[x, y0:y1] = slice_vals
            # safety: NOT_VISIBLE -> WALL
            row_safety = [CellType.WALL.value if v == CellType.NOT_VISIBLE.value else v
                          for v in slice_vals]
            visible_track[x, y0:y1] = row_safety

    return old_state._replace(
        visible_track=visible_track, visible_raw=visible_raw,
        players=players, agent=agent
    )

# --- World model (persistent across turns) ------------------------------------

class WorldModel:
    """
    Persistent map & stats:
      known_map[x,y]: what we (ever) saw there (UNKNOWN initially).
                      We treat UNKNOWN as non-traversable for planning.
      visited_count[x,y]: how many times we stood on (x,y)
    """
    def __init__(self, shape: tuple[int,int]) -> None:
        H, W = shape
        self.shape = shape
        self.known_map = np.full((H, W), CellType.UNKNOWN.value, dtype=int)
        self.visited_count = np.zeros((H, W), dtype=int)

    def update_with_observation(self, st: State) -> None:
        """Integrate the raw window: everything != NOT_VISIBLE becomes known."""
        assert st.visible_raw is not None
        raw = st.visible_raw
        seen_mask = (raw != CellType.NOT_VISIBLE.value)
        self.known_map[seen_mask] = raw[seen_mask]

    # ---- Frontier detection ---------------------------------------------------

    def _is_traversable_known(self, v: int) -> bool:
        # UNKNOWN (2) is considered NOT traversable for planning safety.
        return (v >= 0) and (v != CellType.UNKNOWN.value)

    def _neighbors4(self, x: int, y: int):
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.shape[0] and 0 <= ny < self.shape[1]:
                yield nx, ny

    def frontier_cells(self) -> List[Tuple[int,int]]:
        """Cells that are known-free and adjacent to UNKNOWN."""
        km = self.known_map
        H, W = self.shape
        out = []
        for x in range(H):
            for y in range(W):
                if not self._is_traversable_known(km[x,y]):
                    continue
                for nx, ny in self._neighbors4(x,y):
                    if km[nx,ny] == CellType.UNKNOWN.value:
                        out.append((x,y))
                        break
        return out

    def nearest_frontier_from(self, start: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
        """
        BFS in (x,y) space over known-traversable cells, to get the shortest
        cell-path to the closest frontier cell. Returns the cell-path (including start and target).
        """
        sx, sy = start
        if not (0 <= sx < self.shape[0] and 0 <= sy < self.shape[1]):
            return None
        if not self._is_traversable_known(self.known_map[sx,sy]):
            return None

        goals = set(self.frontier_cells())
        if not goals:
            return None

        q = deque([(sx,sy)])
        prev = { (sx,sy): None }
        while q:
            x, y = q.popleft()
            if (x,y) in goals:
                # reconstruct
                path = []
                cur = (x,y)
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path
            for nx, ny in self._neighbors4(x,y):
                if (nx,ny) in prev:
                    continue
                if self._is_traversable_known(self.known_map[nx,ny]):
                    prev[(nx,ny)] = (x,y)
                    q.append((nx,ny))
        return None

# --- Geometry / collision ------------------------------------------------------

def is_traversable_for_planning(v: int) -> bool:
    """Traversal rule on the PERSISTENT map."""
    return (v >= 0) and (v != CellType.UNKNOWN.value)

def valid_line_on_map(world: WorldModel, p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Conservative 'line of motion' check between two integer points p1 -> p2
    using the persistent known_map. UNKNOWN treated as blocking.
    """
    km = world.known_map
    H, W = km.shape
    if (p1[0] < 0 or p1[1] < 0 or p2[0] < 0 or p2[1] < 0 or
        p1[0] >= H or p1[1] >= W or p2[0] >= H or p2[1] >= W):
        return False

    diff = p2 - p1
    # vertical-ish sampling (as in the baseline code)
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))  # step in x
        for i in range(abs(diff[0]) + 1):
            x = int(p1[0] + i*d)
            y = p1[1] + i*slope*d
            y_ceil = int(np.ceil(y))
            y_floor = int(np.floor(y))
            if not is_traversable_for_planning(km[x, y_ceil]) and \
               not is_traversable_for_planning(km[x, y_floor]):
                return False
    # horizontal-ish sampling
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))  # step in y
        for i in range(abs(diff[1]) + 1):
            x = p1[0] + i*slope*d
            y = int(p1[1] + i*d)
            x_ceil = int(np.ceil(x))
            x_floor = int(np.floor(x))
            if not is_traversable_for_planning(km[x_ceil, y]) and \
               not is_traversable_for_planning(km[x_floor, y]):
                return False
    return True

# --- A* planner on (x,y,vx,vy) ------------------------------------------------

class AStarPlanner:
    def __init__(self,
                 world: WorldModel,
                 v_max: int = 5,
                 max_nodes: int = 20000):
        self.world = world
        self.v_max = v_max
        self.max_nodes = max_nodes

    def heuristic_steps(self, pos: Tuple[int,int], target: Tuple[int,int]) -> float:
        # admissible lower bound on steps: distance / (V_MAX+1)
        dx = pos[0] - target[0]
        dy = pos[1] - target[1]
        dist = math.hypot(dx, dy)
        return dist / (self.v_max + 1)

    def plan(self,
             start_pos: Tuple[int,int],
             start_vel: Tuple[int,int],
             target_pos: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
        """
        Returns a list of accelerations [(ax,ay), ...] from start to reach target_pos
        (or a GOAL cell) under the world constraints. If no plan, returns None.
        Only the FIRST action will be executed by the caller (receding horizon).
        """
        sx, sy = start_pos
        svx, svy = start_vel
        tx, ty = target_pos

        def clamp(v: int) -> int:
            return max(-self.v_max, min(self.v_max, v))

        start_state = (sx, sy, svx, svy)
        # f = g + h; store g and parent/action
        g_cost = { start_state: 0.0 }
        parent: dict[Tuple[int,int,int,int], Tuple[Tuple[int,int,int,int], Tuple[int,int]]] = {}

        # priority queue: (f, counter, state)
        counter = 0
        pq: List[Tuple[float,int,Tuple[int,int,int,int]]] = []
        start_h = self.heuristic_steps((sx,sy), (tx,ty))
        heapq.heappush(pq, (start_h, counter, start_state))
        counter += 1

        nodes_popped = 0
        km = self.world.known_map

        def is_goal_cell(x: int, y: int) -> bool:
            return km[x,y] == CellType.GOAL.value or (x == tx and y == ty)

        while pq:
            _, _, (x, y, vx, vy) = heapq.heappop(pq)
            nodes_popped += 1
            if nodes_popped > self.max_nodes:
                return None

            if is_goal_cell(x, y):
                # reconstruct action sequence
                actions: List[Tuple[int,int]] = []
                cur = (x, y, vx, vy)
                while cur != start_state:
                    prev, act = parent[cur]
                    actions.append(act)
                    cur = prev
                actions.reverse()
                return actions

            # expand neighbors
            for ax in (-1,0,1):
                for ay in (-1,0,1):
                    nvx = clamp(vx + ax)
                    nvy = clamp(vy + ay)
                    nx = x + nvx
                    ny = y + nvy

                    # bounds & collision
                    if not (0 <= nx < km.shape[0] and 0 <= ny < km.shape[1]):
                        continue
                    # unknown treated as blocked
                    if not valid_line_on_map(self.world, np.array([x,y]), np.array([nx,ny])):
                        continue

                    # one-step cost; optionally penalize huge speeds or near-wall
                    step_cost = 1.0
                    ns = (nx, ny, nvx, nvy)
                    tentative = g_cost[(x,y,vx,vy)] + step_cost

                    if tentative < g_cost.get(ns, float("inf")):
                        g_cost[ns] = tentative
                        parent[ns] = ((x,y,vx,vy), (ax,ay))
                        h = self.heuristic_steps((nx,ny), (tx,ty))
                        f = tentative + h
                        heapq.heappush(pq, (f, counter, ns))
                        counter += 1
        return None

# --- High-level decision-making ------------------------------------------------

def choose_target(world: WorldModel, agent_xy: Tuple[int,int]) -> Tuple[str, Optional[Tuple[int,int]], Optional[List[Tuple[int,int]]]]:
    """
    Decide target:
      - If any GOAL known: pick the closest (in (x,y) BFS sense).
      - Else: go to nearest frontier (explore).
    Returns (mode, target_cell, bfs_path_in_xy_space or None).
    """
    km = world.known_map
    H, W = km.shape
    goals = np.argwhere(km == CellType.GOAL.value)
    mode = "explore"
    target = None
    bfs_xy_path = None

    if goals.size > 0:
        # pick nearest goal cell (grid distance)
        sx, sy = agent_xy
        # simple nearest by L1
        dists = [ (abs(int(x)-sx)+abs(int(y)-sy), (int(x),int(y))) for (x,y) in goals ]
        dists.sort()
        target = dists[0][1]
        mode = "goal"
        bfs_xy_path = None
    else:
        # nearest frontier
        path = world.nearest_frontier_from(agent_xy)
        if path is not None and len(path) > 0:
            target = path[-1]
            bfs_xy_path = path
            mode = "explore"
        else:
            target = None
            bfs_xy_path = None
    return mode, target, bfs_xy_path

# --- Fallback (safe-ish) move --------------------------------------------------

def fallback_move(rng: np.random.Generator, state: State) -> Tuple[int,int]:
    """
    A nagyon egyszerű baseline (az eredeti váz logikája) – ha nincs terv.
    Visszaad egy (ax, ay)-t.
    """
    self_pos = state.agent.pos
    def traversable(cell_value: int) -> bool:
        return cell_value >= 0

    def valid_line(state: State, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        track = state.visible_track
        if (np.any(pos1 < 0) or np.any(pos2 < 0) or np.any(pos1 >= track.shape)
                or np.any(pos2 >= track.shape)):
            return False
        diff = pos2 - pos1
        if diff[0] != 0:
            slope = diff[1] / diff[0]
            d = np.sign(diff[0])
            for i in range(abs(diff[0]) + 1):
                x = int(pos1[0] + i*d)
                y = pos1[1] + i*slope*d
                y_ceil = int(np.ceil(y))
                y_floor = int(np.floor(y))
                if (not traversable(track[x, y_ceil])
                        and not traversable(track[x, y_floor])):
                    return False
        if diff[1] != 0:
            slope = diff[0] / diff[1]
            d = np.sign(diff[1])
            for i in range(abs(diff[1]) + 1):
                x = pos1[0] + i*slope*d
                y = int(pos1[1] + i*d)
                x_ceil = int(np.ceil(x))
                x_floor = int(np.floor(x))
                if (not traversable(track[x_ceil, y])
                        and not traversable(track[x_floor, y])):
                    return False
        return True

    def valid_move(next_move):
        return (valid_line(state, self_pos, next_move) and
                (np.all(next_move == self_pos)
                 or not any(np.all(next_move == p.pos) for p in state.players)))

    new_center = self_pos + state.agent.vel
    next_move = new_center
    if (np.any(next_move != self_pos) and valid_move(next_move) and rng.random() > 0.1):
        return (0, 0)
    else:
        valid_moves = []
        valid_stay = None
        for i in range(-1, 2):
            for j in range(-1, 2):
                next_move = new_center + np.array([i, j])
                if valid_move(next_move):
                    if np.all(self_pos == next_move):
                        valid_stay = (i, j)
                    else:
                        valid_moves.append((i, j))
        if valid_moves:
            return tuple(rng.choice(valid_moves))
        elif valid_stay is not None:
            return valid_stay
        else:
            print('Not blind, just being brave! (No valid action found.)', file=sys.stderr)
            return (0, 0)

# --- Main loop ----------------------------------------------------------------

def calculate_move(world: WorldModel, planner: AStarPlanner,
                   rng: np.random.Generator, state: State) -> Tuple[int,int]:
    """
    One full decision:
      1) Update world model from observation
      2) Choose target (goal or frontier)
      3) Plan with A* on (x,y,vx,vy)
      4) Execute only FIRST (ax,ay) (receding horizon)
      5) Log details
    """
    assert state.agent is not None
    assert state.visible_raw is not None

    # 1) Integrate observation
    world.update_with_observation(state)

    ax, ay = 0, 0  # default if everything fails

    # 2) Target selection
    agent_xy = (int(state.agent.x), int(state.agent.y))
    agent_v  = (int(state.agent.vel_x), int(state.agent.vel_y))

    mode, target, _xy_path = choose_target(world, agent_xy)

    # 3) Plan (A*) if we have a target
    if target is not None:
        actions = planner.plan(agent_xy, agent_v, target)
        if actions:
            ax, ay = actions[0]
        else:
            # No plan found → fallback
            ax, ay = fallback_move(rng, state)
            mode += "+fallback(no-plan)"
    else:
        # No target (fully known, or boxed in) → fallback
        ax, ay = fallback_move(rng, state)
        mode += "+fallback(no-target)"

    # Avoid immediate collision with other players (current positions)
    next_pos = state.agent.pos + state.agent.vel + np.array([ax, ay])
    if any(np.all(next_pos == p.pos) for p in state.players):
        # be conservative: cancel accel
        ax, ay = 0, 0
        mode += "+avoid-player"

    # Stats/log
    world.visited_count[agent_xy[0], agent_xy[1]] += 1
    targ_str = f"{target[0]} {target[1]}" if target is not None else "None"
    log_info(f"MODE={mode} | POS=({agent_xy[0]},{agent_xy[1]}) "
             f"VEL=({agent_v[0]},{agent_v[1]}) TARGET=({targ_str}) ACT=({ax},{ay})")
    return ax, ay

def main():
    log_info("Session start")
    circuit = read_initial_observation()
    # initial stub state
    state: Optional[State] = State(circuit, None, None, [], None)

    # world + planner
    world = WorldModel(circuit.track_shape)
    planner = AStarPlanner(world, v_max=5, max_nodes=20000)

    rng = np.random.default_rng(seed=1)

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            log_info("Judge signaled end (~~~END~~~)")
            # (optional) dump visited counts to CSV:
            # np.savetxt("visited_count.csv", world.visited_count, fmt="%d", delimiter=",")
            return
        ax, ay = calculate_move(world, planner, rng, state)
        send_agent_move(ax, ay)

if __name__ == "__main__":
    main()
