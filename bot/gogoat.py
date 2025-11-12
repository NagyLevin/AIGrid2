import sys
import enum
import numpy as np
from typing import Optional, NamedTuple, Tuple, List, Set, Dict
from collections import defaultdict, deque

# =========================
# Judge protocol structures
# =========================

class CellType(enum.Enum):
    NOT_VISIBLE = 3
    WALL = -1

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
    visible_track: np.ndarray   # we will put a "safety map" here each turn
    players: list[Player]
    agent: Player

# =========================
# Persistent bot memory
# =========================

UNKNOWN = -2  # sentinel for our private global map (unknown/unseen)
GLOBAL_KNOWN: Optional[np.ndarray] = None  # persists whole-track knowledge
VISITED: Set[Tuple[int, int]] = set()
VISIT_COUNT: Dict[Tuple[int, int], int] = defaultdict(int)
LAST_HEADING: Optional[np.ndarray] = None  # grid heading in {-1,0,1}^2
CURRENT_PATH: List[Tuple[int, int]] = []
CURRENT_TARGET: Optional[Tuple[int, int]] = None

# =========================
# IO (Tier-2 Judge Protocol)
# =========================

def read_initial_observation() -> Circuit:
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)

def _init_global_known(track_shape: Tuple[int, int]) -> None:
    global GLOBAL_KNOWN
    if GLOBAL_KNOWN is None:
        GLOBAL_KNOWN = np.full(track_shape, UNKNOWN, dtype=int)

def _update_global_known(agent_pos: Tuple[int, int], window_rows: List[List[int]], R: int) -> None:
    """
    Merge current visible window into GLOBAL_KNOWN.
    We DO NOT treat NOT_VISIBLE as walls; we simply skip updating those cells.
    """
    global GLOBAL_KNOWN
    px, py = agent_pos
    H, W = GLOBAL_KNOWN.shape
    for i in range(2 * R + 1):
        x = px - R + i
        if x < 0 or x >= H:
            continue
        row_vals = window_rows[i]
        y0 = py - R
        # clip row to map bounds
        start_clip = max(0, -y0)
        end_clip = max(0, (py + R + 1) - W)
        vals = row_vals[start_clip: len(row_vals) - end_clip if end_clip > 0 else None]
        y_start = max(0, y0)
        for j, v in enumerate(vals):
            y = y_start + j
            # Only write if it's actually visible (i.e., not 3/NOT_VISIBLE)
            if v != CellType.NOT_VISIBLE.value:
                GLOBAL_KNOWN[x, y] = v

def _derive_safety_from_known() -> np.ndarray:
    """
    Build a safety map for path checking:
    - walls stay WALL
    - unknown (UNKNOWN) is treated conservatively as WALL
    - non-negative cells are traversable
    """
    global GLOBAL_KNOWN
    safety = GLOBAL_KNOWN.copy()
    safety[safety == UNKNOWN] = CellType.WALL.value
    return safety

def read_observation(old_state: State) -> Optional[State]:
    """
    Reads one observation block and updates:
      - GLOBAL_KNOWN (persistent map)
      - returns a State whose visible_track is the derived safety map
    """
    line = input()
    if line == '~~~END~~~':
        return None
    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)

    circuit = old_state.circuit
    _init_global_known(circuit.track_shape)

    players: List[Player] = []
    for _ in range(circuit.num_players):
        ppx, ppy = map(int, input().split())
        players.append(Player(ppx, ppy, 0, 0))

    # Read the (2R+1) rows of the visibility window as raw ints
    R = circuit.visibility_radius
    window_rows: List[List[int]] = []
    for _ in range(2 * R + 1):
        window_rows.append([int(a) for a in input().split()])

    # Update persistent known map (do NOT collapse NOT_VISIBLE into walls)
    _update_global_known((posx, posy), window_rows, R)

    # Safety map: unknown as walls
    safety_track = _derive_safety_from_known()

    return old_state._replace(visible_track=safety_track, players=players, agent=agent)

# =========================
# Geometry / helpers
# =========================

def traversable(cell_val: int) -> bool:
    return cell_val >= 0

def in_bounds(arr: np.ndarray, p: np.ndarray) -> bool:
    return 0 <= p[0] < arr.shape[0] and 0 <= p[1] < arr.shape[1]

def cell_is_free(track: np.ndarray, p: np.ndarray) -> bool:
    return in_bounds(track, p) and traversable(track[p[0], p[1]])

def sign_vec(v: np.ndarray) -> np.ndarray:
    return np.array([int(np.sign(v[0])), int(np.sign(v[1]))], dtype=int)

def rotate_left(d: np.ndarray) -> np.ndarray:
    # (dr, dc) -> (-dc, dr)
    return np.array([-d[1], d[0]], dtype=int)

def rotate_right(d: np.ndarray) -> np.ndarray:
    # (dr, dc) -> (dc, -dr)
    return np.array([d[1], -d[0]], dtype=int)

def choose_heading(agent_vel: np.ndarray) -> np.ndarray:
    """
    Determine the heading used for right-wall reference.
    If stationary, keep last heading; default to east (0,1).
    """
    global LAST_HEADING
    if np.any(agent_vel != 0):
        LAST_HEADING = sign_vec(agent_vel)
    if LAST_HEADING is None or np.all(LAST_HEADING == 0):
        LAST_HEADING = np.array([0, 1], dtype=int)
    return LAST_HEADING

def next_pos_from_action(pos: np.ndarray, vel: np.ndarray, a: np.ndarray) -> np.ndarray:
    new_vel = vel + a
    return pos + new_vel

def collides_with_players(p: np.ndarray, players: List[Player]) -> bool:
    pr, pc = int(p[0]), int(p[1])
    for q in players:
        if q.x == pr and q.y == pc:
            return True
    return False

def valid_line(state: State, p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Straight-line clearance against the state.visible_track (our safety map).
    Mirrors the provided logic: blocked if either the direct cell or the
    adjacent pair around the line are both walls.
    """
    track = state.visible_track
    if (np.any(p1 < 0) or np.any(p2 < 0) or
        np.any(p1 >= track.shape) or np.any(p2 >= track.shape)):
        return False

    diff = p2 - p1
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))
        for i in range(abs(int(diff[0])) + 1):
            x = int(p1[0] + i * d)
            y = p1[1] + i * slope * d
            y0 = int(np.floor(y)); y1 = int(np.ceil(y))
            if (not traversable(track[x, y0]) and not traversable(track[x, y1])):
                return False

    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(int(diff[1])) + 1):
            x = p1[0] + i * slope * d
            y = int(p1[1] + i * d)
            x0 = int(np.floor(x)); x1 = int(np.ceil(x))
            if (not traversable(track[x0, y]) and not traversable(track[x1, y])):
                return False

    return True

# =========================
# Global path planning
# =========================

def known_traversable(val: int) -> bool:
    if val == UNKNOWN or val == CellType.WALL.value:
        return False
    if val == CellType.NOT_VISIBLE.value:
        return False
    return val >= 0

def _neighbors4(x: int, y: int, shape: Tuple[int, int]):
    H, W = shape
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < H and 0 <= ny < W:
            yield nx, ny

def _info_gain(kmap: np.ndarray, cell: Tuple[int, int], radius: int) -> int:
    x, y = cell
    H, W = kmap.shape
    r = max(1, radius)
    gain = 0
    for i in range(max(0, x - r), min(H, x + r + 1)):
        for j in range(max(0, y - r), min(W, y + r + 1)):
            if kmap[i, j] == UNKNOWN:
                gain += 1
    return gain

def _is_frontier_cell(kmap: np.ndarray, x: int, y: int) -> bool:
    if not known_traversable(kmap[x, y]):
        return False
    for nx, ny in _neighbors4(x, y, kmap.shape):
        if kmap[nx, ny] == UNKNOWN:
            return True
    return False

def _collect_frontiers(kmap: np.ndarray) -> List[Tuple[int, int]]:
    H, W = kmap.shape
    out: List[Tuple[int, int]] = []
    for x in range(H):
        for y in range(W):
            if _is_frontier_cell(kmap, x, y):
                out.append((x, y))
    return out

def _reconstruct_path(end: Tuple[int, int],
                      parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]]
                      ) -> List[Tuple[int, int]]:
    path: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = end
    while cur is not None:
        path.append(cur)
        cur = parents.get(cur)
    path.reverse()
    return path

def _bfs_to_values(start: Tuple[int, int],
                   kmap: np.ndarray,
                   goal_values: Set[int]) -> Optional[List[Tuple[int, int]]]:
    q: deque[Tuple[int, int]] = deque([start])
    parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    visited: Set[Tuple[int, int]] = {start}
    while q:
        cell = q.popleft()
        x, y = cell
        if kmap[x, y] in goal_values:
            return _reconstruct_path(cell, parents)
        for nx, ny in _neighbors4(x, y, kmap.shape):
            if (nx, ny) in visited:
                continue
            if not known_traversable(kmap[nx, ny]):
                continue
            visited.add((nx, ny))
            parents[(nx, ny)] = cell
            q.append((nx, ny))
    return None

def _bfs_to_best_frontier(start: Tuple[int, int],
                          frontiers: Set[Tuple[int, int]],
                          kmap: np.ndarray,
                          vis_radius: int) -> Optional[List[Tuple[int, int]]]:
    if not frontiers:
        return None
    q: deque[Tuple[int, int]] = deque([start])
    parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    depth: Dict[Tuple[int, int], int] = {start: 0}
    found_depth: Optional[int] = None
    best_frontier: Optional[Tuple[int, int]] = None
    best_score: Optional[Tuple[int, int, int]] = None  # (-info, visits, depth)

    while q:
        cell = q.popleft()
        d = depth[cell]
        if found_depth is not None and d > found_depth:
            break

        if cell in frontiers and cell != start:
            x, y = cell
            info = _info_gain(kmap, cell, vis_radius)
            visits = VISIT_COUNT.get((x, y), 0)
            score = (-info, visits, d)
            if best_score is None or score < best_score:
                best_score = score
                best_frontier = cell
            found_depth = d
            continue

        for nx, ny in _neighbors4(cell[0], cell[1], kmap.shape):
            if (nx, ny) in depth:
                continue
            if not known_traversable(kmap[nx, ny]):
                continue
            depth[(nx, ny)] = d + 1
            parents[(nx, ny)] = cell
            q.append((nx, ny))

    if best_frontier is None:
        return None

    return _reconstruct_path(best_frontier, parents)

def _closest_path_index(pos: Tuple[int, int],
                        path: List[Tuple[int, int]]) -> int:
    best_idx = 0
    best_dist = float("inf")
    for idx, cell in enumerate(path):
        d = abs(cell[0] - pos[0]) + abs(cell[1] - pos[1])
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx

def _path_distance_to_segment(point: Tuple[int, int],
                              segment: List[Tuple[int, int]]) -> float:
    if not segment:
        return 0.0
    pr, pc = point
    best = float("inf")
    for x, y in segment:
        d = abs(pr - x) + abs(pc - y)
        if d < best:
            best = d
    return float(best)

def _derive_subgoal_from_path(state: State,
                              path: List[Tuple[int, int]],
                              pos: np.ndarray,
                              vel: np.ndarray) -> Tuple[np.ndarray,
                                                         np.ndarray,
                                                         List[Tuple[int, int]],
                                                         float]:
    pos_tuple = (int(pos[0]), int(pos[1]))
    current_idx = _closest_path_index(pos_tuple, path)
    speed = float(np.linalg.norm(vel))
    lookahead = max(1, int(round(speed))) + 2
    target_idx = min(len(path) - 1, current_idx + lookahead)
    best_idx = current_idx
    for idx in range(current_idx + 1, target_idx + 1):
        node = np.array(path[idx], dtype=int)
        if valid_line(state, pos, node):
            best_idx = idx

    if best_idx == current_idx and current_idx < len(path) - 1:
        best_idx = current_idx + 1

    subgoal = np.array(path[best_idx], dtype=int)
    ref_prev = pos if best_idx == 0 else np.array(path[max(best_idx - 1, 0)], dtype=int)
    subgoal_heading = sign_vec(subgoal - ref_prev)
    if np.all(subgoal_heading == 0):
        subgoal_heading = sign_vec(subgoal - pos)

    segment = path[current_idx: min(len(path), current_idx + 6)]
    current_path_dist = _path_distance_to_segment(pos_tuple, segment)
    return subgoal, subgoal_heading, segment, current_path_dist

def compute_global_path(agent_pos: Tuple[int, int],
                        visibility_radius: int
                        ) -> Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]:
    global GLOBAL_KNOWN
    if GLOBAL_KNOWN is None:
        return [], None

    kmap = GLOBAL_KNOWN
    path_to_goal = _bfs_to_values(agent_pos, kmap, {100})
    if path_to_goal is not None:
        return path_to_goal, path_to_goal[-1]

    frontiers = _collect_frontiers(kmap)
    frontier_set = {cell for cell in frontiers if cell != agent_pos}
    if frontier_set:
        path_to_frontier = _bfs_to_best_frontier(agent_pos, frontier_set, kmap, visibility_radius)
        if path_to_frontier is not None:
            return path_to_frontier, path_to_frontier[-1] if path_to_frontier else None

    return [], None

# =========================
# Right-wall subgoal planner
# =========================

def is_known_wall(p: np.ndarray) -> bool:
    global GLOBAL_KNOWN
    if GLOBAL_KNOWN is None:
        return False
    if not in_bounds(GLOBAL_KNOWN, p):
        return False
    return GLOBAL_KNOWN[p[0], p[1]] == CellType.WALL.value

def right_wall_lookahead_subgoal(pos: np.ndarray,
                                 heading: np.ndarray,
                                 safety_map: np.ndarray,
                                 max_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Do a virtual walk with the RIGHT-hand rule over the *safety map*,
    while recording positions where the *right-adjacent* cell is a *known* wall
    in GLOBAL_KNOWN. Return (subgoal_position, subgoal_heading).
    """
    cur = pos.copy()
    h = heading.copy()
    last_hug_pos: Optional[np.ndarray] = None
    last_hug_h: Optional[np.ndarray] = None

    for _ in range(max_steps):
        # If the cell to our right is a KNOWN wall, remember this point.
        rvec = rotate_right(h)
        rcell = cur + rvec
        if is_known_wall(rcell):
            last_hug_pos = cur.copy()
            last_hug_h = h.copy()

        # Move one step by right-hand rule on the *safety* map (unknown treated unsafe)
        # Priority: turn right if free; else forward; else left; else U-turn.
        moved = False
        # 1) try to turn right and move
        nh = rotate_right(h)
        np1 = cur + nh
        if cell_is_free(safety_map, np1):
            cur = np1
            h = nh
            moved = True
        else:
            # 2) try forward
            np2 = cur + h
            if cell_is_free(safety_map, np2):
                cur = np2
                moved = True
            else:
                # 3) try left
                nh2 = rotate_left(h)
                np3 = cur + nh2
                if cell_is_free(safety_map, np3):
                    cur = np3
                    h = nh2
                    moved = True
                else:
                    # 4) last resort: U-turn if possible
                    nh3 = -h
                    np4 = cur + nh3
                    if cell_is_free(safety_map, np4):
                        cur = np4
                        h = nh3
                        moved = True

        if not moved:
            break  # dead end in known-safe area

    # Prefer the furthest place where right neighbor is a KNOWN wall
    if last_hug_pos is not None:
        return last_hug_pos, last_hug_h
    # Fallback: try to go one step forward if safe; else stay
    front = pos + heading
    if cell_is_free(safety_map, front):
        return front, heading
    return pos, heading

# =========================
# Action selection
# =========================

def valid_actions(state: State, pos: np.ndarray, vel: np.ndarray) -> List[np.ndarray]:
    actions: List[np.ndarray] = []
    for ax in (-1, 0, 1):
        for ay in (-1, 0, 1):
            a = np.array([ax, ay], dtype=int)
            new_pos = next_pos_from_action(pos, vel, a)
            if not valid_line(state, pos, new_pos):
                continue
            if collides_with_players(new_pos, state.players):
                continue
            actions.append(a)
    return actions

def score_action(a: np.ndarray,
                 pos: np.ndarray,
                 vel: np.ndarray,
                 subgoal: np.ndarray,
                 subgoal_heading: np.ndarray,
                 safety_map: np.ndarray,
                 path_segment: List[Tuple[int, int]],
                 current_path_dist: float,
                 path_goal: Optional[Tuple[int, int]]) -> float:
    """
    Score favors:
      - moving toward the current subgoal / path,
      - reducing distance to the planned path,
      - smooth progress along the desired heading,
      - visiting new or frontier-adjacent cells.
    """
    global GLOBAL_KNOWN

    new_vel = vel + a
    new_pos = pos + new_vel

    # Alignment toward subgoal (use step direction if moving; else penalize)
    step_dir = sign_vec(new_vel) if np.any(new_vel != 0) else np.array([0, 0], dtype=int)
    goal_vec = subgoal - pos
    goal_dir = sign_vec(goal_vec) if np.any(goal_vec != 0) else np.array([0, 0], dtype=int)
    align = float(step_dir[0] * goal_dir[0] + step_dir[1] * goal_dir[1])  # -2..2

    # Distance improvement (L1)
    d0 = abs(int(subgoal[0] - pos[0])) + abs(int(subgoal[1] - pos[1]))
    d1 = abs(int(subgoal[0] - new_pos[0])) + abs(int(subgoal[1] - new_pos[1]))
    gain = float(d0 - d1)  # positive is good

    # Mild smoothness along heading (prefer not to reverse)
    smooth = float(step_dir[0] * subgoal_heading[0] + step_dir[1] * subgoal_heading[1])

    # Novelty: prefer unvisited cells slightly
    novelty = 0.0
    tnp = (int(new_pos[0]), int(new_pos[1]))
    if tnp not in VISITED:
        novelty = 0.8

    # Path adherence bonus
    path_bonus = 0.0
    if path_segment:
        new_path_dist = _path_distance_to_segment(tnp, path_segment)
        path_bonus = 3.2 * (current_path_dist - new_path_dist)

    # Reaching goal bonus
    goal_bonus = 0.0
    if path_goal is not None and tnp == path_goal:
        goal_bonus = 5.0

    # Encourage stepping onto frontier-adjacent cells
    explore_bonus = 0.0
    if GLOBAL_KNOWN is not None:
        nx, ny = tnp
        if 0 <= nx < GLOBAL_KNOWN.shape[0] and 0 <= ny < GLOBAL_KNOWN.shape[1]:
            if _is_frontier_cell(GLOBAL_KNOWN, nx, ny):
                explore_bonus = 1.1

    # Small penalty for idling
    idle_pen = 0.6 if np.all(a == 0) else 0.0

    score = (
        6.0 * align
        + 2.3 * gain
        + 1.1 * smooth
        + novelty
        + path_bonus
        + goal_bonus
        + explore_bonus
        - idle_pen
    )
    return score

def pick_action_right_wall(state: State) -> Tuple[int, int]:
    """
    Hybrid planner:
      1) Builds a global path either to a known goal or to the best frontier.
      2) Tracks along that path using a pure-pursuit style subgoal.
      3) Falls back to right-wall exploration when no path is available.
    """
    global LAST_HEADING, CURRENT_PATH, CURRENT_TARGET

    safety = state.visible_track
    pos = state.agent.pos
    vel = state.agent.vel

    # Track visits
    tpos = (int(pos[0]), int(pos[1]))
    VISITED.add(tpos)
    VISIT_COUNT[tpos] += 1

    heading = choose_heading(vel)
    visibility_radius = state.circuit.visibility_radius

    path, target = compute_global_path(tpos, visibility_radius)
    use_path = bool(path) and (len(path) > 1 or (target is not None and target != tpos))

    if use_path:
        subgoal, subgoal_h, path_segment, current_path_dist = _derive_subgoal_from_path(state, path, pos, vel)
        CURRENT_PATH = path
        CURRENT_TARGET = target
        path_goal = target
    else:
        max_steps = max(4, 3 * visibility_radius)
        subgoal, subgoal_h = right_wall_lookahead_subgoal(pos, heading, safety, max_steps)
        path_segment = []
        current_path_dist = 0.0
        CURRENT_PATH = []
        CURRENT_TARGET = None
        path_goal = None

    # Enumerate legal actions
    actions = valid_actions(state, pos, vel)
    if not actions:
        # brake toward zero as safest fallback
        a = -sign_vec(vel)
        a = np.clip(a, -1, 1).astype(int)
        return int(a[0]), int(a[1])

    # Score and pick best
    best_a = None
    best_s = -1e18
    for a in actions:
        s = score_action(a, pos, vel, subgoal, subgoal_h, safety, path_segment, current_path_dist, path_goal)
        if s > best_s:
            best_s = s
            best_a = a

    if best_a is None:
        best_a = np.array([0, 0], dtype=int)

    # Update heading guess if we will have a nonzero velocity
    new_vel = vel + best_a
    if np.any(new_vel != 0):
        LAST_HEADING = sign_vec(new_vel)

    return int(best_a[0]), int(best_a[1])

# =========================
# Main
# =========================

def main():
    print('READY', flush=True)
    circuit = read_initial_observation()

    # Prepare initial state; visible_track will be filled after the first read
    H, W = circuit.track_shape
    safety_init = np.full((H, W), CellType.WALL.value, dtype=int)  # placeholder
    dummy_agent = Player(0, 0, 0, 0)
    state: Optional[State] = State(circuit, safety_init, [], dummy_agent)  # type: ignore

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return
        ax, ay = pick_action_right_wall(state)
        print(f"{ax} {ay}", flush=True)

if __name__ == "__main__":
    main()
