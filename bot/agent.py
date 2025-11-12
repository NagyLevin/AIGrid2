import sys
import enum
import math
import os  # <-- 1. IMPORT OS
from collections import defaultdict, deque
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
    visible_track: Optional[np.ndarray]    # safety map: NOT_VISIBLE -> WALL
    visible_raw: Optional[np.ndarray]      # raw window with NOT_VISIBLE intact
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
    R = circuit.visibility_radius  # <--- *** HERE IS THE FIX ***
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

# N, E, S, W in grid coords
DIRS_4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
# 8-directional check
DIRS_8 = [
    (-1, 0), (0, 1), (1, 0), (0, -1), # Cardinal
    (-1, -1), (-1, 1), (1, 1), (1, -1)  # Diagonal
]


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

def find_reachable_zero(state: State, world: 'WorldModel', start_pos: np.ndarray) -> bool:
    """
    Performs a BFS within the *visible window* to find a known,
    traversable, unvisited (visited_count == 0) EMPTY cell.
    """
    if state.agent is None: return False
    
    q = deque([(int(start_pos[0]), int(start_pos[1]))])
    visited = set([(int(start_pos[0]), int(start_pos[1]))])
    
    H, W = world.shape
    R = state.circuit.visibility_radius
    ax, ay = int(state.agent.x), int(state.agent.y)
    
    # Bounding box for visibility
    min_x, max_x = max(0, ax - R), min(H, ax + R + 1)
    min_y, max_y = max(0, ay - R), min(W, ay + R + 1)

    while q:
        x, y = q.popleft()
        
        # Check if this is an unvisited "0 cell"
        if (world.known_map[x, y] == CellType.EMPTY.value and
            world.visited_count[x, y] == 0):
            return True # Found one

        # Use 8-directional search for reachability
        for dx, dy in DIRS_8:
            nx, ny = x + dx, y + dy
            
            # Must be within visibility *and* world bounds
            if (min_x <= nx < max_x and min_y <= ny < max_y and
                (nx, ny) not in visited):
                
                # Must be traversable based on our *known map*
                if world.traversable(nx, ny):
                    visited.add((nx, ny))
                    q.append((nx, ny))
    return False

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
        # This is the "marking" system: 0="0 cell", 1="10 cell", 2="11 cell", etc.
        self.visited_count = np.zeros((H, W), dtype=int)
        self.last_pos: Optional[Tuple[int,int]] = None

        # ascii dump bookkeeping
        self.turn = 0
        self.dump_file = "logs/map_dump.txt" # <-- 1. CHANGED PATH
        self._dump_initialized = False

    def updateWithObservation(self, st: State) -> None:
        if st.visible_raw is None: return
        raw = st.visible_raw
        seen = (raw != CellType.NOT_VISIBLE.value)
        self.known_map[seen] = raw[seen]

    def traversable(self, x: int, y: int) -> bool:
        """Checks if a cell is in-bounds and traversable based on the *known map*."""
        H, W = self.shape
        if not (0 <= x < H and 0 <= y < W): return False
        return is_traversable_val(self.known_map[x, y])

# ────────────────────────────────────────────────────────────────────────────────
# Left-hand wall follower (no hand-flipping, deterministic)
# ────────────────────────────────────────────────────────────────────────────────

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
    --- HYBRID POLICY ---
    Uses a different strategy based on the agent's current cell.
    
    --- CORRIDOR LOGIC ---
    Corridor rule now checks 2 tiles to the left/right to detect
    wider hallways and prevent "bouncing".
    """
    def __init__(self, world: WorldModel) -> None:
        self.world = world
        self.heading: Tuple[int,int] = (0, 1)    # default EAST

    # local sensing helpers (use visible_* so it reflects current frame)
    def _is_wall_local(self, state: State, x: int, y: int) -> bool:
        if state.visible_raw is None: return False
        H, W = state.visible_raw.shape
        if not (0 <= x < H and 0 <= y < W): return False
        return state.visible_raw[x, y] == CellType.WALL.value

    def _is_free_local(self, state: State, x: int, y: int) -> bool:
        """Checks if a cell is traversable based on *current visibility*."""
        if state.visible_track is None: return False
        H, W = state.visible_track.shape
        if not (0 <= x < H and 0 <= y < W): return False
        return state.visible_track[x, y] >= 0

    def _get_visit_count(self, x: int, y: int) -> float:
        """Gets the visit count for a cell, or infinity if out of bounds."""
        H, W = self.world.shape
        if not (0 <= x < H and 0 <= y < W):
            return float('inf')
        return float(self.world.visited_count[x, y])

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

    def _get_fallback_4dir_target(self, state: State, 
                                  pos: Tuple[int,int], 
                                  coords: Tuple[int, ...], 
                                  dirs: Tuple[Tuple[int,int], ...]) -> Tuple[Tuple[int,int], str]:
        """Helper for when 8-dir search fails, reverts to 4-dir LFRB."""
        ax, ay = pos
        l1x, l1y, f1x, f1y, r1x, r1y = coords
        dL, dF, dR, dB = dirs
        
        if self._is_free_local(state, l1x, l1y):
            self.heading = dL
            return (l1x, l1y), "fallback_left"
        if self._is_free_local(state, f1x, f1y):
            self.heading = dF
            return (f1x, f1y), "fallback_forward"
        if self._is_free_local(state, r1x, r1y):
            self.heading = dR
            return (r1x, r1y), "fallback_right"
        
        # True stuck
        return (ax, ay), "stuck"

    def next_grid_target(self, state: State) -> Tuple[Tuple[int,int], str]:
        assert state.agent is not None
        ax, ay = int(state.agent.x), int(state.agent.y)
        self._ensure_heading(state)

        # Get visit count of *current* cell. Note: calculateMove()
        # increments this *before* calling this function.
        # So, 1 = "just landed on a 0-cell"
        # >1 = "just landed on a 10/11/etc. cell"
        current_visit_count = self._get_visit_count(ax, ay)

        dF = self.heading
        dL = left_of(dF)
        dR = right_of(dF)
        dB = back_of(dF)

        # --- Define coords for 1 and 2 tiles away ---
        l1x, l1y = ax + dL[0], ay + dL[1]
        l2x, l2y = ax + dL[0]*2, ay + dL[1]*2
        
        r1x, r1y = ax + dR[0], ay + dR[1]
        r2x, r2y = ax + dR[0]*2, ay + dR[1]*2
        
        f1x, f1y = ax + dF[0], ay + dF[1]
        b1x, b1y = ax + dB[0], ay + dB[1]

        # --- Corridor rule checks 2 tiles deep ---
        # "closer than 3 tiles" means at 1 or 2 tiles
        wall_on_left = self._is_wall_local(state, l1x, l1y) or self._is_wall_local(state, l2x, l2y)
        wall_on_right = self._is_wall_local(state, r1x, r1y) or self._is_wall_local(state, r2x, r2y)
        front_is_free = self._is_free_local(state, f1x, f1y)
        
        if wall_on_left and wall_on_right and front_is_free:
            self.heading = dF
            front_visit_count = self._get_visit_count(f1x, f1y)
            if front_visit_count == 0:
                return (f1x, f1y), "corridor_unvisited"
            else:
                return (f1x, f1y), "corridor_visited"

        # --- HYBRID LOGIC ---
        
        # MODE 1: SEARCH MODE (on a "10+" cell)
        # Use 8-directional "nearest 0" logic
        if current_visit_count > 1:
            candidates = []
            for dx, dy in DIRS_8:
                nx, ny = ax + dx, ay + dy
                if self._is_free_local(state, nx, ny):
                    visit_count = self._get_visit_count(nx, ny)
                    distance = math.hypot(dx, dy) # 1.0 for adjacent, 1.414 for diagonal
                    candidates.append((visit_count, distance, (nx, ny), (dx, dy)))

            if not candidates:
                # No 8-dir moves, fall back to 4-dir L-F-R-B
                return self._get_fallback_4dir_target(
                    state, (ax,ay), 
                    (l1x, l1y, f1x, f1y, r1x, r1y), 
                    (dL, dF, dR, dB)
                )
                
            # Sort by visit_count (lowest first), then by distance (lowest first)
            candidates.sort()
            
            best_vc, _, best_target, best_dir = candidates[0]
            
            # Update heading based on *dominant axis* of chosen direction
            if best_dir != (0,0):
                if abs(best_dir[0]) >= abs(best_dir[1]):
                    self.heading = (int(np.sign(best_dir[0])), 0)
                else:
                    self.heading = (0, int(np.sign(best_dir[1])))

            mode = "search_8dir" if best_vc > 0 else "explore_8dir"
            return best_target, mode

        # MODE 2: EXPLORE MODE (on a "0" cell)
        # Use 4-directional, visit-biased Left-Hand-Rule
        else:
            candidates = []
            if self._is_free_local(state, l1x, l1y):
                candidates.append((self._get_visit_count(l1x, l1y), 0, (l1x, l1y), "turn_left", dL))
            if self._is_free_local(state, f1x, f1y):
                candidates.append((self._get_visit_count(f1x, f1y), 1, (f1x, f1y), "forward", dF))
            if self._is_free_local(state, r1x, r1y):
                candidates.append((self._get_visit_count(r1x, r1y), 2, (r1x, r1y), "turn_right", dR))
            if self._is_free_local(state, b1x, b1y):
                candidates.append((self._get_visit_count(b1x, b1y), 3, (b1x, b1y), "turn_back", dB))

            if not candidates:
                return (ax, ay), "stuck" # No locally-free move
                
            # Sort by visit_count (lowest first), then by L-F-R-B preference (lowest first)
            candidates.sort()
            
            best_vc, _, best_target, best_mode, best_heading = candidates[0]
            
            self.heading = best_heading
            
            if best_vc > 0:
                # We are on a 0-cell, but are forced to move to a 10+ cell
                return best_target, "search_4dir_" + best_mode
            else:
                # This is the normal case: on a 0-cell, moving to another 0-cell
                return best_target, best_mode # e.g., "forward" (onto a 0-cell)

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

    # --- Speed Logic ---
    ax_agent, ay_agent = int(state.agent.x), int(state.agent.y)
    
    # 1. Check for "easy" 0-cells (cardinal directions)
    has_adjacent_zero_4dir = False
    for dx, dy in DIRS_4: 
        nx, ny = ax_agent + dx, ay_agent + dy
        if (world.traversable(nx, ny) and
            world.known_map[nx, ny] == CellType.EMPTY.value and
            world.visited_count[nx, ny] == 0):
            has_adjacent_zero_4dir = True
            break
            
    # 2. Check for *any* 0-cells (including diagonal)
    has_adjacent_zero_8dir = False
    if has_adjacent_zero_4dir:
         has_adjacent_zero_8dir = True # Optimization
    else:
        for dx, dy in DIRS_8: 
            nx, ny = ax_agent + dx, ay_agent + dy
            if (world.traversable(nx, ny) and
                world.known_map[nx, ny] == CellType.EMPTY.value and
                world.visited_count[nx, ny] == 0):
                has_adjacent_zero_8dir = True
                break
            
    # 3. If no adjacent "0 cells", scan visible range for one
    visible_zero_reachable = False
    if not has_adjacent_zero_8dir:
        visible_zero_reachable = find_reachable_zero(state, world, state.agent.pos)

    # 4. This is the "stop beforehand" logic
    force_slow_down = (not has_adjacent_zero_4dir and has_adjacent_zero_8dir) or \
                      (not has_adjacent_zero_8dir and visible_zero_reachable)
    
    # --- "FULL STOP" OVERRIDE ---
    # If we are forced to slow down for a turn AND we are currently moving.
    if force_slow_down and (vx != 0 or vy != 0):
        
        # We need to stop. Let's find the best *safe* way to brake.
        # Priority: 1. Full Brake, 2. Coast (0,0)
        
        possible_brakes = [
            (-int(np.sign(vx)), -int(np.sign(vy))), # Full Brake
            (0, 0)                                  # Coast
        ]
        
        for ax, ay in possible_brakes:
            nvx, nvy = vx + ax, vy + ay
            next_pos = p + v + np.array([ax, ay], dtype=int)
            
            if brakingOk(nvx, nvy, rSafe) and \
               validLineLocal(state, p, next_pos) and \
               not any(np.all(next_pos == q.pos) for q in state.players):
                # This is a safe braking move. Take it immediately.
                return ax, ay
        
    # --- 5. Set target speed (This is the main logic) ---
    
    max_safe = max(1.0, math.sqrt(2 * max(0, rSafe)))
    
    if force_slow_down:
        # Rule: (A) Only diagonal '0' nearby OR (B) Only distant '0' visible
        # -> Limit speed to 1.0 (will apply if vx/vy were already 0)
        target_speed = 1.0
        
    # <-- 2. NEW ACCELERATION LOGIC -->
    # Check the target cell *we are aiming for*
    elif 0 <= target_cell[0] < world.shape[0] and 0 <= target_cell[1] < world.shape[1] and \
         world.visited_count[target_cell[0], target_cell[1]] == 0:
        # Rule: We are on a 0-cell (from policy) and *targeting* another 0-cell
        # -> GO FAST!
        target_speed = max_safe
    # <-- END NEW LOGIC -->
    
    elif mode.startswith("search_") or mode == "corridor_visited" or mode.startswith("fallback"):
        # Rule: In "search mode" (moving to a visited cell), be cautious.
        target_speed = max(1.0, 0.5 * max_safe)
        
    else:
        # Fallback (e.g., on 0-cell, but forced to move to a 10-cell)
        target_speed = max(1.5, 0.7 * max_safe)
    # --- End Speed Logic ---

    to_cell = np.array([target_cell[0], target_cell[1]], dtype=float) - p.astype(float)
    n_to = float(np.linalg.norm(to_cell)) or 1.0
    desired_dir = to_cell / n_to

    best = None    # (score, (ax,ay))

    # This loop is now the FALLBACK if the "FULL STOP" override doesn't trigger
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
            speed_pen = abs(speed_next - target_speed) # This now punishes high speed

            heading_pen = 0.0
            if speed_next > 0.0:
                vel_dir = np.array([nvx, nvy], dtype=float) / max(speed_next, 1e-9)
                heading_pen = (1.0 - float(np.dot(vel_dir, desired_dir))) * 0.8

            # --- Visit Penalty ---
            visit_pen = 0.0
            if 0 <= nx < world.shape[0] and 0 <= ny < world.shape[1]:
                # This high penalty strongly encourages landing on the lowest-count cell
                visit_pen = 100.0 * float(world.visited_count[nx, ny])

            score = (2.0 * dist_cell) + (0.9 * speed_pen) + heading_pen + visit_pen
            cand = (score, (ax, ay))
            if (best is None) or (cand[0] < best[0]):
                best = cand

    if best is not None:
        return best[1]

    # Fallbacks: Try to just brake (This is a final failsafe)
    for ax, ay in ((-np.sign(vx), -np.sign(vy)), (0, 0)):
        nvx, nvy = vx + ax, vy + ay
        nxt = p + v + np.array([ax, ay], dtype=int)
        if brakingOk(nvx, nvy, rSafe) and validLineLocal(state, p, nxt):
            if not any(np.all(nxt == q.pos) for q in state.players):
                return int(ax), int(ay)

    return (0, 0)    # last resort

# ────────────────────────────────────────────────────────────────────────────────
# ASCII dump (writes to file map_dump.txt every turn)
# ────────────────────────────────────────────────────────────────────────────────

def dump_ascii(world: WorldModel, policy: LeftWallPolicy, state: State, mode: str) -> None:
    if state.agent is None:
        return
    H, W = world.shape
    km = world.known_map
    vis = world.visited_count # This is the visit count map

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

    # --- Visit Count Marking ---
    for x in range(H):
        for y in range(W):
            vis_val = vis[x, y]
            # Only mark if it's a "traversable" spot
            if vis_val > 0 and grid[x][y] not in ('#', 'G', 'S'):
                if vis_val < 10:
                    grid[x][y] = str(int(vis_val)) # 1-9
                elif vis_val < 36: # 10 -> 'a', 11 -> 'b', ... 35 -> 'z'
                    grid[x][y] = chr(ord('a') + int(vis_val) - 10)
                else:
                    grid[x][y] = '+' # 36+

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
    # --- Updated Legend ---
    hdr.append("LEGEND: #=WALL  ?=UNKNOWN  .=EMPTY  G=GOAL  S=START  [1-9,a-z,+]=visit count  O=other  A=agent")
    lines = ["\n".join(hdr)]
    for x in range(H):
        lines.append("".join(grid[x]))
    lines.append("")

    # <-- 1. CREATE LOGS FOLDER IF NEEDED -->
    log_dir = os.path.dirname(world.dump_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
    # This is where the cell's number is "raised"
    world.visited_count[ax, ay] += 1

    # Policy now implements the HYBRID logic
    target_cell, mode = policy.next_grid_target(state)
    
    # Accel choice now implements the new "FULL STOP" and "GO FAST" logic
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