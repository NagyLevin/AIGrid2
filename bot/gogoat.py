import sys
import enum
import numpy as np
import collections  # NEW: We need a deque for the BFS

from typing import Optional, NamedTuple

# --- Enums and Data Structures (Unchanged) ---

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
        return np.array([self.x, self.y])

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vel_x, self.vel_y])

class Circuit(NamedTuple):
    track_shape: tuple[int, int]
    num_players: int
    visibility_radius: int

class State(NamedTuple):
    circuit: Circuit
    global_map: np.ndarray  # Our persistent memory
    players: list[Player]
    agent: Player

# --- Map/Observation Reading (Unchanged) ---
# This part works well and is necessary for memory.

def read_initial_observation() -> Circuit:
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)

def read_observation(old_state: State) -> Optional[State]:
    line = input()
    if line == '~~~END~~~':
        return None
    
    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)
    players = []
    circuit_data = old_state.circuit
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    new_global_map = old_state.global_map.copy()
    R = circuit_data.visibility_radius
    
    for i in range(2 * R + 1):
        line_cells = [int(a) for a in input().split()]
        map_x = posx - R + i
        if not (0 <= map_x < circuit_data.track_shape[0]):
            continue

        for j, cell_val in enumerate(line_cells):
            map_y = posy - R + j
            if not (0 <= map_y < circuit_data.track_shape[1]):
                continue

            if cell_val != CellType.NOT_VISIBLE.value:
                new_global_map[map_x, map_y] = cell_val

    return old_state._replace(
        global_map=new_global_map, players=players, agent=agent)

# --- Validity Checks (with CRASH FIX) ---

def traversable(cell_value: int) -> bool:
    # We can only move on known, open spaces
    return (cell_value == CellType.EMPTY.value or
            cell_value == CellType.START.value or
            cell_value == CellType.GOAL.value)

def valid_line(state: State, pos1: np.ndarray, pos2: np.ndarray) -> bool:
    """
    This is the line-of-sight check, now with boundary checks
    inside the loops to prevent crashes.
    """
    track = state.global_map
    
    # Check start and end bounds
    if (np.any(pos1 < 0) or np.any(pos2 < 0) or 
        np.any(pos1 >= track.shape) or np.any(pos2 >= track.shape)):
        return False

    # Check landing spot traversability
    try:
        p2_int = pos2.astype(int)
        if not traversable(track[p2_int[0], p2_int[1]]):
            return False
    except IndexError:
        return False # Landed out of bounds

    diff = pos2 - pos1
    
    # Check vertical/diagonal walls
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = np.sign(diff[0])
        for i in range(abs(diff[0]) + 1):
            x = int(pos1[0] + i*d)
            y = pos1[1] + i*slope*d
            y_ceil = np.ceil(y).astype(int)
            y_floor = np.floor(y).astype(int)
            
            # --- CRASH FIX ---
            if not (0 <= x < track.shape[0] and
                    0 <= y_ceil < track.shape[1] and
                    0 <= y_floor < track.shape[1]):
                return False
            # --- END FIX ---
            
            if (not traversable(track[x, y_ceil])
                    and not traversable(track[x, y_floor])):
                return False

    # Check horizontal/diagonal walls
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = np.sign(diff[1])
        for i in range(abs(diff[1]) + 1):
            x = pos1[0] + i*slope*d
            y = int(pos1[1] + i*d)
            x_ceil = np.ceil(x).astype(int)
            x_floor = np.floor(x).astype(int)

            # --- CRASH FIX ---
            if not (0 <= y < track.shape[1] and
                    0 <= x_ceil < track.shape[0] and
                    0 <= x_floor < track.shape[0]):
                return False
            # --- END FIX ---

            if (not traversable(track[x_ceil, y])
                    and not traversable(track[x_floor, y])):
                return False
    return True

# --- NEW: BFS Pathfinding Map ---

def create_distance_map(global_map: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Creates a map where each cell's value is the
    grid-distance to the nearest target.
    """
    distance_map = np.full(global_map.shape, np.inf)
    queue = collections.deque()
    
    # Add all targets to the queue with distance 0
    for x, y in targets:
        if traversable(global_map[x, y]):
            distance_map[x, y] = 0
            queue.append((x, y))

    # 8-directional neighbors
    neighbors = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]

    while queue:
        x, y = queue.popleft()
        current_dist = distance_map[x, y]
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if not (0 <= nx < global_map.shape[0] and 0 <= ny < global_map.shape[1]):
                continue
                
            # If the neighbor is traversable AND we found a shorter path
            if traversable(global_map[nx, ny]) and distance_map[nx, ny] == np.inf:
                distance_map[nx, ny] = current_dist + 1
                queue.append((nx, ny))
                
    return distance_map

# --- NEW: "Path-Guided Greedy" Agent Logic ---

def calculate_move(rng: np.random.Generator, state: State) -> tuple[int, int]:
    
    self_pos = state.agent.pos
    self_vel = state.agent.vel

    # 1. Find Targets
    targets = np.argwhere(state.global_map == CellType.GOAL.value)
    
    if targets.size == 0:
        # No goals visible, switch to explore mode
        targets = np.argwhere(state.global_map == CellType.NOT_VISIBLE.value)

    if targets.size == 0:
        # No goals AND no unknown cells. We're done or trapped.
        print("No targets found. Braking.", file=sys.stderr)
        return (0, 0)
        
    # 2. NEW: Create the BFS distance map
    distance_map = create_distance_map(state.global_map, targets)

    # 3. Score all 9 possible moves
    best_score = float('inf')
    best_move = (0, 0) # Default to braking
    best_vel_norm = float('inf')

    for ax in range(-1, 2):
        for ay in range(-1, 2):
            accel = np.array([ax, ay])
            next_vel = self_vel + accel
            next_pos = self_pos + next_vel
            
            # Check for validity (THIS IS THE CRITICAL PART)
            is_valid = False
            try:
                # Use the fixed, robust valid_line check
                if valid_line(state, self_pos, next_pos):
                    # Check player collisions
                    if not any(np.all(next_pos.astype(int) == p.pos) for p in state.players):
                        is_valid = True
            except Exception as e:
                # Catch any unexpected crash
                print(f"Error checking move: {e}", file=sys.stderr)
                is_valid = False
            
            if is_valid:
                # This is a valid move. Score it.
                # NEW SCORE: Look up the distance from our BFS map
                next_pos_int = next_pos.astype(int)
                score = distance_map[next_pos_int[0], next_pos_int[1]]
                
                vel_norm = np.linalg.norm(next_vel)
                
                if score < best_score:
                    # This is the new best move
                    best_score = score
                    best_move = (ax, ay)
                    best_vel_norm = vel_norm
                elif score == best_score:
                    # Tie-breaker: prefer lower speed
                    if vel_norm < best_vel_norm:
                        best_score = score
                        best_move = (ax, ay)
                        best_vel_norm = vel_norm
            else:
                # Invalid moves get a score of infinity
                pass

    if best_score == float('inf'):
        # All 9 moves were invalid, or led to un-pathable areas.
        print("All moves are invalid or lead to infinity! Braking.", file=sys.stderr)
        return (0, 0)

    return best_move


def main():
    print('READY', flush=True)
    circuit = read_initial_observation()
    
    # Initialize the persistent global_map
    global_map = np.full(circuit.track_shape, CellType.NOT_VISIBLE.value)
    
    state: Optional[State] = State(circuit, global_map, [], None)  # type: ignore
    
    # RNG is not used in this approach, but we keep the object
    rng = np.random.default_rng()
    
    while True:
        assert state is not None
        state = read_observation(state)
        
        if state is None:
            return
            
        delta = calculate_move(rng, state)
        
        print(f'{delta[0]} {delta[1]}', flush=True)

if __name__ == "__main__":
    main()