import sys
import enum
import numpy as np
import collections
from typing import Optional, NamedTuple, List, Tuple, Dict, Set

# --- Enums and Data Structures (Unchanged) ---

class CellType(enum.Enum):
    NOT_VISIBLE = 3
    WALL = -1
    EMPTY = 0
    START = 1
    GOAL = 100
    UNKNOWN = -2 

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

# --- WorldModel (Simplified) ---
# We only need the map and traversability checks.
# All complex state (penalties, etc.) is gone.

class WorldModel:
    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.global_map = np.full(shape, CellType.UNKNOWN.value, dtype=int)
        
    def update_map(self, posx: int, posy: int, R: int, lines: List[str]):
        for i, line_str in enumerate(lines):
            line_cells = [int(a) for a in line_str.split()]
            map_x = posx - R + i
            if not (0 <= map_x < self.shape[0]):
                continue

            for j, cell_val in enumerate(line_cells):
                map_y = posy - R + j
                if not (0 <= map_y < self.shape[1]):
                    continue

                if cell_val != CellType.NOT_VISIBLE.value:
                    self.global_map[map_x, map_y] = cell_val

    def is_traversable(self, x: int, y: int) -> bool:
        if not (0 <= x < self.shape[0] and 0 <= y < self.shape[1]):
            return False
        val = self.global_map[x, y]
        return (val == CellType.EMPTY.value or
                val == CellType.START.value or
                val == CellType.GOAL.value)

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Gets 4-directional traversable neighbors for 2D BFS."""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_traversable(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def find_targets(self, agent_xy: Tuple[int, int]) -> np.ndarray:
        """Finds all goals, or if none, all unknown cells."""
        goals = np.argwhere(self.global_map == CellType.GOAL.value)
        if goals.size > 0:
            return goals
            
        # No goals, find all frontiers (traversable cells next to unknown)
        frontiers = []
        traversable_cells = np.argwhere(
            (self.global_map == CellType.EMPTY.value) |
            (self.global_map == CellType.START.value)
        )
        
        for x, y in traversable_cells:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.shape[0] and 0 <= ny < self.shape[1] and
                        self.global_map[nx, ny] == CellType.UNKNOWN.value):
                    frontiers.append((x, y))
                    break 
                    
        if frontiers:
            return np.array(frontiers)
        
        return np.array([]) 

# --- State Tuple (Unchanged) ---
class State(NamedTuple):
    circuit: Circuit
    world: WorldModel
    players: list[Player]
    agent: Player

# --- I/O Functions (Unchanged) ---
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

    R = circuit_data.visibility_radius
    lines = [input() for _ in range(2 * R + 1)]
    
    old_state.world.update_map(posx, posy, R, lines)

    return old_state._replace(players=players, agent=agent)

# --- valid_line Function (Unchanged) ---
def valid_line(world: WorldModel, pos1: np.ndarray, pos2: np.ndarray) -> bool:
    track = world.global_map
    shape = world.shape
    
    if (np.any(pos1 < 0) or np.any(pos2 < 0) or 
        np.any(pos1 >= shape) or np.any(pos2 >= shape)):
        return False

    try:
        p2_int = pos2.astype(int)
        if not world.is_traversable(p2_int[0], p2_int[1]):
            return False
    except IndexError:
        return False

    diff = pos2 - pos1
    
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = np.sign(diff[0])
        for i in range(abs(diff[0]) + 1):
            x = int(pos1[0] + i*d)
            y = pos1[1] + i*slope*d
            y_ceil = np.ceil(y).astype(int)
            y_floor = np.floor(y).astype(int)
            
            if not (0 <= x < shape[0] and 0 <= y_ceil < shape[1] and 0 <= y_floor < shape[1]):
                return False
            if (not world.is_traversable(x, y_ceil) and not world.is_traversable(x, y_floor)):
                return False

    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = np.sign(diff[1])
        for i in range(abs(diff[1]) + 1):
            x = pos1[0] + i*slope*d
            y = int(pos1[1] + i*d)
            x_ceil = np.ceil(x).astype(int)
            x_floor = np.floor(x).astype(int)

            if not (0 <= y < shape[1] and 0 <= x_ceil < shape[0] and 0 <= x_floor < shape[0]):
                return False
            if (not world.is_traversable(x_ceil, y) and not world.is_traversable(x_floor, y)):
                return False
    return True

# --- **** NEW: 2D BFS Pathfinding **** ---

def find_path_bfs_2d(world: WorldModel, start: Tuple[int, int],
                     targets: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """
    Finds a 2D grid path to the nearest target.
    This is fast and runs every turn.
    """
    if not world.is_traversable(start[0], start[1]):
        # We are on a non-traversable tile? Try to find a neighbor
        for nx, ny in world.get_neighbors(start[0], start[1]):
             start = (nx, ny) # Start from the first valid neighbor
             break
        else:
             return None # Truly trapped

    target_set = set(map(tuple, targets))
    if not target_set:
        return None
        
    q = collections.deque([(start[0], start[1])])
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    
    while q:
        x, y = q.popleft()
        
        if (x, y) in target_set:
            path = []
            curr = (x, y)
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            return path[::-1] # Return from start to goal

        for nx, ny in world.get_neighbors(x, y):
            if (nx, ny) not in parent:
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))
                
    return None 

# --- **** NEW: calculate_move Function **** ---

def calculate_move(state: State) -> tuple[int, int]:
    
    world = state.world
    agent = state.agent
    agent_xy = (agent.x, agent.y)
    
    # --- 1. Find all possible targets ---
    targets = world.find_targets(agent_xy)
    
    if targets.size == 0:
        print("No targets found. Braking.", file=sys.stderr)
        return (0, 0) # No goals, no frontiers. We're done.

    # --- 2. Run the fast 2D BFS to get a path ---
    path = find_path_bfs_2d(world, agent_xy, targets)

    if path is None or len(path) < 2:
        # No path found, or we are on the target.
        # This can happen if we are trapped.
        print("No 2D path found. Braking.", file=sys.stderr)
        return (0, 0)
        
    # --- 3. Find our "target" cell ---
    # Our target is the *next step* on the path.
    # We use a point a few steps ahead for a smoother ride.
    waypoint_idx = min(len(path) - 1, 3) # Aim 3 steps ahead
    target_cell = np.array(path[waypoint_idx])

    # --- 4. Check all 9 moves ---
    best_dist = float('inf')
    best_move = (0, 0) # Default to braking
    
    player_positions = {tuple(p.pos) for p in state.players}

    for ax in range(-1, 2):
        for ay in range(-1, 2):
            accel = np.array([ax, ay])
            next_vel = agent.vel + accel
            next_pos = agent.pos + next_vel

            if not valid_line(world, agent.pos, next_pos):
                continue
                
            if tuple(next_pos.astype(int)) in player_positions:
                continue
                
            # This is a valid move. Score it.
            # Score is just the distance to the target cell.
            dist_to_target = np.linalg.norm(next_pos - target_cell)
            
            if dist_to_target < best_dist:
                best_dist = dist_to_target
                best_move = (ax, ay)
                
    if best_dist == float('inf'):
        # All 9 moves were invalid. We are truly trapped.
        return (0, 0)
        
    return best_move


def main():
    print('READY', flush=True)
    circuit = read_initial_observation()
    
    world = WorldModel(circuit.track_shape)
    
    state: Optional[State] = State(circuit, world, [], None) # type: ignore
    
    while True:
        assert state is not None
        state = read_observation(state)
        
        if state is None:
            return
            
        delta = calculate_move(state)
        
        print(f'{delta[0]} {delta[1]}', flush=True)

if __name__ == "__main__":
    main()