import sys
import enum
import numpy as np

from typing import Optional, NamedTuple

class CellType(enum.Enum):
    # We use NOT_VISIBLE as our "UNKNOWN" state for the global map
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

# MODIFIED: The state now holds the 'global_map'
class State(NamedTuple):
    circuit: Circuit
    global_map: np.ndarray  # This is our persistent memory
    players: list[Player]
    agent: Player

def read_initial_observation() -> Circuit:
    H, W, num_players, visibility_radius = map(int, input().split())
    return Circuit((H, W), num_players, visibility_radius)

# MODIFIED: This function now updates the global_map
def read_observation(old_state: State) -> Optional[State]:
    line = input()
    if line == '~~~END~~~':
        return None
    
    # Read agent and player positions
    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)
    players = []
    circuit_data = old_state.circuit
    for _ in range(circuit_data.num_players):
        pposx, pposy = map(int, input().split())
        # We only care about position for collision checking
        players.append(Player(pposx, pposy, 0, 0))

    # Get a writable copy of our persistent map from the previous state
    new_global_map = old_state.global_map.copy()
    
    R = circuit_data.visibility_radius
    
    # Read the (2R+1) x (2R+1) visible area
    for i in range(2 * R + 1):
        line_cells = [int(a) for a in input().split()]
        
        # Absolute map row index
        map_x = posx - R + i
        if not (0 <= map_x < circuit_data.track_shape[0]):
            continue # This row is off the main map

        for j, cell_val in enumerate(line_cells):
            # Absolute map column index
            map_y = posy - R + j
            if not (0 <= map_y < circuit_data.track_shape[1]):
                continue # This column is off the main map

            # --- This is the new memory logic ---
            # If the cell is NOT 'not visible', update our map.
            # Otherwise, we do nothing, keeping our old knowledge.
            if cell_val != CellType.NOT_VISIBLE.value:
                new_global_map[map_x, map_y] = cell_val

    # Return the new state with the *updated* global map
    return old_state._replace(
        global_map=new_global_map, players=players, agent=agent)

# MODIFIED: Treat "NOT_VISIBLE" (unknown) as non-traversable
def traversable(cell_value: int) -> bool:
    # Only allow movement on cells we *know* are empty, start, or goal.
    return (cell_value == CellType.EMPTY.value or
            cell_value == CellType.START.value or
            cell_value == CellType.GOAL.value)

# MODIFIED: This function now uses the 'global_map'
def valid_line(state: State, pos1: np.ndarray, pos2: np.ndarray) -> bool:
    # Check against our persistent global_map
    track = state.global_map
    
    # Check bounds (also handles cells we've never seen, which are < 0)
    if (np.any(pos1 < 0) or np.any(pos2 < 0) or 
        np.any(pos1 >= track.shape) or np.any(pos2 >= track.shape)):
        return False
        
    # Check if we landed on a non-traversable cell
    # Note: np.floor()...astype(int) is safe for target coordinates
    if not traversable(track[int(np.floor(pos2[0])), int(np.floor(pos2[1]))]):
         return False

    # --- Original line-of-sight check ---
    diff = pos2 - pos1
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = np.sign(diff[0])
        for i in range(abs(diff[0]) + 1):
            x = pos1[0] + i*d
            y = pos1[1] + i*slope*d
            y_ceil = np.ceil(y).astype(int)
            y_floor = np.floor(y).astype(int)
            if (not traversable(track[x, y_ceil])
                    and not traversable(track[x, y_floor])):
                return False
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = np.sign(diff[1])
        for i in range(abs(diff[1]) + 1):
            x = pos1[0] + i*slope*d
            y = pos1[1] + i*d
            x_ceil = np.ceil(x).astype(int)
            x_floor = np.floor(x).astype(int)
            if (not traversable(track[x_ceil, y])
                    and not traversable(track[x_floor, y])):
                return False
    return True

# MODIFIED: This is the new goal-seeking + explorer logic
def calculate_move(rng: np.random.Generator, state: State) -> tuple[int, int]:
    self_pos = state.agent.pos
    self_vel = state.agent.vel

    # --- 1. Find a Target ---
    # Find all known goal cells
    goal_cells = np.argwhere(state.global_map == CellType.GOAL.value)
    target_pos = None
    
    if goal_cells.size > 0:
        # Pick the closest goal as our target
        distances = np.linalg.norm(goal_cells - self_pos, axis=1)
        target_pos = goal_cells[np.argmin(distances)]

    # Helper function to check if a move is valid
    def is_move_valid(pos_after_move):
        return (valid_line(state, self_pos, pos_after_move) and
                (np.all(pos_after_move == self_pos) # Staying still is ok
                 or not any(np.all(pos_after_move == p.pos) for p in state.players)))

    # --- 2. Goal-Seeking Mode ---
    if target_pos is not None:
        best_accel = None
        min_dist = float('inf')

        # Try all 9 possible accelerations
        for i in range(-1, 2):
            for j in range(-1, 2):
                accel = np.array([i, j])
                next_vel = self_vel + accel
                next_pos = self_pos + next_vel
                
                if is_move_valid(next_pos):
                    # This is a valid move, see how good it is
                    dist = np.linalg.norm(next_pos - target_pos)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_accel = (i, j)

        if best_accel is not None:
            return best_accel
        else:
            # All 9 moves are invalid, we're trapped
            print(
                'Greedy: No valid action found. Bracing for impact.',
                file=sys.stderr)
            return (0, 0)

    # --- 3. Explorer Mode (No Goal Seen) ---
    else:
        # Fall back to the original "Lieutenant" random-walk logic
        # to explore the map safely.
        new_center = self_pos + self_vel
        next_move = new_center
        
        if (np.any(next_move != self_pos) and is_move_valid(next_move)
                and rng.random() > 0.1):
            return (0, 0) # 90% chance to coast if it's safe
        else:
            # Check all 9 moves
            valid_moves = []
            valid_stay = None
            for i in range(-1, 2):
                for j in range(-1, 2):
                    accel = np.array([i, j])
                    next_vel_try = self_vel + accel
                    next_pos_try = self_pos + next_vel_try
                    
                    if is_move_valid(next_pos_try):
                        if np.all(self_pos == next_pos_try):
                            valid_stay = (i, j)
                        else:
                            valid_moves.append((i, j))
            
            if valid_moves:
                # Pick a random valid move
                idx = rng.choice(len(valid_moves))
                return valid_moves[idx]
            elif valid_stay is not None:
                # Only safe option is to brake/stay
                return valid_stay
            else:
                # No valid moves at all
                print(
                    'Explorer: No valid action found. Bracing for impact.',
                    file=sys.stderr)
                return (0, 0)

def main():
    print('READY', flush=True)
    circuit = read_initial_observation()
    
    # NEW: Initialize the persistent global_map
    # We fill it with 'NOT_VISIBLE' which our code treats as 'UNKNOWN'
    global_map = np.full(circuit.track_shape, CellType.NOT_VISIBLE.value)
    
    # Create a dummy state for the first call to read_observation
    state: Optional[State] = State(circuit, global_map, [], None)  # type: ignore
    
    # Initialize a random number generator for the explorer mode
    rng = np.random.default_rng()
    
    while True:
        assert state is not None
        # read_observation will update the global_map inside the state
        state = read_observation(state)
        
        if state is None:
            # Game over
            return
            
        # calculate_move will use the global_map for pathfinding
        delta = calculate_move(rng, state)
        
        print(f'{delta[0]} {delta[1]}', flush=True)

if __name__ == "__main__":
    main()