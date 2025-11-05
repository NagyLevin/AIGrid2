import sys
import enum
import numpy as np
from typing import Optional, NamedTuple

# ---- Alap típusok a judge-nek megfelelően ----

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
    visible_track: Optional[np.ndarray]   # (H, W) – ebbe beírjuk a lokális ablakot „rávetítve”
    players: list[Player]
    agent: Optional[Player]

# ---- Beolvasás: pontosan a judge formátuma szerint ----

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

    players: list[Player] = []
    for _ in range(circuit.num_players):
        pposx, pposy = map(int, input().split())
        players.append(Player(pposx, pposy, 0, 0))

    # A judge MINDIG teljes 2R+1 x 2R+1 ablakot küld (nem klippel),
    # falat tesz pályán kívülre, NOT_VISIBLE-t a körön kívülre.
    # Mi ezt visszavetítjük a globális (H,W) mátrixba.
    H, W = circuit.track_shape
    R = circuit.visibility_radius
    visible_track = np.full((H, W), CellType.WALL.value, dtype=int)

    for i in range(2 * R + 1):
        row = [int(a) for a in input().split()]   # pontosan 2R+1 elem
        x = posx - R + i
        if 0 <= x < H:
            y_start = posy - R
            y_end = posy + R + 1
            # vágás a pályahatárhoz (itt most csak akkor kell, ha a középpont szélen van)
            loc = row
            ys = y_start
            if y_start < 0:
                loc = loc[-y_start:]
                ys = 0
            if y_end > W:
                loc = loc[:-(y_end - W)]
            ye = ys + len(loc)
            if ys < ye:
                visible_track[x, ys:ye] = loc

    # óvatosság: ahol NOT_VISIBLE, tekintsük falnak (mint a minta-bot)
    visible_track[visible_track == CellType.NOT_VISIBLE.value] = CellType.WALL.value

    return old_state._replace(visible_track=visible_track, players=players, agent=agent)

# ---- Ütközésvizsgálat (mint a judge/minta-bot logikája) ----

def traversable(v: int) -> bool:
    return v >= 0  # minden nem-negatív mehet

def valid_line(state: State, p1: np.ndarray, p2: np.ndarray) -> bool:
    track = state.visible_track
    if (np.any(p1 < 0) or np.any(p2 < 0) or
        p1[0] >= track.shape[0] or p1[1] >= track.shape[1] or
        p2[0] >= track.shape[0] or p2[1] >= track.shape[1]):
        return False

    diff = p2 - p1
    # kelet-nyugat vizsgálat
    if diff[0] != 0:
        slope = diff[1] / diff[0]
        d = int(np.sign(diff[0]))
        for i in range(abs(diff[0]) + 1):
            x = int(p1[0] + i*d)
            y = p1[1] + i*slope*d
            y_ceil = int(np.ceil(y)); y_floor = int(np.floor(y))
            if (not traversable(track[x, y_ceil]) and
                not traversable(track[x, y_floor])):
                return False
    # észak-dél vizsgálat
    if diff[1] != 0:
        slope = diff[0] / diff[1]
        d = int(np.sign(diff[1]))
        for i in range(abs(diff[1]) + 1):
            x = p1[0] + i*slope*d
            y = int(p1[1] + i*d)
            x_ceil = int(np.ceil(x)); x_floor = int(np.floor(x))
            if (not traversable(track[x_ceil, y]) and
                not traversable(track[x_floor, y])):
                return False
    return True

# ---- Egyszerű, de biztonságos akcióválasztó (delta ∈ {-1,0,1}^2) ----

def choose_delta(state: State, rng: np.random.Generator) -> tuple[int, int]:
    self_pos = state.agent.pos
    center = self_pos + state.agent.vel

    def valid_target(tgt: np.ndarray) -> bool:
        # teljes vonal legyen bejárható és ne ütközzön játékossal
        return (valid_line(state, self_pos, tgt) and
                (np.all(tgt == self_pos) or not any(np.all(tgt == p.pos) for p in state.players)))

    # 1) ha a „középpont” (pos+vel) már jó, 90% eséllyel maradjunk (0,0)
    if np.any(center != self_pos) and valid_target(center) and rng.random() > 0.1:
        return (0, 0)

    # 2) különben keressünk egy érvényes elmozdulást a 9 lehetőségből
    valid_moves = []
    stay = None
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            tgt = center + np.array([dx, dy])
            if valid_target(tgt):
                if np.all(tgt == self_pos):
                    stay = (dx, dy)
                else:
                    valid_moves.append((dx, dy))

    if valid_moves:
        return tuple(rng.choice(valid_moves))
    if stay is not None:
        return stay

    # 3) végső fallback: álljunk (0,0)
    return (0, 0)

# ---- main: szigorú I/O, NINCS loggolás/stdout-zaj ----

def main():
    print("READY", flush=True)  # kötelező kézfogás
    circuit = read_initial_observation()
    state: Optional[State] = State(circuit, None, [], None)  # type: ignore
    rng = np.random.default_rng(seed=1)

    while True:
        assert state is not None
        state = read_observation(state)
        if state is None:
            return
        dx, dy = choose_delta(state, rng)
        # tutira egész és -1..1:
        dx = -1 if dx < -1 else (1 if dx > 1 else int(dx))
        dy = -1 if dy < -1 else (1 if dy > 1 else int(dy))
        print(f"{dx} {dy}", flush=True)

if __name__ == "__main__":
    main()
