import sys
import enum
import math
import heapq
from collections import deque
import numpy as np
from typing import Optional, NamedTuple, Tuple, List

# ----------------------------- Loggolás (egyszerű) ----------------------------

LOG_FILE = "levin.log"

def writeLog(prefix: str, message: str) -> None:
    """
    Egyszerű log: fájlba ír egy sort.

    @param prefix: log prefix (pl. 'JUDGE' / 'Agent' / 'INFO')
    @param message: maga az üzenet
    """
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{prefix}: {message}\n")
            f.flush()
    except Exception as e:
        # Igen tudom, ez túl egyszerű, de hibánál is inkább lássuk mi történt
        print(f"[logging error] {e}", file=sys.stderr)

def logJudge(raw: str) -> None:
    writeLog("JUDGE", raw)

def logAgent(raw: str) -> None:
    writeLog("Agent", raw)

def logInfo(msg: str) -> None:
    writeLog("INFO", msg)

def sendAgentMove(ax: int, ay: int) -> None:
    """
    Elküldi az aktuális lépést a judge felé (stdout).

    @param ax: gyorsulás x
    @param ay: gyorsulás y
    """
    out = f"{ax} {ay}"
    print(out)
    logAgent(out)

# ------------------------- Alap típusok (környezettől) ------------------------

class CellType(enum.Enum):
    GOAL = 100
    START = 1
    WALL = -1
    UNKNOWN = 2         # saját világmodellben: még nem láttuk → tiltott a tervezésben
    EMPTY = 0
    NOT_VISIBLE = 3     # csak az aktuális ablakban; a világmodellbe nem így kerül

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
    visible_track: Optional[np.ndarray]   # safety-map (NOT_VISIBLE->WALL)
    visible_raw: Optional[np.ndarray]     # raw ablak (NOT_VISIBLE megtartva)
    players: List[Player]
    agent: Optional[Player]

# ----------------------------- Bemenet olvasás --------------------------------

def readInitialObservation() -> Circuit:
    """
    Beolvassa az első sort a judge-tól.

    @return Circuit (pályaméret, játékosok száma, látótáv)
    """
    raw = input()
    logJudge(raw)
    H, W, numPlayers, visibilityRadius = map(int, raw.split())
    return Circuit((H, W), numPlayers, visibilityRadius)

def readObservation(oldState: State) -> Optional[State]:
    """
    Egy kör megfigyeléseinek beolvasása.

    @param oldState: előző állapot (csak a circuit kell belőle)
    @return új State vagy None, ha '~~~END~~~'
    """
    line = input()
    logJudge(line)
    if line == '~~~END~~~':
        return None

    posx, posy, velx, vely = map(int, line.split())
    agent = Player(posx, posy, velx, vely)
    circuit = oldState.circuit

    players: List[Player] = []
    for _ in range(circuit.num_players):
        pline = input()
        logJudge(pline)
        pposx, pposy = map(int, pline.split())
        players.append(Player(pposx, pposy, 0, 0))

    H, W = circuit.track_shape
    R = circuit.visibility_radius

    visibleRaw = np.full((H, W), CellType.NOT_VISIBLE.value, dtype=int)
    visibleTrack = np.full((H, W), CellType.WALL.value, dtype=int)  # unseen=WALL

    for i in range(2 * R + 1):
        rowRaw = input()
        logJudge(rowRaw)
        lineVals = [int(a) for a in rowRaw.split()]

        x = posx - R + i
        if x < 0 or x >= H:
            continue
        yStart = posy - R
        yEnd = yStart + (2 * R + 1)

        sliceVals = lineVals[:]
        y0 = yStart
        if yStart < 0:
            sliceVals = sliceVals[-yStart:]
            y0 = 0
        if yEnd > W:
            sliceVals = sliceVals[:-(yEnd - W)]
        y1 = y0 + len(sliceVals)
        if y0 < y1:
            visibleRaw[x, y0:y1] = sliceVals
            # safety map: NOT_VISIBLE -> WALL
            rowSafety = [CellType.WALL.value if v == CellType.NOT_VISIBLE.value else v
                         for v in sliceVals]
            visibleTrack[x, y0:y1] = rowSafety

    return oldState._replace(
        visible_track=visibleTrack, visible_raw=visibleRaw,
        players=players, agent=agent
    )

# ----------------------------- Világmodell ------------------------------------

class WorldModel:
    """
    Tartós térkép:
      known_map[x,y]: amit valaha láttunk (UNKNOWN kezdetben).
      visited_count[x,y]: hányszor álltunk a cellában.

    UNKNOWN-t nem tekintjük bejárhatónak a tervezésnél (óvatos üzem).
    """

    def __init__(self, shape: tuple[int, int]) -> None:
        H, W = shape
        self.shape = shape
        self.known_map = np.full((H, W), CellType.UNKNOWN.value, dtype=int)
        self.visited_count = np.zeros((H, W), dtype=int)

    def updateWithObservation(self, st: State) -> None:
        """
        @param st: State (a raw ablakot használjuk fel)
        """
        assert st.visible_raw is not None
        raw = st.visible_raw
        seen = (raw != CellType.NOT_VISIBLE.value)
        self.known_map[seen] = raw[seen]

    # ---- Frontier keresés ----
    def _trav(self, v: int) -> bool:
        return (v >= 0) and (v != CellType.UNKNOWN.value)

    def _n4(self, x: int, y: int):
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.shape[0] and 0 <= ny < self.shape[1]:
                yield nx, ny

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

# ------------------------- Geometria / ütközés --------------------------------

def isTraversableForPlanning(v: int) -> bool:
    return (v >= 0) and (v != CellType.UNKNOWN.value)

def validLineOnMap(world: WorldModel, p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Egyszerű, mégis szigorú vonal-ellenőrzés a tartós (known_map) térképen.

    @param world: WorldModel
    @param p1: kezdő poz
    @param p2: cél poz
    @return True ha a teljes szakasz (óvatos mód) bejárható
    """
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

# --------------------- Féktávolság segédfüggvények ----------------------------
# ismert sarok-eset: ha fal közelebb van, mint a látósugár, ütközhetünk, hiába stimmel a háromszög-szabály
def tri(n: int) -> int:
    """Háromszögszám: n + (n-1) + ... + 1 = n*(n+1)//2"""
    return n*(n+1)//2

def brakingOk(vx: int, vy: int, rSafe: int) -> bool:
    """
    Féktávolság-invariáns komponensenként: T(|vx|)<=rSafe és T(|vy|)<=rSafe.

    @param vx: sebesség x
    @param vy: sebesség y
    @param rSafe: biztonsági sugár (R - margin)
    """
    return (tri(abs(vx)) <= rSafe) and (tri(abs(vy)) <= rSafe)

# -------------------------- A* állapotrácson ----------------------------------

class AStarPlanner:
    """
    Egyszerű A* a (x,y,vx,vy) rácson, óvatos (féktáv) korlátozással és fordulási büntetéssel,
    hogy ne forduljon vissza könnyen.
    """

    def __init__(self,
                 world: WorldModel,
                 vMax: int,
                 rSafe: int,
                 maxNodes: int = 20000):
        self.world = world
        self.v_max = vMax
        self.R_safe = max(0, rSafe)
        self.max_nodes = maxNodes

        # irányváltás büntetés paraméterei
        self.turn_pen_back = 5.0
        self.turn_pen_half = 2.0
        self.turn_pen_ortho = 0.5

    def heuristicSteps(self, pos: Tuple[int,int], target: Tuple[int,int]) -> float:
        dx = pos[0] - target[0]
        dy = pos[1] - target[1]
        dist = math.hypot(dx, dy)
        # durva alsó korlát lépésekre
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

    def plan(self,
             startPos: Tuple[int,int],
             startVel: Tuple[int,int],
             targetPos: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
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

            for ax in (-1,0,1):
                for ay in (-1,0,1):
                    nvx = self._clampV(vx + ax)
                    nvy = self._clampV(vy + ay)

                    # Féktávolság-invariáns
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
                    stepCost = base + turnp

                    ns = (nx, ny, nvx, nvy)
                    tentative = gCost[(x,y,vx,vy)] + stepCost
                    if tentative < gCost.get(ns, float("inf")):
                        gCost[ns] = tentative
                        parent[ns] = ((x,y,vx,vy), (ax,ay))
                        f = tentative + self.heuristicSteps((nx,ny),(tx,ty))
                        heapq.heappush(pq, (f, counter, ns))
                        counter += 1
        return None

# -------------------------- Célválasztás --------------------------------------

def chooseTarget(world: WorldModel, agentXY: Tuple[int,int]) -> Tuple[str, Optional[Tuple[int,int]], Optional[List[Tuple[int,int]]]]:
    """
    Először cél, ha nincs: legközelebbi frontier, ha az sincs: idle.

    @return (mode, targetCell, pathToFrontierVagyNone)
    """
    km = world.known_map
    goals = np.argwhere(km == CellType.GOAL.value)
    if goals.size > 0:
        sx, sy = agentXY
        dists = [ (abs(int(x)-sx)+abs(int(y)-sy), (int(x),int(y))) for (x,y) in goals ]
        dists.sort()
        return "goal", dists[0][1], None
    path = world.nearestFrontierFrom(agentXY)
    if path:
        return "explore", path[-1], path
    return "idle", None, None

# ----------------------- Fallback (féktáv + forward bias) ---------------------

def fallbackMoveWithBrakeAndBias(state: State, rSafe: int) -> Tuple[int,int]:
    """
    Laza tartalék: érvényes vonal + féktáv feltétel, és preferált irány: v·v' >= 0.

    @param state: látott ablak + agent
    @param rSafe: R - margin
    """
    assert state.agent is not None
    selfPos = state.agent.pos
    v = state.agent.vel
    vx, vy = int(v[0]), int(v[1])

    def traversable(cell_value: int) -> bool:
        return cell_value >= 0

    def validLineLocal(pos1: np.ndarray, pos2: np.ndarray) -> bool:
        track = state.visible_track
        if (np.any(pos1 < 0) or np.any(pos2 < 0) or np.any(pos1 >= track.shape)
                or np.any(pos2 >= track.shape)):
            return False
        diff = pos2 - pos1
        if diff[0] != 0:
            slope = diff[1] / diff[0]; d = np.sign(diff[0])
            for i in range(abs(diff[0]) + 1):
                x = int(pos1[0] + i*d); y = pos1[1] + i*slope*d
                yCeil = int(np.ceil(y)); yFloor = int(np.floor(y))
                if (not traversable(track[x, yCeil]) and
                    not traversable(track[x, yFloor])): return False
        if diff[1] != 0:
            slope = diff[0] / diff[1]; d = np.sign(diff[1])
            for i in range(abs(diff[1]) + 1):
                x = pos1[0] + i*slope*d; y = int(pos1[1] + i*d)
                xCeil = int(np.ceil(x)); xFloor = int(np.floor(x))
                if (not traversable(track[xCeil, y]) and
                    not traversable(track[xFloor, y])): return False
        return True

    newCenter = selfPos + v
    candidates: List[Tuple[Tuple[int,int], float]] = []  # ((ax,ay), rank)

    for ax in (-1,0,1):
        for ay in (-1,0,1):
            nvx, nvy = vx + ax, vy + ay
            if not brakingOk(nvx, nvy, rSafe):
                continue
            nextMove = newCenter + np.array([ax, ay])
            if not validLineLocal(selfPos, nextMove):
                continue
            # forward bias: rangsor
            a = math.hypot(vx, vy); b = math.hypot(nvx, nvy)
            rank = 1.0
            if a > 0 and b > 0:
                cos = (vx*nvx + vy*nvy) / (a*b)
                if cos > 0:      rank = 0.0   # előre
                elif cos == 0:   rank = 0.5   # oldal
                elif cos < 0:    rank = 2.0   # vissza
            elif a == 0 and b > 0:
                rank = 0.8  # nulláról elindulás – oké
            elif b == 0:
                rank = 1.2  # megállás – közepes
            candidates.append(((ax, ay), rank))

    if not candidates:
        # Nincs semmi értelmes: fékezzünk, ha lehet, különben 0,0
        ax = -1 if vx > 0 else (1 if vx < 0 else 0)
        ay = -1 if vy > 0 else (1 if vy < 0 else 0)
        if brakingOk(vx+ax, vy+ay, rSafe):
            return (ax, ay)
        return (0, 0)

    candidates.sort(key=lambda it: it[1])
    return candidates[0][0]

# ---------------------------- Fő döntés ---------------------------------------

def calculateMove(world: WorldModel, planner: AStarPlanner,
                  rng: np.random.Generator, state: State) -> Tuple[int,int]:
    """
    A* (receding horizon): csak az első akciót hajtjuk végre.
    Ha nincs cél vagy nincs terv: fallback, de ugyanúgy féktávolság-őrült óvatos.

    @return (ax, ay)
    """
    assert state.agent is not None
    assert state.visible_raw is not None

    world.updateWithObservation(state)

    R = state.circuit.visibility_radius
    SAFETY_MARGIN = 1
    rSafe = max(0, R - SAFETY_MARGIN)

    agentXY = (int(state.agent.x), int(state.agent.y))
    agentV  = (int(state.agent.vel_x), int(state.agent.vel_y))

    mode, target, _ = chooseTarget(world, agentXY)

    # A*: csak az első akciót hajtjuk végre (receding horizon)
    if target is not None:
        actions = planner.plan(agentXY, agentV, target)
        if actions:
            ax, ay = actions[0]
        else:
            ax, ay = fallbackMoveWithBrakeAndBias(state, rSafe)
            mode += "+fallback(no-plan)"
    else:
        ax, ay = fallbackMoveWithBrakeAndBias(state, rSafe)
        mode += "+fallback(no-target)"

    # Extra óvatosság: ha ütköznénk játékossal, fékezzünk
    nextPos = state.agent.pos + state.agent.vel + np.array([ax, ay])
    if any(np.all(nextPos == p.pos) for p in state.players):
        ax, ay = 0, 0
        mode += "+avoid-player"

    # Log: állapot + cél + akció + féktáv-ok
    vx1, vy1 = agentV
    stopDX, stopDY = tri(abs(vx1)), tri(abs(vy1))
    logInfo(
        f"MODE={mode} | POS=({agentXY[0]},{agentXY[1]}) "
        f"VEL=({vx1},{vy1}) STOP_TRI=({stopDX},{stopDY}) rSafe={rSafe} "
        f"TARGET=({('None' if target is None else f'{target[0]} {target[1]}')}) "
        f"ACT=({ax},{ay})"
    )

    world.visited_count[agentXY[0], agentXY[1]] += 1
    return ax, ay

# -------------------------------- main ----------------------------------------

def main():
    logInfo("Session start")
    circuit = readInitialObservation()
    state: Optional[State] = State(circuit, None, None, [], None)

    world = WorldModel(circuit.track_shape)

    # v_max maradhat nagyobb is, a féktáv ellenőrzés lesz a szigorúbb korlát
    R = circuit.visibility_radius
    SAFETY_MARGIN = 1
    rSafe = max(0, R - SAFETY_MARGIN)

    planner = AStarPlanner(world, vMax=7, rSafe=rSafe, maxNodes=30000)
    rng = np.random.default_rng(seed=1)

    while True:
        assert state is not None
        state = readObservation(state)
        if state is None:
            logInfo("Judge signaled end (~~~END~~~)")
            return
        ax, ay = calculateMove(world, planner, rng, state)
        sendAgentMove(ax, ay)

if __name__ == "__main__":
    main()
