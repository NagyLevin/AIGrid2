# Hybrid Local–Global Navigation Bot — README


> TL;DR pipeline per tick  
> 1) Confirm last move (success vs. collision) → update penalties & freeze if needed.  
> 2) Update the world model with the newly visible window.  
> 3) If frozen: output `0 0`. Otherwise:  
> 4) Pick/refresh a **subgoal** (nearest goal or frontier).  
> 5) Plan a coarse 2D path; follow it with a **pure‑pursuit** driver and **adaptive speed**.  
> 6) If that fails: try **4D A\*** (x,y,vx,vy) toward the subgoal.  
> 7) If that fails: use a **local fallback** that prefers safety, novelty, and anti‑backtrack.  
> 8) Emit `(ax, ay)` in `{-1,0,+1}` and **tentatively** remember the edge to verify next tick.

---

## Table of Contents

- [I/O Protocol](#io-protocol)
- [Core Data Types](#core-data-types)
  - [`CellType` enum](#celltype-enum)
  - [`Player`](#player)
  - [`Circuit`](#circuit)
  - [`State`](#state)
- [Global Helpers](#global-helpers)
  - [`isTraversableForPlanning`](#istraversableforplanning)
  - [`tri`](#tri)
  - [`brakingOk`](#brakingok)
- [World Model](#world-model)
  - **Fields** (persistent state)
  - **Methods** (per‑tick updates & queries)
- [Geometry / Line-of-Sight](#geometry--line-of-sight)
  - [`validLineOnMap`](#validlineonmap)
  - [`validLineLocal`](#validlinelocal)
- [Coarse 2D Planner (A\*)](#coarse-2d-planner-a)
  - [`CoarsePlanner2D` fields & methods](#coarseplanner2d-fields--methods)
- [Velocity‑Aware 4D Planner (A\*)](#velocityaware-4d-planner-a)
  - [`AStarPlanner` fields & methods](#astarplanner-fields--methods)
- [Subgoal Policy](#subgoal-policy)
  - [`_choose_committed_target`](#_choose_committed_target)
- [Local Fallback](#local-fallback)
  - [`fallbackMoveWithBrakeAndBias`](#fallbackmovewithbrakeandbias)
- [Pure‑Pursuit Driver & Speed Control](#purepursuit-driver--speed-control)
  - [`_furthest_visible_on_path`](#_furthest_visible_on_path)
  - [`_path_curvature`](#_path_curvature)
  - [`_target_speed_from_context`](#_target_speed_from_context)
  - [`_score_accel`](#_score_accel)
  - [`pure_pursuit_move`](#pure_pursuit_move)
- [Decision Loop](#decision-loop)
  - [`calculateMove`](#calculatemove)
- [Program Entry Point](#program-entry-point)
  - [`main`](#main)
- [Constants & Tunables](#constants--tunables)
- [Control Flow Summary](#control-flow-summary)
- [Extension Points](#extension-points)

---

## I/O Protocol

**Initial line:**  
`H W num_players visibility_radius`

**Each tick:**

1. Own agent: `x y vx vy`
2. Then `num_players` lines, each: `px py` for the other players.
3. Then `2*R + 1` rows, each with `2*R + 1` integers: the local window centered on `(x, y)`, where `R = visibility_radius`.

Two local views are constructed:

- `visible_raw`: full‑map shaped array; outside the visible window cells are set to `NOT_VISIBLE (3)`.
- `visible_track`: same shape; but `NOT_VISIBLE` cells are **conservatively replaced with `WALL`** to enable safe local planning.

Termination signal: a line with `~~~END~~~`.

---

## Core Data Types

### `CellType` enum

- `GOAL = 100` — goal cells
- `START = 1` — start cells
- `WALL = -1` — impassable
- `UNKNOWN = 2` — not yet observed (treated specially in costs)
- `EMPTY = 0` — traversable
- `NOT_VISIBLE = 3` — not visible **this tick** (mapped to WALL in `visible_track`)

### `Player`

Fields:
- `x: int`, `y: int` — grid coordinates
- `vel_x: int`, `vel_y: int` — velocity components

Properties:
- `pos -> np.ndarray([x, y])`
- `vel -> np.ndarray([vel_x, vel_y])`

### `Circuit`

- `track_shape: (H, W)`
- `num_players: int`
- `visibility_radius: int`

### `State`

- `circuit: Circuit`  
- `visible_track: Optional[np.ndarray]` — local “safe” view (NOT_VISIBLE→WALL)
- `visible_raw: Optional[np.ndarray]` — local raw window (with NOT_VISIBLE)
- `players: list[Player]` — positions of other players
- `agent: Optional[Player]` — our agent state

---

## Global Helpers

### `isTraversableForPlanning(v: int) -> bool`

Returns `True` iff a cell value is considered traversable for path planning:  
`(v >= 0) and (v != CellType.UNKNOWN)`.

### `tri(n: int) -> int`

Triangular number `n*(n+1)/2`. Used to approximate per‑axis braking distance.

### `brakingOk(vx: int, vy: int, rSafe: int) -> bool`

Checks that `tri(|vx|) <= rSafe` and `tri(|vy|) <= rSafe`. This restricts speeds to those that can be safely stopped within the visible horizon `rSafe` (we use `R-1`).

---

## World Model

### Persistent fields

- `shape: (H, W)` — grid size
- `known_map: np.ndarray(HxW)` — our best known global map (updated from `visible_raw`)
- `visited_count: np.ndarray(HxW)` — count of visits per node
- `edge_visits: Dict[(x,y,nx,ny) -> int]` — how often we traversed an edge
- `backtrail: deque[(x,y)], maxlen=80` — short history of visited nodes
- `prev_pos: Optional[(x,y)]` — previous position
- `last_dir: np.ndarray([dx,dy])` — last motion direction (for heading preference)

Exploration state:
- `tried_exits: Dict[(x,y) -> set[(dx,dy)]]` — for DFS‑style branching
- `branch_stack: list[(x,y)]` — junction stack to enable backtracking

Subgoal state:
- `commit_target: Optional[(x,y)]` — currently committed subgoal cell
- `commit_ttl: int` — time‑to‑live for commitment (default 40)
- `stalled_steps: int` — increments if we don’t get closer to the subgoal
- `_COMMIT_TTL_DEFAULT = 40`
- `_STALL_RESET = 8`
- `no_backtrack_lock: int` — temporary lock to avoid step‑back oscillation

Collision / freeze bookkeeping:
- `pending_edge: Optional[(x,y,nx,ny)]` — the edge we **intended** last tick
- `predicted_next: Optional[(nx,ny)]` — expected next node (if success)
- `last_from: Optional[(x,y)]` — where the last move started
- `freeze_ticks_left: int` — if collision detected, we output `0 0` for a few ticks

### Methods

- `updateWithObservation(st: State) -> None`  
  Copies all **currently visible** cells from `visible_raw` into `known_map`.

- `confirm_or_reject_last_move(st: State) -> None`  
  Compares actual `agent` position vs. `predicted_next` from the previous tick:  
  - If equal → success: credit the `pending_edge`.  
  - If not equal → **collision**: add heavy penalties to that edge, set a freeze window, mark `predicted_next` as `WALL` (conservative), drop the current subgoal (forces local replanning).

- `_n4(x,y)` → generator of 4‑neighbors within bounds, yielding `(nx,ny,(dx,dy))`.
- `trav(x,y) -> bool` → `isTraversableForPlanning(known_map[x,y])`.
- `traversable_neighbors(x,y) -> list[(nx,ny,(dx,dy))]`.
- `has_unknown_neighbor(x,y) -> bool` → frontier detector.
- `frontierCells() -> list[(x,y)]` → all traversable cells that border unknowns.
- `nearestFrontierFrom(start) -> Optional[list[(x,y)]]` → BFS to the closest frontier; returns a path or `None`.
- `edge_count(x,y,nx,ny) -> int` and `touch_edge(x,y,nx,ny) -> None` → edge stats.
- `_info_gain(c) -> int` → number of UNKNOWN cells in a 3×3 around `c`.
- `leads_to_frontier(start_xy, max_expansions=3000) -> Optional[(frontier_cell, info_gain)]` → lightweight reachability probe for exits scoring.

---

## Geometry / Line‑of‑Sight

### `validLineOnMap(world, p1, p2) -> bool`

Checks the segment `p1→p2` on the **global known map**. Uses float stepping along x/y and checks both `ceil` and `floor` neighbors to be conservative. Rejects if out of bounds or any tested cell is not traversable for planning.

### `validLineLocal(state, p1, p2) -> bool`

Same as above, but on the **local safe view** `visible_track` (where NOT_VISIBLE has already been converted to WALL).

---

## Coarse 2D Planner (A*)

### `CoarsePlanner2D` fields & methods

Fields (tunables):
- `cost_unknown = 5.0`
- `cost_empty = 1.0`
- `cost_goal  = 1.0`
- `cost_recent_back = 60.0` — strong penalty for very recent backtrail
- `cost_old_back    = 8.0`  — milder penalty for older backtrail

Methods:
- `_neighbors8(x,y)` → yields `(nx,ny,step_len)` for 8‑neighbors (1 or √2).
- `_cell_cost(v, cell)` → per‑cell base cost + backtrail penalty using `world.backtrail` recency.
- `plan_path(start, target) -> Optional[list[(x,y)]]`  
  A* on `world.known_map` with 8‑connectivity and the above costs; returns a path (start..target) or `None`.

---

## Velocity‑Aware 4D Planner (A*)

### `AStarPlanner` fields & methods

Constructor args:
- `vMax: int` — max absolute velocity per axis
- `rSafe: int` — braking horizon; usually `R-1`
- `maxNodes: int` — expansion cap (fails fast if exceeded)

Penalty weights:
- `turn_pen_back = 6.0` (reverse)
- `turn_pen_half = 2.0` (sharp)
- `turn_pen_ortho = 0.6` (orthogonal)
- `w_backtrack = 6.0`
- `w_edge_repeat = 1.0`
- `w_visit = 0.2`

Methods:
- `heuristicSteps(pos, target) -> float` — distance divided by `(vMax+1)`.
- `_clampV(v) -> int` — clamp velocity to `[-vMax, vMax]`.
- `_turnPenalty(vx,vy,nvx,nvy) -> float` — heading change penalty from cosine.
- `plan(startPos, startVel, targetPos) -> Optional[list[(ax,ay)]]`  
  4D A* that only expands moves passing `brakingOk` and `validLineOnMap`. Returns the best **acceleration sequence** from start to goal; usually the caller takes `actions[0]` only.

---

## Subgoal Policy

### `_choose_committed_target(world, agentXY) -> Optional[(x,y)]`

Rules:
1. If any `GOAL` cell is known: commit to the nearest.  
2. If the agent’s current cell isn’t traversable: commit to the end of a BFS path to the **nearest frontier**.  
3. Otherwise, from the current node’s traversable 4‑neighbors, keep those that **lead to a reachable frontier**; score them by a mix of distance, information gain, and alignment with `world.last_dir`. Track tried directions per junction in `world.tried_exits` and push junctions to `world.branch_stack` for DFS‑style backtracking.  
4. If current node is exhausted, pop the stack until an earlier junction still has untried exits (commit to return there).  
5. Fallback: nearest frontier in the whole known map.

Side effects: sets `world.commit_target`, `world.commit_ttl` and updates `tried_exits` / `branch_stack`.

---

## Local Fallback

### `fallbackMoveWithBrakeAndBias(state, world, rSafe) -> (ax, ay)`

Enumerates all 9 accelerations, keeps those that pass `brakingOk` and `validLineLocal`, and scores them by:
- heading preference (continue forward > sideways > reverse),
- **strong** penalty for stepping to the previous cell,
- edge repetition and node visitation penalties,
- exploration bonus if the landing node borders `UNKNOWN`.

If nothing is valid, tries braking toward zero velocity; otherwise returns `(0,0)`.

---

## Pure‑Pursuit Driver & Speed Control

### `_furthest_visible_on_path(world, start_xy, path, max_jump=40) -> (index, cell)`

Walks forward on the path and finds the farthest node still **line‑of‑sight visible** from `start_xy` on `known_map`. Returns its index and cell.

### `_path_curvature(path, i) -> float in [0,1]`

Estimates curvature (turn sharpness) near index `i` using the angle between consecutive segments.

### `_target_speed_from_context(distance_left, curvature, rSafe) -> float`

Heuristic target speed:
- faster if the remaining path is long and straight,
- slower if curvature is high,
- capped by an **R‑based half‑speed** limit: `0.5 * sqrt(2*rSafe)` (at least 1.0).

### `_score_accel(ax, ay, state, world, rSafe, waypoint, target_speed) -> Optional[(score, (nx,ny))]`

Rejects accelerations that fail `brakingOk` or `validLineLocal` or would land on another player. Otherwise computes a score that blends:
- distance to the waypoint,
- deviation from target speed,
- heading alignment penalty,
- node/edge repetition penalties,
- small penalty for fully stopping,
- exploration bonus near unknowns,
- **very large** penalty if it would step back into `prev_pos` while `no_backtrack_lock` is active.

### `pure_pursuit_move(state, world, coarse, rSafe) -> Optional[(ax, ay)]`

1. Choose/refresh a committed target.  
2. Plan a coarse 2D path to it.  
3. Pick the farthest LOS waypoint and compute curvature + distance.  
4. Convert to a target speed and score the 9 accelerations.  
5. Prefer non‑backtracking moves; apply `no_backtrack_lock` when needed.  
6. Return the best `(ax, ay)` or `None` if nothing is valid.

---

## Decision Loop

### `calculateMove(world, planner, state, coarse) -> (ax, ay)`

1. **Confirm last move**: success → credit the edge; failure → penalize, freeze, mark landing as `WALL`, drop subgoal.  
2. **Update** `known_map` with current `visible_raw`.  
3. If `freeze_ticks_left > 0`: output `0 0` and advance visit stats; skip planning.  
4. Compute `rSafe = max(0, R - 1)` where `R` is `visibility_radius`.  
5. Update heading memory (`last_dir`, `backtrail`).  
6. Try `pure_pursuit_move`; if `None`:
   - refresh subgoal, try **4D A\*** via `planner.plan` and take its first action; if that fails,  
   - apply `fallbackMoveWithBrakeAndBias`.
7. Compute the **tentative** next position from current `(pos, vel)` and chosen `(ax, ay)`.  
8. If that would land on **another player**, override decision to `(0,0)`.  
9. Record `visited_count` of the current cell.  
10. If tentative `next` is in bounds: set `pending_edge` and `predicted_next`. Otherwise, penalize the OOB edge, clear pending state, and output `(0,0)`.  
11. If committed to a subgoal, track *stalled* steps; if we fail to get closer for `_STALL_RESET` ticks, drop the commitment and allow replanning.

Returns the final `(ax, ay)` within `[-1, 1]` per axis.

---

## Program Entry Point

### `main()`

- Prints `READY` for synchronization.
- Reads initial `Circuit`.
- Constructs `WorldModel`, `AStarPlanner(vMax=7, rSafe=R-1)`, and `CoarsePlanner2D`.
- Repeats: read observation → compute `(ax, ay)` → clamp to `{-1,0,1}` → print.

---

## Constants & Tunables

- **WorldModel**
  - `_COMMIT_TTL_DEFAULT = 40` — subgoal commitment freshness
  - `_STALL_RESET = 8` — how many non‑progress ticks before dropping the subgoal
  - `freeze_ticks_left ≈ 5` (set when collisions detected)
- **CoarsePlanner2D**
  - `cost_unknown = 5.0`, `cost_empty = 1.0`, `cost_goal = 1.0`
  - `cost_recent_back = 60.0`, `cost_old_back = 8.0`
- **AStarPlanner**
  - `vMax = 7` (constructor), `R_safe = R‑1`  
  - `turn_pen_back = 6.0`, `turn_pen_half = 2.0`, `turn_pen_ortho = 0.6`  
  - `w_backtrack = 6.0`, `w_edge_repeat = 1.0`, `w_visit = 0.2`
  - `maxNodes = 25_000` (in `main()`)

You can safely tweak these without changing the rest of the architecture.


## Extension Points

- **Mapping:** fuse multiple observations (already done), add decay or probabilistic unknowns.  
- **Coarse planner costs:** adjust backtrail penalties or make `UNKNOWN` adaptive.  
- **Subgoal policy:** integrate goal clustering, or change DFS ordering and info‑gain.  
- **Pure‑pursuit:** learn target speed from data; add curvature‑based braking zone.  
- **4D A\*:** add dynamic obstacles cost from `players`; time‑expanded collision checks.  
- **Fallback:** add wall‑hugging bias when frontiers are scarce.

---
