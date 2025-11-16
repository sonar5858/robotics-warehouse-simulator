import pygame
import heapq
import random

# ---------------- CONFIG ----------------
GRID_WIDTH, GRID_HEIGHT = 20, 15
CELL_SIZE = 40

PANEL_WIDTH = 260  # right-side dashboard
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE + PANEL_WIDTH
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

FPS = 10

HOME_POSITIONS = [(0, 0), (0, 1)]        # robot homes
CONVEYOR_Y = GRID_HEIGHT - 1             # bottom row = conveyor

# Shelves / obstacles (avoid bottom conveyor row)
OBSTACLES = (
    {(5, y) for y in range(2, 13)} |
    {(10, y) for y in range(1, 10)}
)

TASK_QUEUE = []  # each task: {"pick": (x,y), "drop": (x,y)}

# ------------- A* PATHFINDING -----------
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, dynamic_blockers=None, allow_goal_on_conveyor=False):
    """
    A* with:
      - static obstacles
      - optional dynamic blockers (other robots)
      - robots are NOT allowed to drive along conveyor
        except stepping onto the goal conveyor cell when dropping.
    """
    if dynamic_blockers is None:
        dynamic_blockers = set()

    blocked = set(OBSTACLES) | set(dynamic_blockers)

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)

            if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                continue

            # Conveyor rule: cannot drive ALONG conveyor
            if ny == CONVEYOR_Y:
                # Only allowed if this is exactly the goal and explicitly allowed
                if not (allow_goal_on_conveyor and neighbor == goal):
                    continue

            if neighbor in blocked:
                continue

            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
    return []

# ------------- HELPERS ------------------
def random_pick_cell():
    """Random free cell for product (not conveyor and not obstacle)."""
    while True:
        x = random.randint(0, GRID_WIDTH - 1)
        y = random.randint(0, GRID_HEIGHT - 2)   # exclude conveyor row
        if (x, y) not in OBSTACLES:
            return (x, y)

def closest_conveyor_cell_to(pick):
    """Choose conveyor cell with minimum Manhattan distance to pick."""
    px, py = pick
    best_cell = None
    best_dist = float("inf")
    for x in range(GRID_WIDTH):
        candidate = (x, CONVEYOR_Y)
        if candidate in OBSTACLES:
            continue
        dist = abs(px - x) + abs(py - CONVEYOR_Y)
        if dist < best_dist:
            best_dist = dist
            best_cell = candidate
    return best_cell

# -------- ROBOT CLASS --------
class Robot:
    def __init__(self, robot_id, home_pos, color):
        self.id = robot_id
        self.home = home_pos
        self.color = color
        self.reset()

    def reset(self):
        self.pos = self.home
        self.mode = "IDLE"         # IDLE / TO_PICK / TO_DROP / RETURN_HOME
        self.task = None
        self.full_path = []
        self.tasks_completed = 0
        self._planned_next = None

    def assign_task(self, task):
        self.task = task
        # If already at pick, skip to drop
        if self.pos == task["pick"]:
            self.mode = "TO_DROP"
        else:
            self.mode = "TO_PICK"
        print(f"[Robot {self.id}] Assigned task: {task}")

    def current_target(self):
        if self.mode == "TO_PICK" and self.task:
            return self.task["pick"]
        if self.mode == "TO_DROP" and self.task:
            return self.task["drop"]
        if self.mode == "RETURN_HOME":
            return self.home
        return None

    def plan_next(self, other_positions):
        """
        Plan the next cell (does not move yet).
        """
        self._planned_next = None  # default: stay

        if self.mode == "IDLE":
            self.full_path = []
            return None

        target = self.current_target()
        if target is None:
            self.mode = "IDLE"
            self.full_path = []
            return None

        # If already at target, no movement; arrival handled later
        if self.pos == target:
            self.full_path = []
            return None

        allow_conveyor_goal = (self.mode == "TO_DROP")
        dynamic_blockers = set(other_positions) - {self.pos}

        path = astar(self.pos, target,
                     dynamic_blockers=dynamic_blockers,
                     allow_goal_on_conveyor=allow_conveyor_goal)

        if not path or len(path) < 2:
            self.full_path = path or []
            return None

        self.full_path = path
        next_pos = path[1]
        self._planned_next = next_pos
        return next_pos

    def apply_move_and_handle_arrival(self):
        """
        After conflict resolution, actually move (if allowed)
        and update mode when reaching targets.
        """
        if self._planned_next is not None:
            self.pos = self._planned_next

        target = self.current_target()
        if target is None:
            if self.mode != "IDLE":
                self.mode = "IDLE"
            return

        # Arrival handling
        if self.pos == target:
            if self.mode == "TO_PICK":
                print(f"[Robot {self.id}] Picked item at {self.pos}")
                self.mode = "TO_DROP"
            elif self.mode == "TO_DROP":
                print(f"[Robot {self.id}] Dropped item on conveyor at {self.pos}")
                self.tasks_completed += 1
                self.task = None
                # Go home next
                if self.pos == self.home:
                    self.mode = "IDLE"
                else:
                    self.mode = "RETURN_HOME"
            elif self.mode == "RETURN_HOME":
                print(f"[Robot {self.id}] Returned home at {self.pos}")
                self.mode = "IDLE"

        # Clear planned step for next cycle
        self._planned_next = None

# ------------- PYGAME SETUP -------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Multi-Robot Warehouse Simulator")

font_small = pygame.font.SysFont("consolas", 16)
font_title = pygame.font.SysFont("consolas", 20, bold=True)

panel_x = GRID_WIDTH * CELL_SIZE
RESET_BUTTON_RECT = pygame.Rect(panel_x + 10, 40, 100, 30)
ADD_BUTTON_RECT   = pygame.Rect(panel_x + 130, 40, 110, 30)

robots = [
    Robot(1, HOME_POSITIONS[0], (0, 170, 255)),  # light blue
    Robot(2, HOME_POSITIONS[1], (0, 90, 220)),   # darker blue
]

conveyor_phase = 0  # for animation

def reset_simulation():
    global TASK_QUEUE, conveyor_phase
    TASK_QUEUE = []
    for r in robots:
        r.reset()
    conveyor_phase = 0
    print("[SIM] Reset simulation.")

def add_task():
    """One click = one order."""
    pick = random_pick_cell()
    drop = closest_conveyor_cell_to(pick)
    task = {"pick": pick, "drop": drop}
    TASK_QUEUE.append(task)
    print(f"[SIM] New task added: {task}")

def assign_tasks_if_any():
    """
    Assign each pending task to the nearest idle robot.
    One robot per task; if no idle robot, task stays queued.
    """
    global TASK_QUEUE
    remaining_tasks = []
    for task in TASK_QUEUE:
        idle_robots = [r for r in robots if r.mode == "IDLE"]
        if not idle_robots:
            remaining_tasks.append(task)
            continue

        pick = task["pick"]
        best_robot = min(
            idle_robots,
            key=lambda r: abs(r.pos[0] - pick[0]) + abs(r.pos[1] - pick[1])
        )
        best_robot.assign_task(task)
    TASK_QUEUE = remaining_tasks

# ------------- MAIN LOOP ----------------
running = True
while running:
    clock.tick(FPS)
    conveyor_phase = (conveyor_phase + 4) % CELL_SIZE  # for moving stripes

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if RESET_BUTTON_RECT.collidepoint(mx, my):
                reset_simulation()
            elif ADD_BUTTON_RECT.collidepoint(mx, my):
                add_task()

    # Assign tasks as robots become idle
    assign_tasks_if_any()

    # --------- PLAN MOVES (PHASE 1) ---------
    all_positions = [r.pos for r in robots]
    planned_next = [None] * len(robots)
    for i, robot in enumerate(robots):
        others = all_positions[:i] + all_positions[i+1:]
        planned_next[i] = robot.plan_next(others)

    # --------- RESOLVE COLLISIONS ----------
    # Rule: if two robots want same cell or want to swap,
    #       lower ID wins, other waits.
    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):
            ni, nj = planned_next[i], planned_next[j]
            if ni is None and nj is None:
                continue

            # Same target cell
            if ni is not None and nj is not None and ni == nj:
                if robots[i].id < robots[j].id:
                    planned_next[j] = None
                    robots[j]._planned_next = None
                else:
                    planned_next[i] = None
                    robots[i]._planned_next = None

            # Swapping cells (i -> j.pos and j -> i.pos)
            if (ni is not None and nj is not None and
                ni == robots[j].pos and nj == robots[i].pos):
                if robots[i].id < robots[j].id:
                    planned_next[j] = None
                    robots[j]._planned_next = None
                else:
                    planned_next[i] = None
                    robots[i]._planned_next = None

    # --------- APPLY MOVES + ARRIVAL (PHASE 2) ---------
    for robot in robots:
        robot.apply_move_and_handle_arrival()

    # ---------- DRAW GRID AREA ----------
    screen.fill((25, 25, 25))

    # grid + obstacles + conveyor
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if y == CONVEYOR_Y:
                # Conveyor background
                color = (60, 60, 20)
            else:
                color = (50, 50, 50)

            if (x, y) in OBSTACLES:
                color = (80, 20, 20)

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (40, 40, 40), rect, 1)

    # Conveyor animation (moving stripes left -> right)
    for x in range(GRID_WIDTH):
        if (x, CONVEYOR_Y) in OBSTACLES:
            continue
        base_x = x * CELL_SIZE
        base_y = CONVEYOR_Y * CELL_SIZE
        # moving vertical bar
        stripe_x = base_x + conveyor_phase
        pygame.draw.rect(screen, (200, 200, 80),
                         (stripe_x % (base_x + CELL_SIZE), base_y + 10, 6, CELL_SIZE - 20))

    # Draw tasks: pick = red, drop = green
    for task in TASK_QUEUE:
        px, py = task["pick"]
        dx, dy = task["drop"]
        pick_rect = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        drop_rect = pygame.Rect(dx * CELL_SIZE, dy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (255, 0, 0), pick_rect, 3)   # red pick
        pygame.draw.rect(screen, (0, 255, 0), drop_rect, 3)   # green drop

    # Also show current task of each robot
    for robot in robots:
        if robot.task:
            px, py = robot.task["pick"]
            dx, dy = robot.task["drop"]
            pick_rect = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            drop_rect = pygame.Rect(dx * CELL_SIZE, dy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (255, 0, 0), pick_rect, 3)
            pygame.draw.rect(screen, (0, 255, 0), drop_rect, 3)

    # ---------- PATH VISUALIZATION ----------
    for robot in robots:
        if robot.full_path and len(robot.full_path) > 1:
            pts = []
            for (gx, gy) in robot.full_path:
                cx = gx * CELL_SIZE + CELL_SIZE // 2
                cy = gy * CELL_SIZE + CELL_SIZE // 2
                pts.append((cx, cy))
            pygame.draw.lines(screen, robot.color, False, pts, 2)

    # ---------- DRAW ROBOTS ----------
    for robot in robots:
        rx, ry = robot.pos
        cx = rx * CELL_SIZE + CELL_SIZE // 2
        cy = ry * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, robot.color, (cx, cy), CELL_SIZE // 3)

    # ---------- DASHBOARD PANEL ----------
    panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, SCREEN_HEIGHT)
    pygame.draw.rect(screen, (15, 15, 15), panel_rect)
    pygame.draw.line(screen, (80, 80, 80), (panel_x, 0), (panel_x, SCREEN_HEIGHT), 2)

    title_surface = font_title.render("WAREHOUSE  DASHBOARD", True, (230, 230, 230))
    screen.blit(title_surface, (panel_x + 10, 10))

    # Buttons
    pygame.draw.rect(screen, (60, 60, 60), RESET_BUTTON_RECT, border_radius=5)
    pygame.draw.rect(screen, (60, 60, 60), ADD_BUTTON_RECT, border_radius=5)
    reset_text = font_small.render("RESET", True, (255, 255, 255))
    add_text = font_small.render("ADD TASK", True, (255, 255, 255))
    screen.blit(reset_text, (RESET_BUTTON_RECT.x + 15, RESET_BUTTON_RECT.y + 6))
    screen.blit(add_text, (ADD_BUTTON_RECT.x + 10, ADD_BUTTON_RECT.y + 6))

    # Robot info
    y_offset = 90
    for robot in robots:
        header = font_small.render(f"Robot {robot.id}", True, robot.color)
        screen.blit(header, (panel_x + 10, y_offset))

        mode_text = font_small.render(f"Mode  : {robot.mode}", True, (220, 220, 220))
        screen.blit(mode_text, (panel_x + 20, y_offset + 18))

        pos_text = font_small.render(f"Pos   : {robot.pos}", True, (200, 200, 200))
        screen.blit(pos_text, (panel_x + 20, y_offset + 36))

        if robot.task:
            pick = robot.task["pick"]
            drop = robot.task["drop"]
            t1 = font_small.render(f"PICK : {pick}", True, (180, 180, 180))
            t2 = font_small.render(f"DROP : {drop}", True, (180, 180, 180))
            screen.blit(t1, (panel_x + 20, y_offset + 54))
            screen.blit(t2, (panel_x + 20, y_offset + 72))
        else:
            t = font_small.render("Task : None", True, (160, 160, 160))
            screen.blit(t, (panel_x + 20, y_offset + 54))

        done = font_small.render(f"Done : {robot.tasks_completed}", True, (160, 220, 160))
        screen.blit(done, (panel_x + 20, y_offset + 90))

        y_offset += 130

    queued = font_small.render(f"Queued tasks: {len(TASK_QUEUE)}", True, (220, 220, 220))
    screen.blit(queued, (panel_x + 10, SCREEN_HEIGHT - 40))

    pygame.display.flip()

pygame.quit()
