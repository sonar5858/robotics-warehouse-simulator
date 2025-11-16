import pygame
import heapq

# ---------------- CONFIG ----------------
GRID_WIDTH, GRID_HEIGHT = 20, 15
CELL_SIZE = 40
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 10

# Obstacles: list of (x, y)
OBSTACLES = {(5, y) for y in range(3, 12)}  # vertical wall

START = (2, 2)
GOAL = (17, 10)

# ------------- A* PATHFINDING -----------
def neighbors(node):
    x, y = node
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and (nx, ny) not in OBSTACLES:
            return (nx, ny)
        # no else: just skip invalid/obstacle
    return None

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        x, y = current
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                continue
            if neighbor in OBSTACLES:
                continue

            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
    return []

# ------------- PYGAME SIMULATION --------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Warehouse Robot - A* Demo")

path = astar(START, GOAL)
path_index = 0
robot_pos = START

running = True
while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move robot along path
    if path and path_index < len(path):
        robot_pos = path[path_index]
        path_index += 1

    # Draw background
    screen.fill((30, 30, 30))

    # Draw grid
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = (50, 50, 50)
            if (x, y) in OBSTACLES:
                color = (80, 20, 20)
            pygame.draw.rect(screen, color, rect, 0)
            pygame.draw.rect(screen, (40, 40, 40), rect, 1)

    # Draw start & goal
    sx, sy = START
    gx, gy = GOAL
    pygame.draw.rect(screen, (0, 100, 0),
                     (sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, (100, 100, 0),
                     (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw robot
    rx, ry = robot_pos
    cx = rx * CELL_SIZE + CELL_SIZE // 2
    cy = ry * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, (0, 150, 255), (cx, cy), CELL_SIZE // 3)

    pygame.display.flip()

pygame.quit()
