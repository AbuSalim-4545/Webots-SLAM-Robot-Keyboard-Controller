# Live 2D mapping (occupancy grid) with MANUAL KEYBOARD CONTROL
#
# This version allows you to drive the robot manually while it builds the map.
#
# INSTRUCTIONS:
# 1. Click on the Webots 3D simulation window to give it focus.
# 2. Use the keyboard to control the robot:
#    - W = Forward
#    - S = Backward
#    - A = Turn Left
#    - D = Turn Right
#    - Q = Emergency Stop
#
# The map will be built in real-time based on your movements.

from controller import Robot, Keyboard
import math
import sys

# Try to import matplotlib for live figure
USE_MATPLOTLIB = True
try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:
    USE_MATPLOTLIB = False

# --- Simulation & Robot Parameters ---
TIME_STEP = 64
BASE_SPEED = 6          # Speed when moving forward/backward
TURN_SPEED = 6          # Speed when turning

# --- Mapping Parameters (Unchanged) ---
RESOLUTION = 0.05
MAP_SIZE_CELLS = 200
LOG_ODDS_OCC = 0.9
LOG_ODDS_FREE = -0.1
LOG_ODDS_MIN = -4.0
LOG_ODDS_MAX = 4.0
MAP_DRAW_INTERVAL = 2

# --- Robot Initialization ---
robot = Robot()

# Devices
lidar = robot.getDevice("LDS-01")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

right_motor = robot.getDevice("right wheel motor")
left_motor = robot.getDevice("left wheel motor")
right_motor.setPosition(float('inf'))
left_motor.setPosition(float('inf'))
right_motor.setVelocity(0.0)
left_motor.setVelocity(0.0)

gps = robot.getDevice("gps")
gps.enable(TIME_STEP)
imu = robot.getDevice("inertial unit")
imu.enable(TIME_STEP)

# --- NEW: Enable Keyboard Control ---
keyboard = robot.getKeyboard()
keyboard.enable(TIME_STEP)

# Lidar parameters
lidar_width = lidar.getHorizontalResolution()
lidar_max_range = lidar.getMaxRange()
lidar_fov = lidar.getFov()

# --- Mapping Data and Functions (Unchanged) ---
grid = [0.0] * (MAP_SIZE_CELLS * MAP_SIZE_CELLS)
start_x, start_y = None, None
step_count = 0

def world_to_map(wx, wy):
    if start_x is None: return None, None
    mx = int((wx - start_x) / RESOLUTION + MAP_SIZE_CELLS / 2)
    my = int((wy - start_y) / RESOLUTION + MAP_SIZE_CELLS / 2)
    return mx, my

def inside_map(mx, my):
    return 0 <= mx < MAP_SIZE_CELLS and 0 <= my < MAP_SIZE_CELLS

def bresenham(x0, y0, x1, y1):
    pts = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1: break
        e2 = 2 * err
        if e2 >= dy: err += dy; x += sx
        if e2 <= dx: err += dx; y += sy
    return pts

def update_map(robot_x, robot_y, robot_yaw, ranges):
    rx, ry = world_to_map(robot_x, robot_y)
    if not inside_map(rx, ry): return

    for i in range(lidar_width):
        r = ranges[i]
        if math.isinf(r) or math.isnan(r): continue
        
        is_max_range = r >= lidar_max_range * 0.995
        if is_max_range: r = lidar_max_range

        relative_beam_angle = (i / (lidar_width - 1)) * lidar_fov - lidar_fov / 2.0
        beam_angle = -relative_beam_angle + robot_yaw

        end_x = robot_x + r * math.cos(beam_angle)
        end_y = robot_y + r * math.sin(beam_angle)
        ex, ey = world_to_map(end_x, end_y)

        cells = bresenham(rx, ry, ex, ey)
        if len(cells) > 1:
            for (cx, cy) in cells[:-1]:
                if inside_map(cx, cy):
                    idx = cy * MAP_SIZE_CELLS + cx
                    grid[idx] = max(LOG_ODDS_MIN, grid[idx] + LOG_ODDS_FREE)
        
        if inside_map(ex, ey) and not is_max_range:
            idx_end = ey * MAP_SIZE_CELLS + ex
            grid[idx_end] = min(LOG_ODDS_MAX, grid[idx_end] + LOG_ODDS_OCC)

# --- Visualization (matplotlib) ---
if USE_MATPLOTLIB:
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    data = np.full((MAP_SIZE_CELLS, MAP_SIZE_CELLS), 0.5, dtype=float)
    im = ax.imshow(data, cmap='gray_r', origin='lower', vmin=0, vmax=1)
    robot_plot, = ax.plot([], [], 'ro', markersize=4)
    ax.set_title("Live Occupancy Grid (Keyboard Control)")
    plt.tight_layout()

def logodds_to_prob_array():
    lo = np.array(grid).reshape((MAP_SIZE_CELLS, MAP_SIZE_CELLS))
    return 1.0 - 1.0 / (1.0 + np.exp(lo))

def draw_map(robot_x, robot_y):
    if not USE_MATPLOTLIB: return
    prob = logodds_to_prob_array()
    im.set_data(prob)
    rx, ry = world_to_map(robot_x, robot_y)
    if rx is not None:
        robot_plot.set_data([rx], [ry])
    plt.pause(0.001)

# --- Main Control Loop ---
while robot.step(TIME_STEP) != -1:
    step_count += 1

    if start_x is None:
        gps_vals = gps.getValues()
        start_x, start_y = gps_vals[0], gps_vals[1]

    # Get sensor data for mapping
    gps_vals = gps.getValues()
    robot_x, robot_y = gps_vals[0], gps_vals[1]
    yaw = imu.getRollPitchYaw()[2]
    lidar_values = lidar.getRangeImage()

    # --- NEW: KEYBOARD CONTROL LOGIC ---
    key = keyboard.getKey()
    
    left_speed = 0.0
    right_speed = 0.0

    if key == ord('W'):
        left_speed = BASE_SPEED
        right_speed = BASE_SPEED
    elif key == ord('S'):
        left_speed = -BASE_SPEED
        right_speed = -BASE_SPEED
    elif key == ord('A'):
        left_speed = -TURN_SPEED
        right_speed = TURN_SPEED
    elif key == ord('D'):
        left_speed = TURN_SPEED
        right_speed = -TURN_SPEED
    elif key == ord('Q'):
        left_speed = 0
        right_speed = 0
        
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    # --- MAPPING & VISUALIZATION (Unchanged) ---
    update_map(robot_x, robot_y, yaw, lidar_values)

    if step_count % MAP_DRAW_INTERVAL == 0:
        draw_map(robot_x, robot_y)

# Cleanup
if USE_MATPLOTLIB:
    plt.ioff()
    print("Simulation finished. Close the plot window to exit.")
    plt.show()