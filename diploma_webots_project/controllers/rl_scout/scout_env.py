import math
import random


class ScoutEnv:

    def __init__(self, robot, robot_node, motors, lidar,
                 imu, gps, gyro, accelerometer,
                 translation_field, rotation_field,
                 goal_translation_field,
                 timestep):

        self.robot = robot
        self.robot_node = robot_node
        self.motors = motors
        self.lidar = lidar
        self.imu = imu
        self.gps = gps
        self.gyro = gyro
        self.accelerometer = accelerometer
        self.translation_field = translation_field
        self.rotation_field = rotation_field
        self.goal_translation_field = goal_translation_field
        self.timestep = timestep
        self.dt = timestep / 1000.0

        # Parametri robot (Pioneer 3-AT)
        self.R = 0.0975
        self.L = 0.33
        self.MAX_WHEEL_SPEED = 6.4

        # Dinamica
        self.v = 0.0
        self.w = 0.0
        self.v_prev = 0.0
        self.w_prev = 0.0

        self.MAX_ACC_V = 1.5
        self.MAX_ACC_W = 4.0

        # Odometrie internă
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.theta = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.gyro_rate = [0.0, 0.0, 0.0]
        self.acceleration = [0.0, 0.0, 0.0]

        # Goal (setat extern sau random)
        self.goal_x = 0.0
        self.goal_y = 0.0

        self.prev_distance = 0.0
        self.prev_nav_kind = "goal"
        self.step_count = 0
        self.max_steps = 1200
        self.simulation_running = True
        self.last_state = None
        self.done_reason = None

        self.state_dim = 16
        self.action_dim = 5

        # Limita arenei (pereții sunt la ±14m, marginea de siguranță)
        self.arena_limit = 12.0
        self.goal_clearance = 1.5
        self.goal_marker_z = 0.051
        self.goal_tolerance = 0.2
        self.goal_on_traversable = False
        self.ramp_access_reached_radius = 0.85
        self.ramp_commit_distance = 1.05
        self.max_traversable_slope_deg = 35.0
        self.max_safe_roll = 0.75
        self.max_safe_pitch = 0.75
        self.traversable_regions = [
            {
                "type": "ramp",
                "name": "Ramp",
                "x": -4.82,
                "y": -7.39,
                "sx": 5.2,
                "sy": 2.4,
                "height": 0.55,
                "top_fraction": 0.45,
                "slope_deg": 30.0,
                "entry_margin": 0.65,
                "side_margin": 0.45,
            },
        ]
        self.current_map_obstacles = [
            {"type": "rect", "name": "Block", "x": 6.0, "y": 6.0, "sx": 3.0, "sy": 1.5},
            {"type": "rect", "name": "Cube1", "x": 3.0, "y": 0.0, "sx": 0.5, "sy": 0.5},
            {"type": "circle", "name": "Cylinder", "x": -6.0, "y": 8.0, "r": 0.5},
            {"type": "circle", "name": "Cone", "x": 7.0, "y": -7.0, "r": 0.5},
        ]

    # ------------------------------------------------------------------

    def set_goal(self, gx, gy):
        """Setează goal-ul din exterior (Tkinter / tastă G)."""
        requested_x = float(gx)
        requested_y = float(gy)
        safe_x, safe_y, adjusted = self._make_goal_safe(requested_x, requested_y)

        self.goal_x = safe_x
        self.goal_y = safe_y
        self.goal_on_traversable = self._point_in_traversable(self.goal_x, self.goal_y)
        self._move_goal_marker()
        nav_x, nav_y, nav_kind = self.get_navigation_target()
        self.prev_nav_kind = nav_kind
        self.prev_distance = math.sqrt((nav_x - self.x) ** 2 + (nav_y - self.y) ** 2)

        if adjusted:
            print(
                f"[WARN] Goal ({requested_x:.2f}, {requested_y:.2f}) era prea aproape "
                f"de obstacol/perete. Ajustat la ({self.goal_x:.2f}, {self.goal_y:.2f})"
            )
        print(f"[ENV] Goal actualizat → ({self.goal_x:.2f}, {self.goal_y:.2f})")

    def _move_goal_marker(self):
        """Mută conul vizual la pozița goal-ului."""
        if self.goal_translation_field is not None:
            marker_z = self._surface_z_at(self.goal_x, self.goal_y) + self.goal_marker_z
            self.goal_translation_field.setSFVec3f(
                [self.goal_x, self.goal_y, marker_z]
            )

    def _point_in_rect_region(self, region, x, y, margin=0.0):
        half_x = region["sx"] / 2.0 + margin
        half_y = region["sy"] / 2.0 + margin
        return (
            region["x"] - half_x <= x <= region["x"] + half_x and
            region["y"] - half_y <= y <= region["y"] + half_y
        )

    def _point_in_traversable(self, x, y, margin=0.0):
        for region in self.traversable_regions:
            if region["slope_deg"] <= self.max_traversable_slope_deg:
                if self._point_in_rect_region(region, x, y, margin):
                    return True
        return False

    def _region_for_point(self, x, y, margin=0.0):
        for region in self.traversable_regions:
            if region["slope_deg"] <= self.max_traversable_slope_deg:
                if self._point_in_rect_region(region, x, y, margin):
                    return region
        return None

    def _clamp(self, value, low, high):
        return max(low, min(value, high))

    def _ramp_lane_y(self, region, y):
        half_y = region["sy"] / 2.0
        side_margin = region.get("side_margin", 0.35)
        lane_half_y = max(0.1, half_y - side_margin)
        return self._clamp(y, region["y"] - lane_half_y, region["y"] + lane_half_y)

    def _point_in_ramp_lane(self, region, x, y, margin=0.0):
        half_x = region["sx"] / 2.0 + margin
        half_y = region["sy"] / 2.0
        side_margin = region.get("side_margin", 0.35)
        lane_half_y = max(0.1, half_y - side_margin) + margin
        return (
            region["x"] - half_x <= x <= region["x"] + half_x and
            region["y"] - lane_half_y <= y <= region["y"] + lane_half_y
        )

    def _ramp_access_target(self, region):
        half_x = region["sx"] / 2.0
        entry_margin = region.get("entry_margin", 0.55)
        entry_y = self._ramp_lane_y(region, self.goal_y)

        left_entry = (region["x"] - half_x - entry_margin, entry_y)
        right_entry = (region["x"] + half_x + entry_margin, entry_y)

        left_dist = math.sqrt((self.x - left_entry[0]) ** 2 + (self.y - left_entry[1]) ** 2)
        right_dist = math.sqrt((self.x - right_entry[0]) ** 2 + (self.y - right_entry[1]) ** 2)
        return left_entry if left_dist <= right_dist else right_entry

    def _ramp_side_clear_target(self, region, access_x):
        half_y = region["sy"] / 2.0
        entry_margin = region.get("entry_margin", 0.55)
        side = 1.0 if self.y >= region["y"] else -1.0
        return access_x, region["y"] + side * (half_y + entry_margin)

    def get_navigation_target(self):
        """Returneaza targetul activ: goal final sau punct de acces la rampa."""
        region = self._region_for_point(self.goal_x, self.goal_y)
        if region is None or region["type"] != "ramp":
            return self.goal_x, self.goal_y, "goal"

        access_x, access_y = self._ramp_access_target(region)
        side_x, side_y = self._ramp_side_clear_target(region, access_x)
        access_distance = math.sqrt((self.x - access_x) ** 2 + (self.y - access_y) ** 2)
        side_distance = math.sqrt((self.x - side_x) ** 2 + (self.y - side_y) ** 2)
        on_valid_lane = self._point_in_ramp_lane(region, self.x, self.y, margin=0.15)

        if not on_valid_lane and access_distance > self.ramp_access_reached_radius:
            if abs(self.x - access_x) > 0.45 and side_distance > 0.35:
                return side_x, side_y, "ramp_side_clear"
            return access_x, access_y, "ramp_access"

        return self.goal_x, self.goal_y, "goal"

    def get_navigation_error(self):
        nav_x, nav_y, nav_kind = self.get_navigation_target()
        dx = nav_x - self.x
        dy = nav_y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.atan2(dy, dx) - self.theta
        angle = math.atan2(math.sin(angle), math.cos(angle))
        return distance, angle, nav_kind

    def _surface_z_at(self, x, y):
        for region in self.traversable_regions:
            if not self._point_in_rect_region(region, x, y):
                continue

            if region["type"] != "ramp":
                return 0.0

            half_x = region["sx"] / 2.0
            top_half = half_x * region["top_fraction"]
            rel_x = x - region["x"]
            height = region["height"]

            if abs(rel_x) <= top_half:
                return height

            ramp_run = max(half_x - top_half, 0.001)
            if rel_x < -top_half:
                return height * max(0.0, min(1.0, (rel_x + half_x) / ramp_run))
            return height * max(0.0, min(1.0, (half_x - rel_x) / ramp_run))

        return 0.0

    def _path_crosses_traversable(self, samples=12):
        for i in range(1, samples + 1):
            t = i / samples
            px = self.x + (self.goal_x - self.x) * t
            py = self.y + (self.goal_y - self.y) * t
            if self._point_in_traversable(px, py, margin=0.35):
                return True
        return False

    def should_drive_over_traversable(self):
        nav_x, nav_y, nav_kind = self.get_navigation_target()
        if nav_kind != "goal":
            return False

        if abs(self.roll) > self.max_safe_roll or abs(self.pitch) > self.max_safe_pitch:
            return False

        region = self._region_for_point(self.goal_x, self.goal_y)
        if region is None and not self._path_crosses_traversable():
            return False

        if region is not None:
            access_x, access_y = self._ramp_access_target(region)
            near_access = math.sqrt((self.x - access_x) ** 2 + (self.y - access_y) ** 2) < self.ramp_commit_distance
            on_valid_lane = self._point_in_ramp_lane(region, self.x, self.y, margin=0.35)
            if not (near_access or on_valid_lane):
                return False

        dx = nav_x - self.x
        dy = nav_y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance > 8.0:
            return False

        goal_angle = math.atan2(dy, dx) - self.theta
        goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))
        return abs(goal_angle) < 0.65

    def can_attempt_traversable_ahead(self, state):
        front, left, right, front_left, front_right, upper_front, mid_front, \
            height_profile, distance_norm, angle_norm, v_norm, w_norm, \
            z_norm, roll_norm, pitch_norm, yaw_rate_norm = state

        if abs(self.roll) > self.max_safe_roll or abs(self.pitch) > self.max_safe_pitch:
            return False

        nav_distance, nav_angle, _ = self.get_navigation_error()
        if nav_distance > 8.0 or abs(nav_angle) > 0.55:
            return False

        lower_close = front < 0.22 or front_left < 0.18 or front_right < 0.18
        upper_clear = upper_front > front + 0.10 and upper_front > 0.20
        middle_not_blocked = mid_front > front + 0.03
        return lower_close and upper_clear and middle_not_blocked

    def is_on_traversable_terrain(self):
        region = self._region_for_point(self.x, self.y, margin=0.25)
        return region is not None and self.z > 0.12

    def _make_goal_safe(self, gx, gy):
        """Mută goal-ul în afara obstacolelor simple din harta curentă."""
        adjusted = False
        margin = self.goal_clearance

        x = max(min(gx, self.arena_limit - margin), -self.arena_limit + margin)
        y = max(min(gy, self.arena_limit - margin), -self.arena_limit + margin)
        adjusted = adjusted or x != gx or y != gy

        for obstacle in self.current_map_obstacles:
            if obstacle["type"] == "rect":
                half_x = obstacle["sx"] / 2.0 + margin
                half_y = obstacle["sy"] / 2.0 + margin
                min_x = obstacle["x"] - half_x
                max_x = obstacle["x"] + half_x
                min_y = obstacle["y"] - half_y
                max_y = obstacle["y"] + half_y

                if min_x <= x <= max_x and min_y <= y <= max_y:
                    distances = {
                        "left": abs(x - min_x),
                        "right": abs(max_x - x),
                        "bottom": abs(y - min_y),
                        "top": abs(max_y - y),
                    }
                    nearest_side = min(distances, key=distances.get)
                    if nearest_side == "left":
                        x = min_x - 0.05
                    elif nearest_side == "right":
                        x = max_x + 0.05
                    elif nearest_side == "bottom":
                        y = min_y - 0.05
                    else:
                        y = max_y + 0.05
                    adjusted = True

            elif obstacle["type"] == "circle":
                dx = x - obstacle["x"]
                dy = y - obstacle["y"]
                distance = math.sqrt(dx ** 2 + dy ** 2)
                min_distance = obstacle["r"] + margin

                if distance < min_distance:
                    angle = math.atan2(dy, dx) if distance > 0 else 0.0
                    x = obstacle["x"] + math.cos(angle) * (min_distance + 0.05)
                    y = obstacle["y"] + math.sin(angle) * (min_distance + 0.05)
                    adjusted = True

        x = max(min(x, self.arena_limit - margin), -self.arena_limit + margin)
        y = max(min(y, self.arena_limit - margin), -self.arena_limit + margin)
        return x, y, adjusted

    # ------------------------------------------------------------------

    def reset(self, start_x=0.0, start_y=0.0, start_theta=0.0, random_goal=True):
        """
        Resetează episodul.
        - start_x/y/theta: poziția de start a robotului
        - random_goal: True în training, False în eval (goal setat extern)
        """

        for m in self.motors:
            m.setVelocity(0.0)

        self.translation_field.setSFVec3f([start_x, start_y, 0.05])

        # Conversie theta -> rotatie Webots in planul hartii (axa Z este verticala).
        self.rotation_field.setSFRotation([0, 0, 1, start_theta])

        self.robot.simulationResetPhysics()
        self.simulation_running = True
        if self.robot.step(self.timestep) == -1:
            self.simulation_running = False

        self.x = start_x
        self.y = start_y
        self.z = 0.05
        self.theta = start_theta
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = start_theta
        self.gyro_rate = [0.0, 0.0, 0.0]
        self.acceleration = [0.0, 0.0, 0.0]

        self.v = 0.0
        self.w = 0.0
        self.v_prev = 0.0
        self.w_prev = 0.0
        self.step_count = 0
        self.done_reason = None

        if random_goal:
            # Goal random, garantat diferit de start și în arenă
            while True:
                gx = random.uniform(-self.arena_limit, self.arena_limit)
                gy = random.uniform(-self.arena_limit, self.arena_limit)
                gx, gy, adjusted = self._make_goal_safe(gx, gy)
                dist = math.sqrt((gx - start_x) ** 2 + (gy - start_y) ** 2)
                if dist > 2.0 and not adjusted:  # minim 2m distanță față de start
                    break
            self.goal_x = gx
            self.goal_y = gy
            self.goal_on_traversable = self._point_in_traversable(self.goal_x, self.goal_y)
            self._move_goal_marker()

        nav_x, nav_y, nav_kind = self.get_navigation_target()
        self.prev_nav_kind = nav_kind
        self.prev_distance = math.sqrt((nav_x - self.x) ** 2 + (nav_y - self.y) ** 2)

        self.last_state = self._get_state()
        return self.last_state

    # ------------------------------------------------------------------

    def step(self, action):

        if not self.simulation_running:
            self.done_reason = "simulation_stopped"
            return self.last_state, 0.0, True

        self.step_count += 1
        self._apply_action(action)

        if self.robot.step(self.timestep) == -1:
            self.simulation_running = False
            self.done_reason = "simulation_stopped"
            return self.last_state, 0.0, True

        self._update_odometry()

        state = self._get_state()
        self.last_state = state
        self.done_reason = None
        reward, done = self._compute_reward(state)

        if self.step_count >= self.max_steps:
            done = True
            self.done_reason = "timeout"

        return state, reward, done

    # ------------------------------------------------------------------

    def _apply_action(self, action):
        # 0 - forward
        # 1 - forward-left
        # 2 - forward-right
        # 3 - rotate left
        # 4 - rotate right

        if action == 0:
            v_target, w_target = 0.55,  0.0
        elif action == 1:
            v_target, w_target = 0.45,  1.0
        elif action == 2:
            v_target, w_target = 0.45, -1.0
        elif action == 3:
            v_target, w_target = 0.0,  1.6
        elif action == 4:
            v_target, w_target = 0.0, -1.6
        else:
            v_target, w_target = 0.0,  0.0

        dv = v_target - self.v_prev
        dv = max(min(dv, self.MAX_ACC_V * self.dt), -self.MAX_ACC_V * self.dt)
        self.v = self.v_prev + dv

        dw = w_target - self.w_prev
        dw = max(min(dw, self.MAX_ACC_W * self.dt), -self.MAX_ACC_W * self.dt)
        self.w = self.w_prev + dw

        self.v_prev = self.v
        self.w_prev = self.w

        v_left  = (self.v - (self.L / 2.0) * self.w) / self.R
        v_right = (self.v + (self.L / 2.0) * self.w) / self.R

        v_left  = max(min(v_left,  self.MAX_WHEEL_SPEED), -self.MAX_WHEEL_SPEED)
        v_right = max(min(v_right, self.MAX_WHEEL_SPEED), -self.MAX_WHEEL_SPEED)

        # [FL, BL, FR, BR]
        self.motors[0].setVelocity(v_left)
        self.motors[1].setVelocity(v_left)
        self.motors[2].setVelocity(v_right)
        self.motors[3].setVelocity(v_right)

    # ------------------------------------------------------------------

    def _update_odometry(self):
        # In simulator folosim GPS daca exista; Supervisor ramane fallback pentru testare.
        pos = self.gps.getValues() if self.gps is not None else self.robot_node.getPosition()
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

        if self.imu is not None:
            self.roll, self.pitch, self.yaw = self.imu.getRollPitchYaw()
            self.theta = self.yaw
        else:
            # Orientare reala: axa locala +X a robotului proiectata in planul XY.
            orientation = self.robot_node.getOrientation()
            self.theta = math.atan2(orientation[3], orientation[0])

        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        if self.gyro is not None:
            self.gyro_rate = self.gyro.getValues()
        if self.accelerometer is not None:
            self.acceleration = self.accelerometer.getValues()

    # ------------------------------------------------------------------

    def _safe_range(self, ranges, idx, max_range):
        if not ranges:
            return max_range
        idx = max(0, min(idx, len(ranges) - 1))
        value = ranges[idx]
        if math.isinf(value) or math.isnan(value):
            return max_range
        return min(value, max_range)

    def _lidar_value(self, ranges, layer, horizontal_index, max_range):
        try:
            horizontal_resolution = self.lidar.getHorizontalResolution()
            layers = self.lidar.getNumberOfLayers()
        except AttributeError:
            horizontal_resolution = len(ranges)
            layers = 1

        if layers <= 1 or horizontal_resolution <= 0:
            return self._safe_range(ranges, horizontal_index, max_range)

        layer = max(0, min(layer, layers - 1))
        horizontal_index = max(0, min(horizontal_index, horizontal_resolution - 1))
        idx = layer * horizontal_resolution + horizontal_index
        return self._safe_range(ranges, idx, max_range)

    def _get_lidar_features(self):
        ranges = self.lidar.getRangeImage()
        max_range = 10.0

        try:
            horizontal_resolution = self.lidar.getHorizontalResolution()
            layers = self.lidar.getNumberOfLayers()
        except AttributeError:
            horizontal_resolution = len(ranges)
            layers = 1

        n = max(1, horizontal_resolution)
        lower_layer = max(0, layers - 1)
        upper_layer = 0
        mid_layer = max(0, layers // 2)

        front_raw = self._lidar_value(ranges, lower_layer, n // 2, max_range)
        left_raw = self._lidar_value(ranges, lower_layer, 3 * n // 4, max_range)
        right_raw = self._lidar_value(ranges, lower_layer, n // 4, max_range)
        front_left_raw = self._lidar_value(ranges, lower_layer, 5 * n // 8, max_range)
        front_right_raw = self._lidar_value(ranges, lower_layer, 3 * n // 8, max_range)
        upper_front_raw = self._lidar_value(ranges, upper_layer, n // 2, max_range)
        mid_front_raw = self._lidar_value(ranges, mid_layer, n // 2, max_range)

        front       = front_raw / max_range
        left        = left_raw / max_range
        right       = right_raw / max_range
        front_left  = front_left_raw / max_range
        front_right = front_right_raw / max_range
        upper_front = upper_front_raw / max_range
        mid_front = mid_front_raw / max_range
        height_profile = max(-1.0, min(1.0, (upper_front_raw - front_raw) / max_range))

        return front, left, right, front_left, front_right, upper_front, mid_front, height_profile

    def _get_state(self):
        front, left, right, front_left, front_right, upper_front, mid_front, height_profile = \
            self._get_lidar_features()

        nav_x, nav_y, _ = self.get_navigation_target()
        dx = nav_x - self.x
        dy = nav_y - self.y

        distance = math.sqrt(dx ** 2 + dy ** 2)
        angle    = math.atan2(dy, dx) - self.theta
        angle    = math.atan2(math.sin(angle), math.cos(angle))

        distance_norm = min(distance, 40.0) / 40.0
        angle_norm    = angle / math.pi
        v_norm        = self.v / 2.0
        w_norm        = self.w / 4.0
        z_norm        = max(0.0, min(self.z, 2.0)) / 2.0
        roll_norm     = max(-1.0, min(1.0, self.roll / math.pi))
        pitch_norm    = max(-1.0, min(1.0, self.pitch / math.pi))
        yaw_rate_norm = max(-1.0, min(1.0, self.gyro_rate[2] / 5.0))

        return [
            front, left, right, front_left, front_right, upper_front, mid_front, height_profile,
            distance_norm, angle_norm,
            v_norm, w_norm,
            z_norm, roll_norm, pitch_norm, yaw_rate_norm
        ]

    # ------------------------------------------------------------------

    def _compute_reward(self, state):
        front, left, right, front_left, front_right, upper_front, mid_front, \
            height_profile, distance_norm, angle_norm, v_norm, w_norm, \
            z_norm, roll_norm, pitch_norm, yaw_rate_norm = state

        reward = 0.0

        final_distance = math.sqrt(
            (self.goal_x - self.x) ** 2 +
            (self.goal_y - self.y) ** 2
        )

        # Goal atins doar cand centrul robotului ajunge in centrul discului.
        if final_distance <= self.goal_tolerance:
            self.done_reason = "goal"
            return 200.0, True

        # 1. Progress spre goal
        nav_x, nav_y, nav_kind = self.get_navigation_target()
        nav_distance = math.sqrt((nav_x - self.x) ** 2 + (nav_y - self.y) ** 2)
        if nav_kind != self.prev_nav_kind:
            self.prev_distance = nav_distance
            self.prev_nav_kind = nav_kind

        progress = self.prev_distance - nav_distance
        reward += progress * 20.0

        # 2. Orientare spre goal
        if abs(angle_norm) < 0.15:
            reward += 1.0
        reward -= abs(angle_norm) * 0.02

        # 3. Bonus mișcare în spațiu liber
        min_dist = min(front, left, right, front_left, front_right)
        if min_dist > 0.15:
            reward += 0.2

        # 4. Penalizări apropriere obstacol
        if front < 0.10:
            reward -= 1.0
        if front < 0.05:
            reward -= 2.0
        if (left < 0.05 or right < 0.05 or
                front_left < 0.05 or front_right < 0.05):
            reward -= 0.1

        # 5. Living cost + penalizare rotație excesivă
        reward -= 0.0005
        reward -= abs(self.w) * 0.002

        # 6. Coliziune
        if min_dist < 0.025:
            if (self.should_drive_over_traversable() or
                    self.can_attempt_traversable_ahead(state) or
                    self.is_on_traversable_terrain()):
                reward -= 1.0
            else:
                self.done_reason = "collision"
                return -20.0, True

        self.prev_distance = nav_distance
        return reward, False
