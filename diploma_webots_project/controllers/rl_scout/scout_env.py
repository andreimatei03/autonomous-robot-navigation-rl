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
        self.max_steps = 1800
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
        self.path_clearance = 0.85
        self.corner_clearance = 0.35
        self.avoidance_reached_radius = 0.45
        self.current_avoidance_target = None
        self.completed_avoidance_targets = []
        self.underpass_reached_radius = 0.55
        self.current_underpass_stage = 0
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
        self.current_avoidance_target = None
        self.completed_avoidance_targets = []
        self.current_underpass_stage = 0
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

    def _underpass_target(self):
        """Planifica trecerea pe sub rampa cand goal-ul este pe sol, dincolo de ea."""
        if self.goal_on_traversable:
            return None

        for region in self.traversable_regions:
            if region["type"] != "ramp":
                continue
            if not self._path_crosses_traversable():
                continue

            half_y = region["sy"] / 2.0
            entry_margin = region.get("entry_margin", 0.55) + 0.25
            goal_side = 1.0 if self.goal_y >= region["y"] else -1.0
            robot_side = 1.0 if self.y >= region["y"] else -1.0

            if goal_side == robot_side and not self._point_in_rect_region(region, self.x, self.y, margin=0.35):
                continue

            entry_side = -goal_side
            entry = (region["x"], region["y"] + entry_side * (half_y + entry_margin))
            exit_target = (region["x"], region["y"] + goal_side * (half_y + entry_margin))
            aligned_with_underpass = abs(self.x - region["x"]) <= 0.9

            if self.current_underpass_stage == 0:
                entry_reached = self._target_distance(entry) <= self.underpass_reached_radius
                crossed_entry_side = aligned_with_underpass and (self.y - region["y"]) * entry_side < half_y
                if entry_reached or crossed_entry_side:
                    self.current_underpass_stage = 1
                else:
                    return entry[0], entry[1], "underpass_entry"

            if self.current_underpass_stage == 1:
                exit_reached = self._target_distance(exit_target) <= self.underpass_reached_radius
                crossed_exit_side = aligned_with_underpass and (self.y - region["y"]) * goal_side > half_y
                if exit_reached or crossed_exit_side:
                    self.current_underpass_stage = 2
                    return None
                return exit_target[0], exit_target[1], "underpass_exit"

        return None

    def get_navigation_target(self):
        """Returneaza targetul activ: goal final, punct de acces sau waypoint de ocolire."""
        region = self._region_for_point(self.goal_x, self.goal_y)
        if region is None or region["type"] != "ramp":
            underpass_target = self._underpass_target()
            if underpass_target is not None:
                return underpass_target

            avoid_target = self._obstacle_avoidance_target()
            if avoid_target is not None:
                return avoid_target[0], avoid_target[1], "obstacle_avoid"
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

    def _target_distance(self, target):
        return math.sqrt((self.x - target[0]) ** 2 + (self.y - target[1]) ** 2)

    def _rect_bounds(self, obstacle, margin=0.0):
        half_x = obstacle["sx"] / 2.0 + margin
        half_y = obstacle["sy"] / 2.0 + margin
        return (
            obstacle["x"] - half_x,
            obstacle["x"] + half_x,
            obstacle["y"] - half_y,
            obstacle["y"] + half_y,
        )

    def _point_in_obstacle(self, obstacle, x, y, margin=0.0):
        if obstacle["type"] == "rect":
            min_x, max_x, min_y, max_y = self._rect_bounds(obstacle, margin)
            return min_x <= x <= max_x and min_y <= y <= max_y

        if obstacle["type"] == "circle":
            return math.sqrt((x - obstacle["x"]) ** 2 + (y - obstacle["y"]) ** 2) <= obstacle["r"] + margin

        return False

    def _segment_intersects_rect(self, x1, y1, x2, y2, obstacle, margin=0.0):
        min_x, max_x, min_y, max_y = self._rect_bounds(obstacle, margin)
        dx = x2 - x1
        dy = y2 - y1
        t_min = 0.0
        t_max = 1.0

        for p, q in ((-dx, x1 - min_x), (dx, max_x - x1), (-dy, y1 - min_y), (dy, max_y - y1)):
            if abs(p) < 1e-9:
                if q < 0.0:
                    return False
                continue

            t = q / p
            if p < 0.0:
                if t > t_max:
                    return False
                t_min = max(t_min, t)
            else:
                if t < t_min:
                    return False
                t_max = min(t_max, t)

        return t_max > 0.02 and t_min < 0.98

    def _segment_intersects_circle(self, x1, y1, x2, y2, obstacle, margin=0.0):
        radius = obstacle["r"] + margin
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-9:
            return self._point_in_obstacle(obstacle, x1, y1, margin)

        t = ((obstacle["x"] - x1) * dx + (obstacle["y"] - y1) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return math.sqrt((closest_x - obstacle["x"]) ** 2 + (closest_y - obstacle["y"]) ** 2) <= radius

    def _segment_intersects_obstacle(self, x1, y1, x2, y2, obstacle, margin=0.0):
        if obstacle["type"] == "rect":
            return self._segment_intersects_rect(x1, y1, x2, y2, obstacle, margin)
        if obstacle["type"] == "circle":
            return self._segment_intersects_circle(x1, y1, x2, y2, obstacle, margin)
        return False

    def _first_blocking_obstacle(self, x1, y1, x2, y2, margin=None, ignore=None):
        margin = self.path_clearance if margin is None else margin
        closest_obstacle = None
        closest_distance = float("inf")

        for obstacle in self.current_map_obstacles:
            if ignore is obstacle:
                continue
            if not self._segment_intersects_obstacle(x1, y1, x2, y2, obstacle, margin):
                continue

            distance = math.sqrt((obstacle["x"] - x1) ** 2 + (obstacle["y"] - y1) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_obstacle = obstacle

        return closest_obstacle

    def _waypoint_is_clear(self, wx, wy, blocking_obstacle):
        if abs(wx) > self.arena_limit - 0.4 or abs(wy) > self.arena_limit - 0.4:
            return False

        for done_x, done_y in self.completed_avoidance_targets:
            if math.sqrt((wx - done_x) ** 2 + (wy - done_y) ** 2) < self.avoidance_reached_radius:
                return False

        for obstacle in self.current_map_obstacles:
            if self._point_in_obstacle(obstacle, wx, wy, self.path_clearance * 0.7):
                return False

        start_blocked = self._first_blocking_obstacle(
            self.x, self.y, wx, wy,
            margin=0.15,
        )
        goal_blocked = self._first_blocking_obstacle(
            wx, wy, self.goal_x, self.goal_y,
            margin=0.15,
        )
        return start_blocked is None and goal_blocked is None

    def _rect_avoidance_candidates(self, obstacle):
        min_x, max_x, min_y, max_y = self._rect_bounds(
            obstacle,
            self.path_clearance + self.corner_clearance,
        )
        return [
            (min_x, min_y),
            (min_x, max_y),
            (max_x, min_y),
            (max_x, max_y),
        ]

    def _circle_avoidance_candidates(self, obstacle):
        radius = obstacle["r"] + self.path_clearance + self.corner_clearance
        return [
            (obstacle["x"] + radius, obstacle["y"]),
            (obstacle["x"] - radius, obstacle["y"]),
            (obstacle["x"], obstacle["y"] + radius),
            (obstacle["x"], obstacle["y"] - radius),
        ]

    def _obstacle_avoidance_target(self):
        if self.current_avoidance_target is not None:
            if self._first_blocking_obstacle(
                    self.x, self.y, self.goal_x, self.goal_y,
                    margin=self.path_clearance) is None:
                self.current_avoidance_target = None
                return None

            if self._target_distance(self.current_avoidance_target) <= self.avoidance_reached_radius:
                self.completed_avoidance_targets.append(self.current_avoidance_target)
                self.current_avoidance_target = None
            else:
                return self.current_avoidance_target

        blocking_obstacle = self._first_blocking_obstacle(
            self.x, self.y, self.goal_x, self.goal_y,
            margin=self.path_clearance,
        )
        if blocking_obstacle is None:
            return None

        if blocking_obstacle["type"] == "rect":
            candidates = self._rect_avoidance_candidates(blocking_obstacle)
        elif blocking_obstacle["type"] == "circle":
            candidates = self._circle_avoidance_candidates(blocking_obstacle)
        else:
            return None

        best = None
        best_score = float("inf")
        for wx, wy in candidates:
            if not self._waypoint_is_clear(wx, wy, blocking_obstacle):
                continue

            score = (
                math.sqrt((wx - self.x) ** 2 + (wy - self.y) ** 2) +
                math.sqrt((self.goal_x - wx) ** 2 + (self.goal_y - wy) ** 2)
            )
            if score < best_score:
                best_score = score
                best = (wx, wy)

        self.current_avoidance_target = best
        return best

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
        if region is None:
            return False

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

        nav_distance, nav_angle, nav_kind = self.get_navigation_error()
        if not self.goal_on_traversable or nav_kind not in ("goal", "ramp_access"):
            return False

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

        self._update_odometry()

        self.v = 0.0
        self.w = 0.0
        self.v_prev = 0.0
        self.w_prev = 0.0
        self.step_count = 0
        self.done_reason = None
        self.current_avoidance_target = None
        self.completed_avoidance_targets = []
        self.current_underpass_stage = 0

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
        # -1 - reverse scurt (folosit doar de controllerul simplu pentru recuperare)
        # 0 - forward
        # 1 - forward-left
        # 2 - forward-right
        # 3 - rotate left
        # 4 - rotate right

        if action == -1:
            v_target, w_target = -0.25,  0.0
        elif action == 0:
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
        if ranges is None or len(ranges) == 0:
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

        if horizontal_resolution > 0 and len(ranges) >= horizontal_resolution:
            detected_layers = max(1, len(ranges) // horizontal_resolution)
            layers = min(max(1, layers), detected_layers)

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

        if horizontal_resolution > 0 and len(ranges) >= horizontal_resolution:
            detected_layers = max(1, len(ranges) // horizontal_resolution)
            layers = min(max(1, layers), detected_layers)

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
