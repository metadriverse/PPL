from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.utils import Config
from metadrive.utils.math import norm
from panda3d.core import LVector3
import math
import torch
from metadrive.utils.coordinates_shift import panda_vector, metadrive_vector, panda_heading
from collections import deque
import numpy as np
import copy

class BasePredictionEnv(SafeMetaDriveEnv):
    def default_config(self) -> Config:
        config = super(BasePredictionEnv, self).default_config()
        config.update(
            {
                "future_steps_predict": 20,
                "update_future_freq": 10,
                "future_steps_preference": 3,
                "expert_noise": 0,
            },
            allow_add_new_key=True
        )
        return config

    def get_state(self):
        """
        Fetch more information
        """
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        vehicle = self.vehicle
        state = super(BaseVehicle, vehicle).get_state()
        state.update(
            {
                "steering": vehicle.steering,
                "throttle_brake": vehicle.throttle_brake,
                "crash_vehicle": vehicle.crash_vehicle,
                "crash_object": vehicle.crash_object,
                "crash_building": vehicle.crash_building,
                "crash_sidewalk": vehicle.crash_sidewalk,
                "crash_human": vehicle.crash_human,
                "size": (vehicle.LENGTH, vehicle.WIDTH, vehicle.HEIGHT),
                "length": vehicle.LENGTH,
                "width": vehicle.WIDTH,
                "height": vehicle.HEIGHT,
                "red_light": vehicle.red_light,
                "yellow_light": vehicle.yellow_light,
                "green_light": vehicle.green_light,
                "on_yellow_continuous_line": vehicle.on_yellow_continuous_line,
                "on_white_continuous_line": vehicle.on_white_continuous_line,
                "on_broken_line": vehicle.on_broken_line,
                "on_crosswalk": vehicle.on_crosswalk,
                "contact_results": list(vehicle.contact_results) if vehicle.contact_results is not None else [],
            }
        )
        state.update({
            "last_current_action": list(vehicle.last_current_action),
            "last_position": vehicle.last_position,
            "last_heading_dir": vehicle.last_heading_dir,
            "dist_to_left_side": vehicle.dist_to_left_side,
            "dist_to_right_side": vehicle.dist_to_right_side,
            "last_velocity": vehicle.last_velocity,
            "last_speed": vehicle.last_speed,
            "out_of_route": vehicle.out_of_route,
            "on_lane": vehicle.on_lane,
            "spawn_place": vehicle.spawn_place,
            "takeover": vehicle.takeover,
            "expert_takeover": vehicle.expert_takeover,
            "energy_consumption": vehicle.energy_consumption,
            "break_down": vehicle.break_down,
        })
        
        if vehicle.navigation is not None:
            state["spawn_road"] = vehicle.navigation.spawn_road
            state["destination"] = (vehicle.navigation.final_road.start_node, vehicle.navigation.final_road.end_node) if vehicle.navigation.final_road is not None else None
            state["checkpoints"] = vehicle.navigation.checkpoints 
            state["_target_checkpoints_index"] = vehicle.navigation._target_checkpoints_index

            state["current_road"] = (vehicle.navigation.current_road.start_node, vehicle.navigation.current_road.end_node) if vehicle.navigation.current_road is not None else None
            state["next_road"] = (vehicle.navigation.next_road.start_node, vehicle.navigation.next_road.end_node) if vehicle.navigation.next_road is not None else None
            state["final_road"] = (vehicle.navigation.final_road.start_node, vehicle.navigation.final_road.end_node) if vehicle.navigation.final_road is not None else None

            state["current_ref_lane_indices"] = [lane.index for lane in vehicle.navigation.current_ref_lanes] if vehicle.navigation.current_ref_lanes is not None else None
            state["next_ref_lane_indices"] = [lane.index for lane in vehicle.navigation.next_ref_lanes] if vehicle.navigation.next_ref_lanes is not None else None
            state["total_length"] = vehicle.navigation.total_length
            state["travelled_length"] = vehicle.navigation.travelled_length
            state["_last_long_in_ref_lane"] = vehicle.navigation._last_long_in_ref_lane

            state["_navi_info"] = vehicle.navigation._navi_info.tolist() if hasattr(vehicle.navigation, "_navi_info") and vehicle.navigation._navi_info is not None else None
            state["navi_arrow_dir"] = vehicle.navigation.navi_arrow_dir if hasattr(vehicle.navigation, "navi_arrow_dir") else None
            
        return copy.deepcopy(state)
        
    def set_state(self, state):
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        vehicle = self.vehicle
        super(BaseVehicle, vehicle).set_state(state)
        vehicle.set_throttle_brake(float(state["throttle_brake"]))
        vehicle.set_steering(float(state["steering"]))
        vehicle.last_current_action = deque(state["last_current_action"], maxlen=2)
        vehicle.last_position = state["last_position"]
        vehicle.last_heading_dir = state["last_heading_dir"]
        vehicle.dist_to_left_side = state["dist_to_left_side"]
        vehicle.dist_to_right_side = state["dist_to_right_side"]
        vehicle.last_velocity = state["last_velocity"]
        vehicle.last_speed = state["last_speed"]
        vehicle.out_of_route = state["out_of_route"]
        vehicle.on_lane = state["on_lane"]
        vehicle.spawn_place = state["spawn_place"]
        vehicle.takeover = state["takeover"]
        vehicle.expert_takeover = state["expert_takeover"]
        vehicle.energy_consumption = state["energy_consumption"]
        vehicle.break_down = state["break_down"]
        vehicle.crash_vehicle = state["crash_vehicle"]
        vehicle.crash_human = state["crash_human"]
        vehicle.crash_object = state["crash_object"]
        vehicle.crash_sidewalk = state["crash_sidewalk"]
        vehicle.crash_building = state["crash_building"]
        vehicle.red_light = state["red_light"]
        vehicle.yellow_light = state["yellow_light"]
        vehicle.green_light = state["green_light"]
        vehicle.on_yellow_continuous_line = state["on_yellow_continuous_line"]
        vehicle.on_white_continuous_line = state["on_white_continuous_line"]
        vehicle.on_broken_line = state["on_broken_line"]
        vehicle.on_crosswalk = state["on_crosswalk"]
        vehicle.contact_results = set(state["contact_results"]) if "contact_results" in state else set()
        if vehicle.navigation is not None:
            from metadrive.component.road_network import Road
            vehicle.navigation.spawn_road = state.get("spawn_road", None)
            dest = state.get("destination", None)
            if dest is not None:
                vehicle.navigation.final_road = Road(dest[0], dest[1])
            else:
                vehicle.navigation.final_road = None
            vehicle.navigation.checkpoints = state.get("checkpoints", None)
            vehicle.navigation._target_checkpoints_index = state.get("_target_checkpoints_index", None)
            current_road = state.get("current_road", None)
            if current_road is not None:
                vehicle.navigation.current_road = Road(current_road[0], current_road[1])
            else:
                vehicle.navigation.current_road = None

            next_road = state.get("next_road", None)
            if next_road is not None:
                vehicle.navigation.next_road = Road(next_road[0], next_road[1])
            else:
                vehicle.navigation.next_road = None

            final_road = state.get("final_road", None)
            if final_road is not None:
                vehicle.navigation.final_road = Road(final_road[0], final_road[1])
            else:
                vehicle.navigation.final_road = None
            current_ref_lane_indices = state.get("current_ref_lane_indices", None)
            if current_ref_lane_indices is not None:
                vehicle.navigation.current_ref_lanes = [vehicle.navigation.map.road_network.get_lane(idx) for idx in current_ref_lane_indices]
            else:
                vehicle.navigation.current_ref_lanes = None

            next_ref_lane_indices = state.get("next_ref_lane_indices", None)
            if next_ref_lane_indices is not None:
                vehicle.navigation.next_ref_lanes = [vehicle.navigation.map.road_network.get_lane(idx) for idx in next_ref_lane_indices]
            else:
                vehicle.navigation.next_ref_lanes = None
            vehicle.navigation.total_length = state.get("total_length", 0.0)
            vehicle.navigation.travelled_length = state.get("travelled_length", 0.0)
            vehicle.navigation._last_long_in_ref_lane = state.get("_last_long_in_ref_lane", 0.0)
            navi_info_list = state.get("_navi_info", None)
            if navi_info_list is not None:
                vehicle.navigation._navi_info = np.array(navi_info_list)
            else:
                vehicle.navigation._navi_info = None
            vehicle.navigation.navi_arrow_dir = state.get("navi_arrow_dir", None)

    
    def predict_agent_future_trajectory(self, current_obs, n_steps, action_behavior = None, return_all_states = False):
        info = dict()
        saved_state = self.get_state()
        
        all_states = []
        traj = []
        obs = current_obs
        total_reward = 0
        failure = False
        
        for step in range(n_steps):
            old_pos = copy.deepcopy(self.vehicle.position)
            action = action_behavior
            if action_behavior is None:
                action = self.agent_action
                if hasattr(self, "model"):
                     action, _ = self.model.policy.predict(obs, deterministic=True)
            
            if self.config["use_discrete"]:
                action = self.discrete_to_continuous(action)
            actions = self._preprocess_actions(action) 
            dt = self.config["physics_world_step_size"] * self.config["decision_repeat"]
            self.vehicle.before_step(action)
                
            params = self.vehicle.get_dynamics_parameters()
            mass = params["mass"]
            max_engine_force = params["max_engine_force"]
            max_brake_force = params["max_brake_force"]

            throttle = self.vehicle.throttle_brake
            if throttle >= 0:
                if self.vehicle.speed >= self.vehicle.max_speed_m_s:
                    a = 0.0
                else:
                    engine_force = max_engine_force * throttle
                    a = engine_force / mass * 4
            else:
                brake_force = max_brake_force * abs(throttle)
                a = -brake_force / mass * 4

            new_speed = self.vehicle.speed + a * dt
            new_speed = max(new_speed, 0.0)

            step_info = self.vehicle.after_step()
            current_steering = self.vehicle.steering
            max_steering_rad = math.radians(self.vehicle.config["max_steering"])

            L = self.vehicle.FRONT_WHEELBASE + self.vehicle.REAR_WHEELBASE
            new_heading = self.vehicle.heading_theta + (new_speed / L) * math.tan(current_steering * max_steering_rad) * dt

            new_x = self.vehicle.position[0] + new_speed * dt * math.cos(new_heading)
            new_y = self.vehicle.position[1] + new_speed * dt * math.sin(new_heading)
            new_position = [new_x, new_y]
            new_velocity = [new_speed * math.cos(new_heading), new_speed * math.sin(new_heading)]

            self.set_position(new_position)
            self.vehicle.set_heading_theta(new_heading)
            self.vehicle.set_velocity(new_velocity)
            self.vehicle.navigation.update_localization(self.vehicle)
            r = self.reward_function('default_agent')[0]
            total_reward += r
            
            if return_all_states:
                all_states.append(self.get_state())
            d = self.done_function('default_agent')[0]

            new_obs = self.get_single_observation().observe(self.vehicle)
            
            traj.append({
                "obs": obs.copy(),
                "action": action.copy(),
                "reward": r,
                "next_obs": new_obs.copy(),
                "done": d,
                "pos": old_pos,
                "next_pos": copy.deepcopy(self.vehicle.position),
            })
            obs = new_obs.copy()
            
            if d:
                failure = (r < 0)
                if r < 0:
                    total_reward = -100
                break
        
        self.set_state(saved_state)
        
        failure = failure or (total_reward <= 10)
        
        if total_reward <= 10:
            total_reward = -100
        info["all_states"] = all_states
        info["failure"] = failure
        info["total_reward"] = total_reward
        
        return traj, info
    
    def set_position(self, position, height=None):
        vehicle = self.vehicle
        assert len(position) == 2 or len(position) == 3
        if len(position) == 3:
            height = position[-1]
            position = position[:-1]
        else:
            if height is None:
                height = vehicle.origin.getPos()[-1]
        vehicle.origin.setPos(panda_vector(position, height))
    

    def draw_points(self, points, colors=None):
        """
        Draw a set of points with colors
        Args:
            points: a set of 3D points
            colors: a list of color for each point

        Returns: None

        """
        from panda3d.core import VBase4, NodePath, Material
        from panda3d.core import LVecBase4f
        from metadrive.engine.asset_loader import AssetLoader
        drawer = self.drawer
        new_points = []
        for k, point in enumerate(points):
            if len(drawer._dying_points) > 0:
                np = drawer._dying_points.pop()
            else:
                np = NodePath("debug_point")
                model = drawer.engine.loader.loadModel(AssetLoader.file_path("models", "sphere.egg"))
                model.setScale(drawer.scale)
                model.reparentTo(np)
            material = Material()
            if colors:
                material.setBaseColor(LVecBase4f(*colors[k]))
            else:
                material.setBaseColor(LVecBase4f(1, 1, 1, 1))
            material.setShininess(64)
            np.setMaterial(material, True)
            np.setPos(*point)
            np.reparentTo(drawer)
            drawer._existing_points.append(np)
            new_points.append(np)
        return new_points

    def _get_reset_return(self, reset_info):
        o, info = super(BasePredictionEnv, self)._get_reset_return(reset_info)
        if hasattr(self,"drawer"):
            for npp in self.drawn_points:
                npp.detachNode()
                self.drawer._dying_points.append(npp)
            self.drawn_points = []
        else:
            self.drawn_points = []
            self.drawer = self.engine.make_point_drawer(scale=3)
        return o, info
    
    
    def render_reset(self):
        if hasattr(self,"drawer"):
            drawer = self.drawer
        else:
            self.drawer = self.engine.make_point_drawer(scale=3)
        for npp in self.drawn_points:
            npp.detachNode()
            self.drawer._dying_points.append(npp)
        self.drawn_points = []
        
    def render_traj(self, predicted_traj, color):
        if self.config["use_render"]:
            points, colors = [], []
            for j in range(len(predicted_traj)):
                points.append((predicted_traj[j]["next_pos"][0], predicted_traj[j]["next_pos"][1], 0.5)) # define line 1 for test
                colors.append(np.clip(np.array([*color,1]), 0., 1.0))
            self.drawn_points = self.drawn_points + self.draw_points(points, colors)
