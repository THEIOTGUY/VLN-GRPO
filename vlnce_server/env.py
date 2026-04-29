import json
import math
import numpy as np
import os
import time
import ray
import uuid
import gzip

import habitat
from habitat.core.registry import registry
from habitat.core.embodied_task import Measure
from VLN_CE.habitat_extensions.task import RxRVLNCEDatasetV1
from dtw import dtw
from fastdtw import fastdtw
from VLN_CE.vlnce_baselines.config.default import get_config

from .utils.context_utils import convert_numpy_to_PIL
from .utils.parse_utils import PARSE_FUNC_MAP
from .utils.draw_utils import render_frame_pil, save_video
from .utils.parse_utils import parse_action
from .constants import (
    ACTION_LOOKUP, STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT,
    REASON_FORMAT_MISMATCH, REASON_SUCCESS, REASON_GOAL_NOT_REACHED,
    REWARD_SUCCESS, REWARD_SUCCESS_NDTW, REWARD_WEIGHTED_SUCCESS,
    REWARD_WEIGHTED_SUCCESS_NDTW, REASON_EPISODE_STEPS_EXCEEDED,
    REASON_EPISODE_TURNS_EXCEEDED, REASON_OBSTACLE_REPLAN,
    DEFAULT_OBSTACLE_DEPTH_THRESHOLD,
)
from .prompt import init_observation_template, action_template, format_prompt
from .env_config import VLNCEEnvConfig
from .memory import SpatialMemory


def euclidean_distance(
    pos_a, pos_b
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)

 
@ray.remote
class VLNDataset:
    def __init__(self, config_path):
        
        config = get_config(config_path, None)
        self.config = config
        dataset = habitat.datasets.make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)

        self.episodes = dataset.episodes
        self.mapping = {self.episodes[i].episode_id:i for i in range(len(self.episodes))}

        self.gt_json = {}
        
        if "{role}" in config.TASK_CONFIG.TASK.NDTW.GT_PATH:
            # rxr
            for role in RxRVLNCEDatasetV1.annotation_roles:
                with gzip.open(config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=config.TASK_CONFIG.TASK.NDTW.SPLIT, role=role), "rt") as f:
                    self.gt_json.update(json.load(f))  
        else:
            # r2r
            with gzip.open(config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=config.TASK_CONFIG.TASK.NDTW.SPLIT), "rt") as f:
                self.gt_json.update(json.load(f))  
                
        # slim dataset to reduce memory usage
        dataset.episodes = dataset.episodes[:2]        
        self.dataset = dataset
   
    def get_episode(self, episode_id):
        assert str(episode_id) in self.mapping
        i = self.mapping[str(episode_id)]
        return self.episodes[i]
    
    def get_gt_locations(self, episode_id):
        assert episode_id in self.gt_json
        return self.gt_json[episode_id]["locations"]

    def get_task_config(self):
        return self.config.TASK_CONFIG
    
    def get_slim_dataset(self):
        return self.dataset
        
    
@ray.remote
class Simulator:
    def __init__(self, gpu_id, dataset: VLNDataset):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.dataset = dataset
        self._init_env()
        self._checkpoint = {}

    def _init_env(self):
        habitat_cfg, habitat_dataset = ray.get([
            self.dataset.get_task_config.remote(),
            self.dataset.get_slim_dataset.remote(),
        ])
        self._register_ndtw_measure()
        env = habitat.Env(habitat_cfg,dataset=habitat_dataset)
        self.env = env   
        
    def _register_ndtw_measure(self):
    # Re-register the NDTW measure to save memory, by avoiding duplicated GT_PATH loading
        dataset = self.dataset
        
        @registry.register_measure
        class NDTW(Measure):
            """NDTW (Normalized Dynamic Time Warping)
            ref: https://arxiv.org/abs/1907.05446
            """

            cls_uuid: str = "ndtw"

            def __init__(
                self, *args, sim, config, **kwargs
            ):
                self._sim = sim
                self._config = config
                self.dtw_func = fastdtw if config.FDTW else dtw
                super().__init__()

            def _get_uuid(self, *args, **kwargs) -> str:
                return self.cls_uuid

            def reset_metric(self, *args, episode, **kwargs):
                self.locations = []
                self.gt_locations = ray.get(dataset.get_gt_locations.remote(episode.episode_id))
                self.update_metric()

            def update_metric(self, *args, **kwargs):
                current_position = self._sim.get_agent_state().position.tolist()
                if len(self.locations) == 0:
                    self.locations.append(current_position)
                else:
                    if current_position == self.locations[-1]:
                        return
                    self.locations.append(current_position)

                dtw_distance = self.dtw_func(
                    self.locations, self.gt_locations, dist=euclidean_distance
                )[0]

                nDTW = np.exp(
                    -dtw_distance
                    / (len(self.gt_locations) * self._config.SUCCESS_DISTANCE)
                )
                self._metric = nDTW                              

        import VLN_CE
        VLN_CE.habitat_extensions.measures.NDTW = NDTW
    
    def start_new_episode(self, episode_id):
        self._checkpoint = {}
        env = self.env
    
        env._reset_stats()
        assert len(env.episodes) > 0, "Episodes list is empty"
        if env._current_episode is not None:
            env._current_episode._shortest_path_cache = None
        
        env._current_episode = ray.get(self.dataset.get_episode.remote(episode_id))
        env.reconfigure(env._config)

        observations = env.task.reset(episode=env.current_episode)
        env._task.measurements.reset_measures(
            episode=env.current_episode, task=env.task
        )
        return observations
    
    def save_current_state(self):
        env = self.env
        
        previous_state = env.sim.get_agent_state()
        previous_metrics = env.get_metrics()
        locations_length = len(env._task.measurements.measures['ndtw'].locations)
        
        state_id = str(uuid.uuid4())
        self._checkpoint[state_id] = {
            "episode_id": env.current_episode.episode_id,
            "state": {
                "previous_state": previous_state,
                "previous_metrics": previous_metrics,
                "locations_length": locations_length,
                "elapsed_steps": env._elapsed_steps,
            }
        }
        return state_id
        
    def load_state(self, state_id):
        assert state_id in self._checkpoint, f"State ID {state_id} not found in checkpoint"
        state = self._checkpoint[state_id]["state"]
        episode_id = self._checkpoint[state_id]["episode_id"]
        assert episode_id == self.env.current_episode.episode_id, f"Episode ID {episode_id} does not match current episode ID {self.env.current_episode.episode_id}"

        previous_state = state["previous_state"]
        previous_metrics = state["previous_metrics"]
        locations_length = state["locations_length"]
        env = self.env
        reset_keys = ['ndtw', 'success', 'spl', 'oracle_success', 'oracle_spl']
        
        # mimic `_reset_stats``
        env._episode_start_time = time.time()
        env._elapsed_steps = state["elapsed_steps"]
        env._episode_over = False

        env.sim.set_agent_state(previous_state.position,previous_state.rotation)
        env._task.measurements.measures['ndtw'].locations = env._task.measurements.measures['ndtw'].locations[:locations_length]
        env._task.measurements.update_measures(episode=env.current_episode, action=None, task=env._task)
        for key, value in previous_metrics.items():
            if key in reset_keys:
                env._task.measurements.measures[key]._metric = value

    def step(self, action_index):
        assert action_index != 0
        return self.env.step(action_index)
    
    def get_metrics(self):
        return self.env.get_metrics()
    
    def close(self):
        assert False, "should not close the env"
        return self.env.close()
    
    def get_sensor_observations(self):
        return self.env.sim.get_sensor_observations()

    def get_depth_observation(self):
        """Return the raw depth frame (H×W float32, metres) or None if unavailable."""
        try:
            obs = self.env.sim.get_sensor_observations()
            return obs.get('depth', None)
        except Exception:
            return None

    def get_agent_position(self) -> np.ndarray:
        """Return current agent 3-D position as (x, y, z) numpy array."""
        return np.array(self.env.sim.get_agent_state().position)

    def get_locations(self):
        return {
            "locations": np.array(self.env._task.measurements.measures['ndtw'].locations),
            "gt_locations": np.array(self.env._task.measurements.measures['ndtw'].gt_locations),
        }

    
class VLNCEEnv:

    def __init__(self, config: VLNCEEnvConfig, simulator, save_video_dir):
        self.config = config
        self.save_video_dir = save_video_dir

        self.sim = simulator
        obs = ray.get(self.sim.start_new_episode.remote(self.config.episode_id))

        self.instruction = obs["instruction"]["text"]
        self._current_local_step = 0  # excludes gt actions
        self._current_local_turn = 0
        self._history_renders = []    # stores gt action renders
        self._start_step = len(self.config.history_actions) + 1
        self._gen_traj = []           # (response, actions, reward) per turn
        self._save_as_video_raw = []
        self._reset_keys = ['ndtw', 'success', 'spl', 'oracle_success', 'oracle_spl']
        self._reward_list = []

        # Format/parse helpers
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
        self._episode_start_time = 0
        self._success = False

        # --- Spatial memory module ---
        if self.config.enable_memory_module:
            self._spatial_memory: SpatialMemory = SpatialMemory(
                grid_resolution=self.config.memory_grid_resolution,
            )
            self._spatial_memory.landmarks = SpatialMemory.extract_landmarks(self.instruction)
        else:
            self._spatial_memory = None

        # --- Semantic landmark reward ---
        # GT waypoints are loaded after _get_initial_checkpoint; populated there.
        self._landmark_waypoints: list = []   # list of (label, np.ndarray) pairs
        self._landmark_reward_total: float = 0.0

        # Obstacle replanning state
        self._last_replan_step: int = -1  # prevent firing twice for the same step

        self.initial_state_id = self._get_initial_checkpoint()

    def _get_initial_checkpoint(self):
        history_actions = self.config.history_actions
        
        self._history_renders.append(self._render(init_obs=True))
        self._save_as_video_raw.append({
            "render": self._history_renders[-1],
            "metrics": self._get_metrics(),
        })
        
        max_actions_per_step = self.config.get('max_actions_per_step')
        action_sep = self.config.get('action_sep')
        
        i = 0
        while i < len(history_actions):
            ready_actions = history_actions[i:i + max_actions_per_step]
            i += max_actions_per_step
            
            for j, action_str in enumerate(ready_actions):
                action_name, action_value = parse_action(action_str, action_space=self.config.action_space)
                action_index = ACTION_LOOKUP[action_name]
                num_action_repeat = 0
                if action_name == STOP:
                    assert False
                elif action_name == MOVE_FORWARD:
                    num_action_repeat = action_value // 25
                elif action_name == TURN_LEFT or action_name == TURN_RIGHT:
                    if self.config.action_space == 'r2r':
                        num_action_repeat = action_value // 15
                    elif self.config.action_space == 'rxr':
                        num_action_repeat = action_value // 30
                    else:
                        assert False
                else:
                    assert False

                for _ in range(num_action_repeat):
                    ray.get(self.sim.step.remote(action_index))
                    
                if j == len(ready_actions) - 1:
                    fake_response = action_sep.join(ready_actions) # Fake response in the format of the agent's response
                    self._history_renders.append({
                        **self._render(init_obs=False),
                        "response": fake_response,
                    })
                    
                    if self.config.save_as_video:
                        self._save_as_video_raw[-1]["response"] = fake_response
                        self._save_as_video_raw.append({
                            "render": self._history_renders[-1],
                            "metrics": self._get_metrics(),
                        })            
        locations = ray.get(self.sim.get_locations.remote())
        self.history_locations = locations["locations"]
        gt_locations = locations["gt_locations"]
        assert len(self.history_locations) <= len(gt_locations)

        mean_diff = np.linalg.norm(self.history_locations - gt_locations[:len(self.history_locations)], axis=1).mean()
        print(f"Mean difference between history locations and gt locations: {mean_diff:.4f}")

        # Build landmark waypoints: pair each GT location with the landmark
        # vocabulary extracted from the instruction.  During a rollout the env
        # checks proximity to these positions to award the landmark bonus.
        if self.config.enable_landmark_reward and self._spatial_memory is not None:
            lm_labels = self._spatial_memory.landmarks or ["waypoint"]
            step = max(1, len(gt_locations) // max(len(lm_labels), 1))
            for i, label in enumerate(lm_labels):
                idx = min(i * step, len(gt_locations) - 1)
                self._landmark_waypoints.append((label, gt_locations[idx]))

        return ray.get(self.sim.save_current_state.remote())
        
        
    def reset(self, seed=None):
        assert len(self._history_renders) == math.ceil(len(self.config.history_actions) / self.config.max_actions_per_step) + 1
        if self.config.save_as_video:
            self._save_as_video_raw = self._save_as_video_raw[:len(self._history_renders)]

        assert self._start_step == len(self.config.history_actions) + 1, \
            f"Start step {self._start_step} does not match history actions length {len(self.config.history_actions)}"
        self._current_local_step = 0
        self._current_local_turn = 0
        self._episode_start_time = time.time()
        self._success = False
        self._gen_traj = []
        self._reward_list = []
        self._last_replan_step = -1
        self._landmark_reward_total = 0.0

        if self._spatial_memory is not None:
            self._spatial_memory.reset()

        ray.get(self.sim.load_state.remote(self.initial_state_id))

        return self._history_renders, {}

    def step(self, response: str):
        self._current_local_turn += 1
        rst = self.parse_func(
            response=response,
            special_token_list=[],
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
        )
        
        format_correct = len(rst['actions']) > 0 and rst.get("format_correct", True)
        parsed_action_list = [] # each element is a tuple of (action_name, action_value, action_str)
        
        for action in rst['actions']:
            try:
                action_name, action_value = parse_action(action, action_space=self.config.action_space)

                parsed_action_list.append((action_name, action_value, action))
            except ValueError as e:
                print(f"Failed to parse action: {action}. Error: {e}")
                parsed_action_list.append((None, None, action))
                format_correct = False
        
        reward = 0.0    # step reward
        done = False    # whether the episode is over
        success = False # whether the episode is successful (reached the goal)
        end_reason = '' # reason for ending the episode
        
        reward_components = {
            "success_reward": 0.0,
            "ndtw_reward": 0.0,
            "landmark_reward": 0.0,
        }
                
        def set_ending_signals(is_goal_reached, reason):
            nonlocal done, success, reward, end_reason
            end_reason = reason
            done = True
            success = is_goal_reached
            
            metr = self._get_metrics(return_gen_ndtw=True)
            ndtw = metr['gen_ndtw']
            # sometimes d2g goes inf. we need to be careful with that
            d2g = metr["distance_to_goal"]
            if math.isinf(d2g):
                print("Warning: distance to goal is inf")
            
            if self.config.reward_type == REWARD_SUCCESS:
                reward_components["success_reward"] = is_goal_reached * self.config.success_reward_base
                reward_components["ndtw_reward"] = 0.0
                
            elif self.config.reward_type == REWARD_WEIGHTED_SUCCESS:
                reward_components["success_reward"] = is_goal_reached * self.config.success_reward_base * (1 - min(d2g / 3.0, 1))
                reward_components["ndtw_reward"] = 0.0
                
            elif self.config.reward_type == REWARD_SUCCESS_NDTW:
                reward_components["success_reward"] = is_goal_reached * self.config.success_reward_base
                if reason in [REASON_SUCCESS, REASON_GOAL_NOT_REACHED]:
                    reward_components["ndtw_reward"] = ndtw * self.config.ndtw_reward_base
                else:
                    reward_components["ndtw_reward"] = 0.0
                    
            elif self.config.reward_type == REWARD_WEIGHTED_SUCCESS_NDTW:
                reward_components["success_reward"] = is_goal_reached * self.config.success_reward_base * (1 - min(d2g / 3.0, 1))
                if reason in [REASON_SUCCESS, REASON_GOAL_NOT_REACHED]:
                    reward_components["ndtw_reward"] = ndtw * self.config.ndtw_reward_base
                else:
                    reward_components["ndtw_reward"] = 0.0
            else:
                raise ValueError(f"Unknown reward type: {self.config.reward_type}")
            
            reward += sum(reward_components.values())
            assert isinstance(reward, (int, float)) and not math.isnan(reward)
            self._reward_list.append(reward)

        # --- Obstacle replanning check & memory update are done per-action below ---
        obstacle_detected = False
        replan_triggered = False

        if self._current_local_turn > self.config.turn_budget:
            set_ending_signals(
                is_goal_reached=False,
                reason=REASON_EPISODE_TURNS_EXCEEDED,
            )  
        elif not format_correct:
            set_ending_signals(
                is_goal_reached=False, 
                reason=REASON_FORMAT_MISMATCH
            )
        else:
            assert len(parsed_action_list) <= self.config.get('max_actions_per_step')
            for action_name, action_value, action_str in parsed_action_list:
                
                assert action_name is not None

                self._current_local_step += 1

                # For sake of efficiency, we don't execute the stop action in the simulator.
                if action_name != STOP:
                    self._execute_action(action_name, action_value)

                    # --- Spatial memory update (post-action position) ---
                    if self._spatial_memory is not None:
                        agent_pos = ray.get(self.sim.get_agent_position.remote())
                        self._spatial_memory.update(agent_pos)

                        # --- Semantic landmark reward ---
                        if self.config.enable_landmark_reward and self._landmark_waypoints:
                            newly_found = self._spatial_memory.check_landmark_proximity(
                                self._landmark_waypoints,
                                threshold=self.config.landmark_proximity_threshold,
                            )
                            lm_bonus = len(newly_found) * self.config.landmark_reward_base
                            if lm_bonus > 0:
                                reward_components["landmark_reward"] += lm_bonus
                                reward += lm_bonus
                                self._landmark_reward_total += lm_bonus

                    # --- Closed-loop obstacle replanning ---
                    if (
                        self.config.enable_obstacle_replanning
                        and self._current_local_step != self._last_replan_step
                        and self._detect_obstacle()
                    ):
                        obstacle_detected = True
                        replan_triggered = True
                        self._last_replan_step = self._current_local_step

                # Check whether the conditions of terminating the episode are met
                if action_name == STOP:
                    metr = self._get_metrics()
                    oracle_success = metr['oracle_success']

                    if oracle_success:
                        # successfully reached the goal
                        set_ending_signals(
                            is_goal_reached=True, 
                            reason=REASON_SUCCESS, 
                        )
                    else:
                        # stopped but goal not reached
                        set_ending_signals(
                            is_goal_reached=False, 
                            reason=REASON_GOAL_NOT_REACHED
                        )
                elif self._current_local_step > self.config.step_budget:
                    # episode length exceeded
                    set_ending_signals(
                        is_goal_reached=False,
                        reason=REASON_EPISODE_STEPS_EXCEEDED,
                    )
                
                # Terminate the episode if `done` is set to True (by the set_ending_signals function)
                if done:
                    break
    
        # Format reward — added after set_ending_signals so we patch the last entry
        if format_correct and self.config.format_reward:
            reward += self.config.format_reward
            if self._reward_list:
                self._reward_list[-1] += self.config.format_reward

        # Update info dict
        metrics = self._get_metrics()
        distance_to_goal = metrics['distance_to_goal']
        oracle_success = metrics['oracle_success']
        
        self._gen_traj.append({
            "response": response,
            "extracted_actions": rst['actions'],
            "reward": reward,
        })
        
        info = {
            **rst,

            # outcome of this step
            "done": done,
            "task_success": success,
            "end_reason": end_reason,

            # latest metrics
            "distance_to_goal": distance_to_goal,
            "oracle_success": oracle_success,

            # episode metadata
            "episode_id": self.config.episode_id,
            "data_source": self.config.data_source,
            "action_space": self.config.action_space,
            "instruction": self.instruction,

            # step counters
            "env_local_step": self._current_local_step,
            "env_global_step": self._current_local_step + len(self.config.history_actions),
            "episode_elapsed_seconds": time.time() - self._episode_start_time,
            "global_start_step": self._start_step,

            # budgets
            "step_budget": self.config.step_budget,
            "turn_budget": self.config.turn_budget,

            # rewards
            "total_reward": sum(self._reward_list),
            "reward_components": reward_components,

            # trajectory generated so far
            "gen_traj": self._gen_traj,

            # --- Closed-loop replanning ---
            "obstacle_detected": obstacle_detected,
            "replan_triggered": replan_triggered,

            # --- Landmark reward ---
            "landmark_reward_total": self._landmark_reward_total,
        }
                
        render_result = self._render(replan=replan_triggered)

        if self.config.save_as_video:
            self._save_as_video_raw[-1]["response"] = response
            self._save_as_video_raw.append({
                "render": render_result,
                "metrics": self._get_metrics(),
            })
            if done:
                video_path = self._save_as_video(self._save_as_video_raw, success)
                info['saved_video_path'] = video_path
        
        return render_result, reward, done, info
    
    def _save_as_video(self, dict_list, success):
        if not self.config.save_as_video:
            return
        
        assert len(dict_list) > len(self._history_renders)
        
        n_total_steps = len(dict_list) - 1
        n_gt_steps = len(self._history_renders) - 1 
        n_gen_steps = n_total_steps - n_gt_steps
        
        video_path = os.path.join(self.save_video_dir, self.config.experiment_name, self.config.data_source, str(self.config.episode_id))
        if success:
            video_path = os.path.join(video_path, "success")
        else:
            video_path = os.path.join(video_path, "failure")

        os.makedirs(video_path, exist_ok=True)

        video_path = os.path.join(video_path,
            f"{n_total_steps}={n_gt_steps}+{n_gen_steps}_{int(time.time())}.mp4"
        )

        video_frames = []
        for i, item in enumerate(self._save_as_video_raw, 1):
            
            if i < len(self._save_as_video_raw):
                response = item["response"]
                
                if i <= n_gt_steps:
                    img_text = f"[GT] {response}"
                else:
                    img_text = f"[Gen] {response}"
                
                user_text = item['render']['obs_str']
                assistant_text = img_text
            else:
                user_text = ''
                assistant_text = ''
                    
            status = {
                'Distance2Goal': f"{item['metrics']['distance_to_goal']:.2f}m", 
                'Path Length': f"{item['metrics']['path_length']:.2f}m",
                'nDTW': f"{item['metrics']['ndtw']:.2f}", 
                'Step': f"{i}/{n_total_steps}" if i <= n_total_steps else "done",
            }
                
            output_im = np.array(item["render"]["multi_modal_data"][self.config.image_placeholder][0])
            render_img = render_frame_pil(
                image_np=output_im,
                status_info=status,
                instruction=self.instruction,
                user_text=user_text,
                assistant_text=assistant_text,
                image_width=640,
                image_height=480,
                text_width=640,
                font_path=os.path.join(os.path.dirname(__file__), "resources/NotoSans-Regular.ttf"),
                bold_font_path=os.path.join(os.path.dirname(__file__), "resources/NotoSans-Bold.ttf"),
            )
            video_frames.append(render_img)
            
        video_frames = np.array(video_frames)
        save_video(video_frames, video_path, fps=1)
        print(f"Video saved to {video_path}")
        return video_path

    def close(self):
        raise NotImplementedError("VLNCEEnv close method should not be called directly. Should be handled by the resource pool.")

    # ------------------------------------------------------------------
    # Obstacle detection
    # ------------------------------------------------------------------

    def _detect_obstacle(self) -> bool:
        """Return True if the central field of view is blocked by a near obstacle.

        Uses the raw depth image from the simulator.  The central H/3 × W/3
        crop is checked; if any pixel is below obstacle_depth_threshold the
        path is considered blocked.  Returns False gracefully when depth is
        unavailable (sensor not configured, or remote call fails).
        """
        try:
            depth = ray.get(self.sim.get_depth_observation.remote())
            if depth is None:
                return False
            threshold = self.config.get('obstacle_depth_threshold', DEFAULT_OBSTACLE_DEPTH_THRESHOLD)
            h, w = depth.shape[:2]
            # Central horizontal strip, middle third vertically
            crop = depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
            return bool(np.any(crop < threshold))
        except Exception:
            return False

    def _execute_action(self, action_name, action_value):
        assert action_name in ACTION_LOOKUP, f"Invalid action name: {action_name}"
        action_index = ACTION_LOOKUP[action_name]
        if action_name == STOP:
            assert False, "Stop action should not be executed"
        elif action_name == MOVE_FORWARD:
            assert action_value in [25, 50, 75]
            for _ in range(action_value // 25):
                ray.get(self.sim.step.remote((action_index)))
        elif action_name == TURN_LEFT or action_name == TURN_RIGHT:
            if self.config.action_space == "r2r":
                assert action_value in [15, 30, 45], "Turn left/right action should be 15, 30 or 45 degrees"
                for _ in range(action_value // 15):
                    ray.get(self.sim.step.remote(action_index))
            elif self.config.action_space == "rxr":
                assert action_value in [30, 60, 90], "Turn left/right action should be 30, 60 or 90 degrees"
                for _ in range(action_value // 30):
                    ray.get(self.sim.step.remote(action_index))
            else:
                assert False, f"Unknown action space: {self.config.action_space}"
        
    def _get_metrics(self, return_gen_ndtw=False):
        all_metrics = ray.get(self.sim.get_metrics.remote())
        distance_to_goal = all_metrics['distance_to_goal']
        oracle_success = distance_to_goal <= 3.0
        metrics = {
            "oracle_success": oracle_success,
            "distance_to_goal": distance_to_goal,
            "ndtw": all_metrics['ndtw'],
            "path_length": all_metrics['path_length'],
            
        }
        if return_gen_ndtw:
            locations = ray.get(self.sim.get_locations.remote())
            gt_locations_remain = locations["gt_locations"][len(self.history_locations):]
            gen_locations = locations["locations"][len(self.history_locations):]
            if len(gt_locations_remain) == 0 and len(gen_locations) == 0:
                gen_nDTW = 1.0
            elif len(gt_locations_remain) == 0:
                gen_nDTW = 0.0
            elif len(gen_locations) == 0:
                gen_nDTW = 0.0
            else:
                dtw_distance = fastdtw(gen_locations, gt_locations_remain, dist=euclidean_distance)[0]
                gen_nDTW = np.exp(
                    -dtw_distance
                    / (len(gt_locations_remain) * 3.0)
                )
                # self._config.SUCCESS_DISTANCE = 3.0
            metrics.update({
                "gen_ndtw": gen_nDTW,
            })
        
        return metrics
    
    def _render(self, init_obs: bool = False, replan: bool = False):
        """Render the current environment state as a VLM observation dict.

        Returns:
            dict with keys:
                "obs_str" – formatted text prompt for the VLM
                "multi_modal_data" – {image_placeholder: [PIL.Image]}
        """
        if init_obs:
            assert len(self._history_renders) == 0
        img_placeholder = self.config.get("image_placeholder", "<image>")

        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False,
        )

        frame = ray.get(self.sim.get_sensor_observations.remote())['rgb'][:, :, :-1]
        frame = np.copy(frame)
        multi_modal_data = {
            img_placeholder: [convert_numpy_to_PIL(frame)]
        }

        memory_summary = (
            self._spatial_memory.get_summary()
            if self._spatial_memory is not None
            else ""
        )

        if init_obs:
            obs_str = init_observation_template(
                observation=img_placeholder,
                instruction=self.instruction,
                memory_summary=memory_summary,
            ) + "\n" + format_prompt_text
        else:
            obs_str = action_template(
                observation=img_placeholder,
                instruction=self.instruction,
                memory_summary=memory_summary,
                replan=replan,
            ) + "\n" + format_prompt_text

        return {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data,
        }