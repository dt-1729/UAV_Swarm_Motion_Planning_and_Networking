# Copyright © MERL 2026
# Description: Defines the multi-agent 3D UAV swarm environment and reward dynamics.
import numpy as np
import cvxpy as cp
import torch
import cbf_filter as cf
import ma_connectivity_filter as mcf

class UAVSwarmEnv3D:

	def __init__(self, n_uavs, n_sensors, n_obstacles, box_size, min_uav_height, com_params, simulation_params, reward_params, power_params, head_v_max, follower_v_max, seed=0, sensor_sequence_strategy="random", sensor_init_mode="stratified", n_sensor_clusters=3, n_chains=1, connectivity_update_freq=1):

		# initialize environment parameters
		if n_uavs < 2:
			raise ValueError("n_uavs must be >= 2 (one head UAV plus at least one follower)")
		self.n_uavs = n_uavs
		# maximum number of active links per UAV in the connectivity graph
		self.max_uav_degree = 4
		# relative slack when deciding whether a link is still in range at graph refresh
		self.link_range_tolerance = 1e-3
		# the connectivity filter keeps links inside this fraction of u2u_range so they never sit on the boundary
		self.filter_range_margin = 0.98
		self.filter_failures = 0
		self.n_sensors = n_sensors
		self.box_size = box_size
		self.seed = seed
		self.rng = np.random.RandomState(self.seed)
		self.sensor_sequence_strategy = sensor_sequence_strategy
		self.sensor_init_mode = str(sensor_init_mode).lower()
		self.n_sensor_clusters = max(1, int(n_sensor_clusters))
		self.connectivity_update_freq = max(1, int(connectivity_update_freq))
		self.update_counter = 0
		self.cumulative_reward = 0.0

		# head UAV
		self.head_uav_idx = 0
		# CBF connectivity constraint neighbor choice: "upstream" or "nearest".

		# tree topology: distribute n_followers evenly across n_chains branches from head
		self.n_chains = max(1, n_chains)
		self._build_chain_topology(self.n_chains)
		self.reparent_on_init_failure = False
		self.enforce_chain_spawn = True
		self.enforce_nonchild_outside_head_range = True

		# set 3D cube region as operational area
		self.space_min = np.array([0.0, 0.0, 0.0])
		self.space_max = np.array([1.0, 1.0, 0.4])*self.box_size
		self.space_span = self.space_max - self.space_min

		# communication parameters
		self.max_dist = np.max(np.abs(self.space_span))
		self.u2s_range = com_params['u2s_range']*self.max_dist/100.0
		self.u2u_range = com_params['u2u_range']*self.max_dist/100.0
		self.u2b_range = com_params['u2b_range']*self.max_dist/100.0
		self.u2u_safe_dist = com_params['u2u_safe_dist']*self.max_dist/100.0
		self.u2s_visibility_range = com_params['u2s_visibility_range']*self.max_dist/100.0
		if self.u2u_safe_dist >= self.u2u_range:
			raise ValueError("u2u_safe_dist must be smaller than u2u_range after scaling.")

		print(f"UAV-to-sensor range: {self.u2s_range}")
		print(f"UAV-to-UAV range: {self.u2u_range}")
		print(f"UAV-to-base station range: {self.u2b_range}")
		print(f"UAV-to-UAV safe distance: {self.u2u_safe_dist}")

		# lower height limit for UAVs
		self.min_uav_height = min_uav_height*self.max_dist/100
		if self.min_uav_height >= self.u2s_range:
			raise ValueError("min_uav_height must be smaller than u2s_range after scaling.")		
		print(f"Minimum UAV height: {self.min_uav_height}")

		# initialize base_station location
		self.bs_loc = self.rng.uniform(0.7*self.space_max, self.space_max, (3,))

		# initialize sensor locations, and visited state
		self.init_sensor_locs()

		# intialize uav locations
		try:
			self.init_uav_locs()
		except RuntimeError:
			# Retry with relaxed head-range exclusion when strict spawn geometry
			# leaves no feasible sample for deeper chain followers.
			_prev = getattr(self, "enforce_nonchild_outside_head_range", True)
			self.enforce_nonchild_outside_head_range = False
			try:
				self.init_uav_locs()
			finally:
				self.enforce_nonchild_outside_head_range = _prev

		# initialize uav control actions
		self.uav_control = np.zeros((self.n_uavs, 3))

		# initialize visible sensor set for each UAV
		self.visible_sensors = [set() for _ in range(self.n_uavs)]
		self.visible_sensors[self.head_uav_idx] = set(range(self.n_sensors))
		self.visible_sensor_counts = np.zeros(self.n_uavs, dtype=int)
		self.uav_sensor_coverage = np.zeros(self.n_uavs, dtype=float)

		# initialize UAV neighbor sets and connectivity matrix
		self.uav_neighbors = [set() for _ in range(self.n_uavs)]
		self.update_uav_connectivity()
		self.update_counter = 0  # Reset counter after initialization

		# reward shaping coefficients
		self.sensor_collect_coeff = reward_params.get('sensor_collect_coeff', np.array([1.0]*self.n_uavs, dtype=float))
		self.sensor_collect_coeff[self.head_uav_idx] = 0.0  # head UAV does not contribute to sensor collection reward
		self.time_penalty_coeff = reward_params.get('time_penalty_coeff', 1.0)
		# self.energy_consumption_coeff = reward_params.get('energy_consumption_coeff', 1e-3)
		# self.terminal_sensor_coverage_coeff = reward_params.get('terminal_sensor_coverage_coeff', 0.0)
		self.potential_diff_coeff = reward_params.get('potential_diff_coeff', np.array([1000.0]*self.n_uavs, dtype=float))
		self.potential_diff_coeff[self.head_uav_idx] = 0.0  # head UAV does not contribute to potential difference reward
		self.min_uav_height_penalty_coeff = reward_params.get('min_uav_height_penalty_coeff', 10.0)
		self.collision_penalty_coeff = reward_params.get('collision_penalty_coeff', 10.0)
		self.connectivity_penalty_coeff = reward_params.get('connectivity_penalty_coeff', 10.0)
		self.cbf_control_penalty_coeff = reward_params.get('cbf_control_penalty_coeff', 0.1)

		# initialze rewards
		self.collision_penalty = 0.0
		self.connectivity_penalty = 0.0
		self.sensor_potential_reward = 0.0
		self.sensor_collection_reward = 0.0

		# violation counts
		self.connectivity_violations = 0
		self.collision_violations = 0

		# power parameters
		self.power_params = power_params

		# initialize maximum UAV speed (m/s)
		self.head_v_max = head_v_max
		self.follower_v_max = follower_v_max

		# simulation parameters
		self.current_time = simulation_params['init_time']
		self.dt = simulation_params['dt']
		self.max_time = simulation_params['max_time']
		self.max_energy = simulation_params.get('max_energy', 1.0e5)
		self.remaining_energy = self.max_energy * np.ones(self.n_uavs)
		self.energy_eps = 1e-6

		# energy and time to reach sensors
		self.sensor_energy_to_reach = np.zeros((self.n_uavs, self.n_sensors))
		self.sensor_time_to_reach = np.zeros((self.n_uavs, self.n_sensors))
		for i in range(self.n_uavs):
			for j in range(self.n_sensors):
				# sets the initial energy required to reach the sensors
				if i == self.head_uav_idx:
					self.compute_head_sensor_potential(uav_id=i, sensor_idx=j)
				elif i != self.head_uav_idx:
					self.compute_follower_sensor_potential(uav_id=i, sensor_idx=j)

	def _build_chain_topology(self, n_chains):
		"""Build uav_parent and uav_parent_template for a given number of chains."""
		n_followers = self.n_uavs - 1
		n_chains = max(1, min(int(n_chains), n_followers)) if n_followers > 0 else 1
		self.n_chains = n_chains
		base, remainder = divmod(n_followers, n_chains)
		chain_lengths = [base + (1 if c < remainder else 0) for c in range(n_chains)]
		self.uav_parent = np.full(self.n_uavs, -1, dtype=int)
		uav_idx = 1
		for chain_len in chain_lengths:
			chain_parent = self.head_uav_idx
			for _ in range(chain_len):
				self.uav_parent[uav_idx] = chain_parent
				chain_parent = uav_idx
				uav_idx += 1
		self.uav_parent_template = self.uav_parent.copy()


	def _normalize_positions(self, positions):
		"""Normalize coordinates per axis using [space_min, space_max]."""
		positions = np.asarray(positions, dtype=float)
		return (positions - self.space_min) / np.abs(np.max(self.space_span))


	def _compute_sensor_visit_sequence(self):
		"""
		Build a sensor visit sequence based on the configured strategy.
		Strategies:
		- "random": random permutation of sensors
		- "tsp": multi-start nearest-neighbor + 2-opt improved path
		- "ordered": [0, 1, ..., n_sensors - 1]
		"""
		strategy = str(getattr(self, "sensor_sequence_strategy", "random")).lower()

		if strategy == "random":
			return self.rng.permutation(int(0.5*self.n_sensors)).astype(int)

		if strategy == "tsp":
			if self.n_sensors <= 1:
				return np.arange(self.n_sensors, dtype=int)

			coords = self.sensor_locs
			dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
			n = self.n_sensors

			def path_cost(seq):
				if len(seq) <= 1:
					return 0.0
				return float(sum(dist[seq[k], seq[k + 1]] for k in range(len(seq) - 1)))

			def nearest_neighbor_path(start_idx):
				remaining = set(range(n))
				path = [int(start_idx)]
				remaining.remove(int(start_idx))
				while remaining:
					last_idx = path[-1]
					next_idx = min(remaining, key=lambda idx: dist[last_idx, idx])
					path.append(int(next_idx))
					remaining.remove(int(next_idx))
				return path

			def two_opt_path(path):
				improved = True
				while improved:
					improved = False
					for i in range(1, n - 2):
						for j in range(i + 1, n - 1):
							a, b = path[i - 1], path[i]
							c, d = path[j], path[j + 1]
							current = dist[a, b] + dist[c, d]
							candidate = dist[a, c] + dist[b, d]
							if candidate + 1e-12 < current:
								path[i : j + 1] = reversed(path[i : j + 1])
								improved = True
				return path

			best_path = None
			best_cost = float("inf")
			for start_idx in range(n):
				candidate_path = nearest_neighbor_path(start_idx)
				candidate_path = two_opt_path(candidate_path)
				candidate_cost = path_cost(candidate_path)
				if candidate_cost < best_cost:
					best_cost = candidate_cost
					best_path = candidate_path

			return np.asarray(best_path, dtype=int)

		if strategy == "ordered":
			return np.arange(self.n_sensors, dtype=int)

		raise ValueError(
			f"Unknown sensor_sequence_strategy: {self.sensor_sequence_strategy}. "
			"Use one of: 'random', 'tsp', 'ordered'."
		)

	def _sample_stratified_sensor_locs(self):
		"""Sample near-uniform XY sensor locations with jittered stratification."""
		if self.n_sensors <= 0:
			return np.empty((0, 3), dtype=float)

		x_min, y_min = float(self.space_min[0]), float(self.space_min[1])
		x_max, y_max = float(self.space_max[0]), float(self.space_max[1])
		width = max(1e-9, x_max - x_min)
		height = max(1e-9, y_max - y_min)
		aspect = width / height

		n_cols = int(np.ceil(np.sqrt(self.n_sensors * aspect)))
		n_cols = max(1, n_cols)
		n_rows = int(np.ceil(self.n_sensors / n_cols))

		x_edges = np.linspace(x_min, x_max, n_cols + 1)
		y_edges = np.linspace(y_min, y_max, n_rows + 1)

		cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
		if len(cells) > self.n_sensors:
			perm = self.rng.permutation(len(cells))[: self.n_sensors]
			cells = [cells[idx] for idx in perm]

		sensor_locs = np.zeros((self.n_sensors, 3), dtype=float)
		for idx, (r, c) in enumerate(cells):
			x0, x1 = x_edges[c], x_edges[c + 1]
			y0, y1 = y_edges[r], y_edges[r + 1]
			# Keep points away from exact cell boundaries to reduce clustering.
			x = self.rng.uniform(x0 + 0.1 * (x1 - x0), x1 - 0.1 * (x1 - x0))
			y = self.rng.uniform(y0 + 0.1 * (y1 - y0), y1 - 0.1 * (y1 - y0))
			sensor_locs[idx] = [x, y, 0.0]

		return sensor_locs


	def _sample_clustered_sensor_locs(self):
		"""Sample XY sensor locations around random cluster centers with Gaussian spread."""
		if self.n_sensors <= 0:
			return np.empty((0, 3), dtype=float)

		x_min, y_min = float(self.space_min[0]), float(self.space_min[1])
		x_max, y_max = float(self.space_max[0]), float(self.space_max[1])
		width = max(1e-9, x_max - x_min)
		height = max(1e-9, y_max - y_min)

		n_clusters = max(1, min(int(self.n_sensor_clusters), int(self.n_sensors)))
		cluster_centers = np.zeros((n_clusters, 2), dtype=float)
		cluster_centers[:, 0] = self.rng.uniform(x_min + 0.0 * width, x_max - 0.0 * width, size=n_clusters)
		cluster_centers[:, 1] = self.rng.uniform(y_min + 0.0 * height, y_max - 0.0 * height, size=n_clusters)

		base, remainder = divmod(int(self.n_sensors), n_clusters)
		counts = np.full(n_clusters, base, dtype=int)
		counts[:remainder] += 1
		counts = counts[self.rng.permutation(n_clusters)]

		sigma_x = 0.04 * width
		sigma_y = 0.04 * height
		sensor_locs = np.zeros((self.n_sensors, 3), dtype=float)

		idx = 0
		for c_idx, count in enumerate(counts):
			if count <= 0:
				continue
			cx, cy = cluster_centers[c_idx]
			x = self.rng.normal(loc=cx, scale=sigma_x, size=int(count))
			y = self.rng.normal(loc=cy, scale=sigma_y, size=int(count))
			x = np.clip(x, x_min, x_max)
			y = np.clip(y, y_min, y_max)
			next_idx = idx + int(count)
			sensor_locs[idx:next_idx, 0] = x
			sensor_locs[idx:next_idx, 1] = y
			sensor_locs[idx:next_idx, 2] = 0.0
			idx = next_idx

		if idx < self.n_sensors:
			# Fill any leftover entries defensively if integer rounding paths change.
			left = self.n_sensors - idx
			sensor_locs[idx:, 0] = self.rng.uniform(x_min, x_max, size=left)
			sensor_locs[idx:, 1] = self.rng.uniform(y_min, y_max, size=left)
			sensor_locs[idx:, 2] = 0.0

		self.rng.shuffle(sensor_locs)
		return sensor_locs


	def init_sensor_locs(self):
		init_mode = str(getattr(self, "sensor_init_mode", "stratified")).lower()
		if init_mode == "stratified":
			sensor_locs = self._sample_stratified_sensor_locs()
		elif init_mode == "clustered":
			sensor_locs = self._sample_clustered_sensor_locs()
		else:
			raise ValueError(
				f"Unknown sensor_init_mode: {self.sensor_init_mode}. "
				"Use one of: 'stratified', 'clustered'."
			)
		self.sensor_locs = sensor_locs
		self.visited = np.zeros(self.n_sensors, dtype=bool)
		self.sensor_visit_sequence = self._compute_sensor_visit_sequence()


	def init_uav_locs(self, max_attempts: int = 5000):

		# Reset to configured parent template each spawn so episodes start from the same chain design.
		self.uav_parent = self.uav_parent_template.copy()

		uav_min = np.array([self.space_min[0], self.space_min[1], self.min_uav_height])
		uav_max = self.space_max.copy()
		uav_max[2] = self.min_uav_height  # spawn the whole chain at the minimum safe altitude

		uav_locs = np.zeros((self.n_uavs, 3))
		self.uav_connectivity = np.zeros((self.n_uavs, self.n_uavs), dtype=bool)

		# Sensors are on the ground, so avoid teaching every follower to first descend.
		head_pos = self.rng.uniform(uav_min, uav_max)
		uav_locs[0] = head_pos
		# uav_locs[0] = (uav_min + uav_max)/2

		# tree topology initialization: by default keep configured chain parents.
		# optional re-parenting can be enabled via self.reparent_on_init_failure.
		for i in range(1, self.n_uavs):
			configured_parent_idx = int(self.uav_parent[i])
			parent_idx = configured_parent_idx
			placed = False
			chosen_parent_idx = parent_idx

			other_candidates = [j for j in range(i) if j != parent_idx]
			if len(other_candidates) > 0:
				other_candidates = self.rng.permutation(other_candidates).tolist()
			if bool(getattr(self, "reparent_on_init_failure", False)):
				candidate_parents = [parent_idx] + other_candidates
			else:
				candidate_parents = [parent_idx]
			attempts_per_parent = max(1, max_attempts // max(1, len(candidate_parents)))

			def _is_valid_candidate(candidate_pos, candidate_parent_pos):
				if not (np.all(candidate_pos >= uav_min) and np.all(candidate_pos <= uav_max)):
					return False

				if hasattr(self, 'check_collision') and self.check_collision(candidate_pos):
					return False

				dist_to_parent = np.linalg.norm(candidate_pos - candidate_parent_pos)
				if dist_to_parent > 0.95 * self.u2u_range or dist_to_parent < self.u2u_safe_dist:
					return False

				# Keep deeper-hop followers outside direct head communication range to prevent star collapse at spawn.
				if bool(getattr(self, "enforce_nonchild_outside_head_range", True)) and configured_parent_idx != self.head_uav_idx:
					dist_to_head = np.linalg.norm(candidate_pos - uav_locs[self.head_uav_idx])
					if dist_to_head <= 0.98 * self.u2u_range:
						return False

				# Enforce collision-safe spacing with already placed UAVs.
				# Optionally enforce chain spawn by disallowing non-parent communication links at initialization.
				for j in range(i):
					dist = np.linalg.norm(candidate_pos - uav_locs[j])
					if dist < self.u2u_safe_dist:
						return False
					if bool(getattr(self, "enforce_chain_spawn", True)) and j != candidate_parent_idx and dist <= 0.95 * self.u2u_range:
						return False

				return True

			for candidate_parent_idx in candidate_parents:
				candidate_parent_pos = uav_locs[candidate_parent_idx]
				for _ in range(attempts_per_parent):
					r = self.rng.uniform(self.u2u_safe_dist, 0.95 * self.u2u_range)
					phi   = self.rng.uniform(0, np.pi)
					theta = self.rng.uniform(0, 2 * np.pi)
					offset = np.array([
						r * np.sin(phi) * np.cos(theta),
						r * np.sin(phi) * np.sin(theta),
						r * np.cos(phi),
					])
					candidate_pos = candidate_parent_pos + offset

					if not _is_valid_candidate(candidate_pos, candidate_parent_pos):
						continue

					uav_locs[i] = candidate_pos
					chosen_parent_idx = int(candidate_parent_idx)
					placed = True
					break
				if placed:
					break

			# Fallback: sample globally to avoid dead-ends caused by local geometry.
			if not placed:
				for _ in range(max_attempts):
					candidate_pos = self.rng.uniform(uav_min, uav_max)

					valid_parent_idxs = []
					for j in range(i):
						dist_to_j = np.linalg.norm(candidate_pos - uav_locs[j])
						if self.u2u_safe_dist <= dist_to_j <= 0.95 * self.u2u_range:
							if bool(getattr(self, "enforce_chain_spawn", True)):
								if j == configured_parent_idx:
									valid_parent_idxs.append(j)
							else:
								valid_parent_idxs.append(j)

					if len(valid_parent_idxs) == 0:
						continue

					if bool(getattr(self, "enforce_nonchild_outside_head_range", True)) and configured_parent_idx != self.head_uav_idx:
						dist_to_head = np.linalg.norm(candidate_pos - uav_locs[self.head_uav_idx])
						if dist_to_head <= 0.98 * self.u2u_range:
							continue

					if hasattr(self, 'check_collision') and self.check_collision(candidate_pos):
						continue

					too_close = any(
						np.linalg.norm(candidate_pos - uav_locs[j]) < self.u2u_safe_dist
						for j in range(i)
					)
					if too_close:
						continue

					uav_locs[i] = candidate_pos
					chosen_parent_idx = int(valid_parent_idxs[0])
					placed = True
					break

			if not placed:
				raise RuntimeError(
					f"init_uav_locs: could not place UAV {i} (parent={parent_idx}) after "
					f"{max_attempts} attempts. Consider increasing space size or reducing n_uavs."
				)

			self.uav_parent[i] = chosen_parent_idx
		# uav_locs[:,2] = uav_min[2]  # set all UAVs to minimum height
		self.uav_locs = uav_locs
				

	def update_uav_connectivity(self):
		"""
		Computes the graph connectivity set for all UAVs based on their current positions.
		Two UAVs are considered neighbors if their Euclidean distance is within u2u_range,
		subject to a hard per-UAV degree cap of self.max_uav_degree neighbors.

		Updates:
			self.uav_connectivity (np.ndarray): Boolean adjacency matrix of shape (n_uavs, n_uavs).
			self.uav_neighbors (list[set[int]]): List of neighbor sets, one per UAV.
				uav_neighbors[i] contains the indices of all UAVs within u2u_range of UAV i.

		Returns:
			list[set[int]]: The neighbor sets for all UAVs.
		"""
		self.uav_neighbors = [set() for _ in range(self.n_uavs)]
		self.uav_connectivity[:] = False
		degrees = np.zeros(self.n_uavs, dtype=int)

		candidate_edges = []
		for i in range(self.n_uavs):
			for j in range(i + 1, self.n_uavs):
				dist = np.linalg.norm(self.uav_locs[i] - self.uav_locs[j])
				# small tolerance: the filter holds links at the range limit, so rounding must not drop them
				if dist <= self.u2u_range * (1.0 + self.link_range_tolerance):
					is_parent_link = self.uav_parent[i] == j or self.uav_parent[j] == i
					candidate_edges.append((not is_parent_link, dist, i, j))

		# Greedily add edges, keeping the chain-parent links first (they carry the swarm's connectivity
		# back to the head), then the shortest remaining ones, while respecting the max degree.
		candidate_edges.sort(key=lambda x: (x[0], x[1]))
		for _, _, i, j in candidate_edges:
			if degrees[i] >= self.max_uav_degree or degrees[j] >= self.max_uav_degree:
				continue
			self.uav_neighbors[i].add(j)
			self.uav_neighbors[j].add(i)
			self.uav_connectivity[i, j] = True
			self.uav_connectivity[j, i] = True
			degrees[i] += 1
			degrees[j] += 1

		self.compute_hop_ids()

		
	def check_uav_connectivity(self):
		uav_connected = np.ones(self.n_uavs, dtype=bool)
		for uav_id in range(self.n_uavs):
			uav_nbrs = self.uav_neighbors[uav_id]
			upstream_present = 0
			for nbr_id in uav_nbrs:
				if self.hop_ids[nbr_id] < self.hop_ids[uav_id]:
					upstream_present += 1
					break
			if upstream_present == 0 and uav_id != self.head_uav_idx:
				uav_connected[uav_id] = False
		return uav_connected

	def compute_hop_ids(self):
		# BFS from head UAV; disconnected UAVs keep hop_id = inf
		hop_ids = np.full(self.n_uavs, np.inf)
		hop_ids[self.head_uav_idx] = 0
		queue = [self.head_uav_idx]
		while queue:
			current = queue.pop(0)
			for nbr in self.uav_neighbors[current]:
				if np.isinf(hop_ids[nbr]):
					hop_ids[nbr] = hop_ids[current] + 1
					queue.append(nbr)
		self.hop_ids = hop_ids
		return hop_ids

	def compute_follower_sensor_potential(self, sensor_idx, uav_id):
		potential = 0.0
		distance_to_sensor = np.maximum(0.0, np.linalg.norm(self.uav_locs[uav_id] - self.sensor_locs[sensor_idx]) - self.u2s_range)
		sensor_not_visited = not self.visited[sensor_idx]
		potential = sensor_not_visited * np.exp(-(distance_to_sensor/self.max_dist))
		return potential

	def compute_head_sensor_potential(self, sensor_idx, uav_id):
		potential = 0.0
		distance_to_sensor = np.maximum(0.0, np.linalg.norm(self.uav_locs[uav_id] - self.sensor_locs[sensor_idx]) - self.u2s_range)
		uav_altitude = self.uav_locs[uav_id][2] - self.min_uav_height
		sensor_not_visited = not self.visited[sensor_idx]
		if distance_to_sensor > 0 and uav_altitude > 0 and sensor_not_visited:
			P_max = self.compute_power(speed=self.head_v_max)
			time_to_sensor = distance_to_sensor / self.head_v_max
			energy_to_reach_sensor = P_max * time_to_sensor
			self.sensor_energy_to_reach[uav_id, sensor_idx] = energy_to_reach_sensor
			self.sensor_time_to_reach[uav_id,sensor_idx] = time_to_sensor
		if distance_to_sensor <= 0 and uav_altitude > 0:
			self.sensor_energy_to_reach[uav_id, sensor_idx] = 0.0
			self.sensor_time_to_reach[uav_id,sensor_idx] = 0.0
		potential = np.exp(-(self.sensor_energy_to_reach[uav_id, sensor_idx]/(self.remaining_energy[uav_id]+self.energy_eps)))

		return potential

	def total_potential(self, uav_id):
		total = 0.0
		for j in range(self.n_sensors):
			if self.visited[j]:
				continue
			if uav_id == self.head_uav_idx:
				potential = self.compute_head_sensor_potential(j, uav_id)
			elif uav_id != self.head_uav_idx:
				potential = self.compute_follower_sensor_potential(j, uav_id)
			total += potential
		# remaining_sensors = self.n_sensors - np.sum(self.visited)
		return total #/ remaining_sensors if remaining_sensors > 0 else 0.0

	def update_uav_positions(self):
		"""
		Update the positions of all UAVs based on their current control actions.
		"""
		sensor_potential_diffs = np.zeros(self.n_uavs)
		connectivity_potential_diffs = np.zeros(self.n_uavs)
		for uav_id in range(self.n_uavs):
			current_sensor_potential = self.total_potential(uav_id)
			current_connectivity_potential = self.compute_connectivity_potential(uav_id)
			self.uav_locs[uav_id] += self.uav_controls[uav_id] * self.dt
			new_sensor_potential = self.total_potential(uav_id)
			sensor_potential_diffs[uav_id] = new_sensor_potential - current_sensor_potential
			new_connectivity_potential = self.compute_connectivity_potential(uav_id)
			connectivity_potential_diffs[uav_id] = new_connectivity_potential - current_connectivity_potential
		return sensor_potential_diffs, connectivity_potential_diffs


	# accumulate collision penalty from all neighbors of the UAV
	def compute_collision_penalty(self, uav_id):
		penalty = 0.0
		sweet_dist = 2/3*self.u2u_safe_dist + 1/3*self.u2u_range
		uav_neighbors = sorted(self.uav_neighbors[uav_id])
		for nbr_id in uav_neighbors:
			if nbr_id > uav_id:
				distance = np.linalg.norm(self.uav_locs[uav_id] - self.uav_locs[nbr_id])
				if distance < sweet_dist and distance > 0.0:
					penalty += np.minimum((sweet_dist/self.u2u_safe_dist)**2, (sweet_dist/distance)**2)
		return penalty
	
	def compute_connectivity_potential(self, uav_id):
		potential = 0.0
		# sweet_dist = 2/3*self.u2u_range + 1/3*self.u2u_safe_dist
		sweet_dist = 0.6*self.u2u_range
		marginal_dist = 0.8*self.u2u_range
		uav_neighbors = list(self.uav_neighbors[uav_id])
		finite_neighbors = [nbr_id for nbr_id in uav_neighbors if np.isfinite(self.hop_ids[nbr_id])]
		upstream_neighbor = finite_neighbors[int(np.argmin([self.hop_ids[nbr_id] for nbr_id in finite_neighbors]))] if finite_neighbors else None
		distance = np.linalg.norm(self.uav_locs[uav_id] - self.uav_locs[upstream_neighbor]) if upstream_neighbor is not None else 0.0
		if distance > sweet_dist:
			potential += -((distance-sweet_dist)/(marginal_dist-sweet_dist))**3
		# elif distance >= self.u2u_range:
		# 	potential += -1.0
		return potential

	def compute_power(self, velocity=None, speed=None):
		"""
		Computes the instantaneous power consumption of the UAV for a given velocity.
		"""
		params = self.power_params
		P0 = params["P0"]
		P1 = params["P1"]
		v_tip = params["v_tip"]
		v0 = params["v0"]
		d0 = params["d0"]
		rho = params["rho"]
		s = params["s"]
		A = params["A"]
		if speed is None:
			speed = np.linalg.norm(velocity)
		
		blade_profile = P0 * (1.0 + 3.0 * speed**2 / v_tip**2)
		induced = P1 * (np.sqrt(1.0 + speed**4 / (4.0 * v0**4)) - speed**2 / (2.0 * v0**2))**0.5
		parasite = 0.5 * d0 * rho * s * A * speed**3
		
		total_power = blade_profile + induced + parasite
		return total_power

	def consumed_energy_step(self, velocity):
		"""
		Computes the energy consumed by the UAV for a given velocity step.
		"""
		total_power = self.compute_power(velocity=velocity)
		consumed_energy = total_power * self.dt

		return consumed_energy


	def collect_sensor_data(self, uav_id, sensor_idx):
		"""
		Collects sensor data for the UAV at its current location.
		Updates the visited state for each sensor.
		"""
		reward = 0.0
		if uav_id == self.head_uav_idx:
			return reward  # Head UAV does not collect sensor data
		
		sensor_pos = self.sensor_locs[sensor_idx]
		distance_to_sensor = np.linalg.norm(sensor_pos - self.uav_locs[uav_id]) - self.u2s_range
		not_visited = not self.visited[sensor_idx]
		if distance_to_sensor <= 0.0 and not_visited:
			reward += 1.0/self.n_sensors*self.sensor_collect_coeff[uav_id]
			self.visited[sensor_idx] = True
			self.uav_sensor_coverage[uav_id] += 1.0
			self.sensor_energy_to_reach[:, sensor_idx] = 0.0
			self.sensor_time_to_reach[:, sensor_idx] = 0.0

		return reward

	def collect_rewards(self, sensor_potential_diffs, connectivity_potential_diffs):

		reward = 0.0
		collision_penalty = 0.0
		connectivity_reward = 0.0
		sensor_potential_reward = 0.0
		sensor_collection_reward = 0.0

		for uav_id in range(self.n_uavs):
			# collision penalty
			collision_penalty -= self.collision_penalty_coeff * self.compute_collision_penalty(uav_id)
			# connectivity penalty
			connectivity_reward += self.connectivity_penalty_coeff * connectivity_potential_diffs[uav_id]
			# sensor potential
			sensor_potential_reward += self.potential_diff_coeff[uav_id] * sensor_potential_diffs[uav_id]
			# sensor collection reward
			# visible_sensors = self.visible_sensors[uav_id]
			for sensor_idx in range(self.n_sensors):
				if np.linalg.norm(self.uav_locs[uav_id] - self.sensor_locs[sensor_idx]) <= self.u2s_range:
					sensor_collection_reward += self.collect_sensor_data(uav_id, sensor_idx)
					break  # Only collect from one sensor per UAV per step
			# energy consumption penalty
			# to be added
			# latency penalty
			# to be added

		reward += (collision_penalty + connectivity_reward + sensor_potential_reward + sensor_collection_reward)

		# time penalty	
		reward -= self.time_penalty_coeff * self.dt/self.max_time

		self.collision_penalty = collision_penalty
		self.connectivity_reward = connectivity_reward
		self.sensor_potential_reward = sensor_potential_reward
		self.sensor_collection_reward = sensor_collection_reward

		return reward/(self.n_uavs-1) if self.n_uavs > 1 else reward

	def update_states(self):
		next_states = {
			"follower_state": {},
			"visible_sensor_counts": np.zeros(self.n_uavs, dtype=int),
		}

		# follower uav state
		for uav_id in range(self.n_uavs):
			if uav_id == self.head_uav_idx:
				continue
			
			uav_neighbors = list(self.uav_neighbors[uav_id])
			upstream_neighbor = None
			if uav_neighbors:
				finite_neighbors = [nbr_id for nbr_id in uav_neighbors if np.isfinite(self.hop_ids[nbr_id])]
				if finite_neighbors:
					upstream_neighbor = finite_neighbors[int(np.argmin([self.hop_ids[nbr_id] for nbr_id in finite_neighbors]))]
			if upstream_neighbor is None:
				# no in-range neighbour (link lost): fall back to the configured chain parent
				upstream_neighbor = int(self.uav_parent[uav_id])
			unvisited_sensor_locs = self.sensor_locs[~self.visited] if np.any(~self.visited) else np.empty((0, 3))
			next_states["follower_state"][uav_id] = {
				"sensor_pos": self._normalize_positions(unvisited_sensor_locs),
				"self_pos": self._normalize_positions(self.uav_locs[uav_id].reshape(1, 3)),
				"self_vel": self.uav_controls[uav_id].reshape(1, 3)/self.follower_v_max,
				"upstream_neighbor_pos": self._normalize_positions(self.uav_locs[upstream_neighbor].reshape(1, 3)),
				"upstream_neighbor_vel": self.uav_controls[upstream_neighbor].reshape(1, 3)/self.follower_v_max,
				"u2s_range": self.u2s_range/np.abs(np.max(self.space_span)),
				"u2u_range": self.u2u_range/np.abs(np.max(self.space_span)),
				"self_sensor_coverage": self.uav_sensor_coverage[uav_id] / self.n_sensors,
				"remaining_sensor_coverage": 1 - np.sum(self.visited) / self.n_sensors,
				"remaining_time": 1 - self.current_time / self.max_time,
			}

		return next_states


	def set_head_motion(self, mode="stationary", turn_rate=0.3, correlation_time=5.0, boundary_margin_frac=0.1, boundary_gain=1.0,
			n_head_clusters=5, head_max_turn_rate=0.5, head_arrival_radius=None, head_loop_tour=True):
		"""
		Configure the head UAV motion.

		mode: "stationary" keeps the head fixed; "random" moves it at head_v_max along a smooth random path;
			"cluster_tsp" clusters the sensors, orders the cluster centroids with a TSP tour and flies the head through them.
		turn_rate: stationary std (rad/s) of the random steering rate.
		correlation_time: time constant (s) of the steering noise; larger gives smoother, straighter paths.
		boundary_margin_frac / boundary_gain: soft steering away from the workspace walls within this fraction of the span.
		n_head_clusters: number of sensor clusters for "cluster_tsp".
		head_max_turn_rate: maximum heading rotation rate (rad/s) when flying to a cluster, which keeps the path smooth.
		head_arrival_radius: distance at which a cluster counts as visited (defaults to the u2s range).
		head_loop_tour: restart the tour after the last cluster; otherwise the head hovers there.
		"""
		if mode not in ("stationary", "random", "cluster_tsp"):
			raise ValueError(f"Unknown head motion mode: {mode}")
		self.head_motion_mode = mode
		self.head_turn_rate = float(turn_rate)
		self.head_correlation_time = float(correlation_time)
		self.head_boundary_margin_frac = float(boundary_margin_frac)
		self.head_boundary_gain = float(boundary_gain)
		self.n_head_clusters = max(1, int(n_head_clusters))
		self.head_max_turn_rate = float(head_max_turn_rate)
		self.head_arrival_radius = float(self.u2s_range if head_arrival_radius is None else head_arrival_radius)
		self.head_loop_tour = bool(head_loop_tour)
		self._head_tour = None
		self._reset_head_heading()

	def _reset_head_heading(self):
		heading = self.rng.normal(size=3)
		heading[2] *= 0.2  # start mostly horizontal
		self._head_heading = heading / np.linalg.norm(heading)
		self._head_steer = np.zeros(3)
		self._head_tour = None  # sensors may have been resampled, so replan lazily

	def _kmeans(self, points, k, n_iters=50):
		"""k-means with k-means++ seeding; returns the centroids of the non-empty clusters."""
		k = min(k, len(points))
		centers = [points[self.rng.randint(len(points))]]
		while len(centers) < k:
			d2 = np.min([np.sum((points - c) ** 2, axis=1) for c in centers], axis=0)
			if d2.sum() <= 0:
				break
			centers.append(points[self.rng.choice(len(points), p=d2 / d2.sum())])
		centers = np.array(centers)
		for _ in range(n_iters):
			labels = np.argmin(np.linalg.norm(points[:, None] - centers[None], axis=-1), axis=1)
			new_centers = np.array([points[labels == c].mean(axis=0) if np.any(labels == c) else centers[c] for c in range(len(centers))])
			if np.allclose(new_centers, centers):
				break
			centers = new_centers
		labels = np.argmin(np.linalg.norm(points[:, None] - centers[None], axis=-1), axis=1)
		return np.array([points[labels == c].mean(axis=0) for c in range(len(centers)) if np.any(labels == c)])

	def _tsp_open_path(self, start, points):
		"""Visit order of `points` (open path from `start`) via nearest-neighbour + 2-opt."""
		nodes = np.vstack([start[None], points])
		dist = np.linalg.norm(nodes[:, None] - nodes[None], axis=-1)
		n = len(nodes)
		path, remaining = [0], set(range(1, n))
		while remaining:
			nxt = min(remaining, key=lambda j: dist[path[-1], j])
			path.append(nxt)
			remaining.remove(nxt)
		improved = n > 3
		while improved:
			improved = False
			for i in range(1, n - 1):
				for j in range(i + 1, n):
					a, b = path[i - 1], path[i]
					c = path[j]
					d = path[j + 1] if j + 1 < n else None
					old = dist[a, b] + (dist[c, d] if d is not None else 0.0)
					new = dist[a, c] + (dist[b, d] if d is not None else 0.0)
					if new + 1e-12 < old:
						path[i : j + 1] = reversed(path[i : j + 1])
						improved = True
		return [p - 1 for p in path[1:]]

	def _plan_head_tour(self):
		"""Cluster the sensors and order the cluster centroids into a TSP tour starting from the head."""
		centroids = self._kmeans(np.asarray(self.sensor_locs, dtype=float), self.n_head_clusters)
		centroids[:, 2] = np.clip(centroids[:, 2], self.min_uav_height, self.space_max[2])
		order = self._tsp_open_path(self.uav_locs[self.head_uav_idx], centroids)
		self.head_tour_centroids = centroids[order]
		self._head_tour = {"waypoints": self.head_tour_centroids, "idx": 0}

	def _cluster_tsp_head_direction(self):
		if self._head_tour is None:
			self._plan_head_tour()
		tour = self._head_tour
		pos = self.uav_locs[self.head_uav_idx]
		waypoints = tour["waypoints"]
		for _ in range(len(waypoints)):
			if np.linalg.norm(waypoints[tour["idx"]] - pos) > self.head_arrival_radius:
				break
			if tour["idx"] + 1 < len(waypoints):
				tour["idx"] += 1
			elif self.head_loop_tour and len(waypoints) > 1:
				tour["idx"] = 0
			else:
				return np.zeros(3)  # tour finished: hover
		else:
			return np.zeros(3)  # every waypoint is already within the arrival radius
		to_target = waypoints[tour["idx"]] - pos
		desired = to_target / max(np.linalg.norm(to_target), 1e-9)

		# rotate the heading toward the target by at most head_max_turn_rate * dt so the direction stays smooth
		heading = self._head_heading
		angle = np.arccos(np.clip(np.dot(heading, desired), -1.0, 1.0))
		max_angle = self.head_max_turn_rate * self.dt
		if angle <= max_angle:
			new_heading = desired
		else:
			perp = desired - np.dot(desired, heading) * heading
			norm = np.linalg.norm(perp)
			if norm < 1e-9:  # target directly behind: pick any perpendicular turn axis
				perp = np.cross(heading, [1.0, 0.0, 0.0]) if abs(heading[0]) < 0.9 else np.cross(heading, [0.0, 1.0, 0.0])
				norm = np.linalg.norm(perp)
			new_heading = np.cos(max_angle) * heading + np.sin(max_angle) * perp / norm
		self._head_heading = new_heading / np.linalg.norm(new_heading)
		return self._head_heading.copy()

	def _next_head_direction(self):
		"""
		Unit direction for the head UAV. In random mode the heading is rotated by a smooth
		Ornstein-Uhlenbeck steering rate (plus a soft wall-avoidance term), so it is continuous in time.
		"""
		mode = getattr(self, "head_motion_mode", "stationary")
		if mode == "stationary":
			return np.zeros(3)
		if not hasattr(self, "_head_heading"):
			self._reset_head_heading()
		if mode == "cluster_tsp":
			return self._cluster_tsp_head_direction()

		tau = self.head_correlation_time
		noise = self.rng.normal(size=3)
		self._head_steer += -self._head_steer / tau * self.dt + self.head_turn_rate * np.sqrt(2.0 * self.dt / tau) * noise
		heading = self._head_heading
		steer = self._head_steer - np.dot(self._head_steer, heading) * heading

		lo = self.space_min.copy()
		lo[2] = self.min_uav_height
		margin = self.head_boundary_margin_frac * (self.space_max - lo)
		pos = self.uav_locs[self.head_uav_idx]
		near_lo = np.clip((margin - (pos - lo)) / margin, 0.0, 1.0)
		near_hi = np.clip((margin - (self.space_max - pos)) / margin, 0.0, 1.0)
		repulsion = self.head_boundary_gain * (near_lo**2 - near_hi**2)

		new_heading = heading + (steer + repulsion) * self.dt
		self._head_heading = new_heading / np.linalg.norm(new_heading)
		return self._head_heading.copy()

	def step(self, env_action):
		"""Apply follower direction actions in ascending UAV-index order, excluding the head."""
		follower_ids = [uav_id for uav_id in range(self.n_uavs) if uav_id != self.head_uav_idx]
		env_action = np.asarray(env_action, dtype=float)
		if env_action.shape != (len(follower_ids), 3):
			raise ValueError(
				f"Expected env_action with shape {(len(follower_ids), 3)}, got {env_action.shape}"
			)
		# nominal unit directions; the head direction comes from the head motion mode (zero if stationary)
		policy_dirs = np.zeros((self.n_uavs, 3))
		policy_dirs[follower_ids] = env_action
		policy_dirs[self.head_uav_idx] = self._next_head_direction()
		max_speeds = self.follower_v_max * np.ones(self.n_uavs)
		max_speeds[self.head_uav_idx] = self.head_v_max

		# Update connectivity based on frequency setting (default: every step)
		if self.update_counter % self.connectivity_update_freq == 0:
			self.update_uav_connectivity()
		self.update_counter += 1
		uav_connected = self.check_uav_connectivity()
		if not uav_connected.all():
			self.connectivity_violations += 1
		
		# filter the policy control to preserve connectivity
		result = mcf.multi_agent_connectivity_filter(self.uav_locs, policy_dirs, max_speeds, self.dt, self.u2u_range * self.filter_range_margin, self.uav_connectivity)
		filtered_dirs = result.x.reshape(self.n_uavs, 3)
		if not result.success:
			self.filter_failures += 1
			follower_norms = np.linalg.norm(filtered_dirs[follower_ids], axis=1)
			if not (np.all(np.isfinite(filtered_dirs)) and np.allclose(follower_norms, 1.0, atol=1e-2)):
				# unusable solution: reuse the last valid filtered directions (or the nominal ones on the first step)
				fallback = getattr(self, "_last_filtered_dirs", None)
				if fallback is None:
					fallback = policy_dirs.copy()
				filtered_dirs = fallback.copy()
			if self.filter_failures == 1:
				print(f"[swarm_env] connectivity filter did not converge ({result.message}); using fallback directions")
		filtered_dirs[self.head_uav_idx] = policy_dirs[self.head_uav_idx]
		self._last_filtered_dirs = filtered_dirs.copy()
		self.uav_controls = filtered_dirs * max_speeds[:, None]

		# penalty for deviating from the policy direction
		filtered_control_penalty = 1/(self.n_uavs-1) * np.sum(np.linalg.norm(filtered_dirs[1:,:2] - policy_dirs[1:,:2], axis=1)) * self.cbf_control_penalty_coeff

		# apply the filtered velocities to update UAV positions
		sensor_potential_diffs, connectivity_potential_diffs = self.update_uav_positions()

		reward = self.collect_rewards(sensor_potential_diffs, connectivity_potential_diffs)
		reward -= filtered_control_penalty
		self.cumulative_reward += reward

		# update current time
		self.current_time += self.dt

		# update the states
		next_state = self.update_states()

		# terminate when all sensors have been visited, the head's energy is spent, or time is up
		if all(self.visited) or self.remaining_energy[self.head_uav_idx] <= 0.0 or self.current_time >= self.max_time:
			terminated = True
		else:
			terminated = False
		
		if self.connectivity_violations > 100:
			truncated = True
		else:
			truncated = False

		info = {
			"reward_total": float(reward),
			"reward_collision_penalty": float(self.collision_penalty),
			"reward_connectivity_penalty": float(self.connectivity_penalty),
			"reward_sensor_potential": float(self.sensor_potential_reward),
			"reward_sensor_collection": float(self.sensor_collection_reward),
			"visited_fraction": float(np.mean(self.visited.astype(np.float32))),
		}

		print(f"time: {self.current_time:.4f}\tcumulative_reward: {self.cumulative_reward:.4f}")

		return next_state, reward, terminated, truncated, info

	def reset_uav_states(self):
		"""
		Resets the UAV states to their initial values.
		"""
		# Preserve configured chain count across resets.
		self._build_chain_topology(self.n_chains)
		try:
			self.init_uav_locs()
		except RuntimeError:
			# Relax the non-child-outside-head-range constraint and retry; the
			# annulus of valid placements can be empty for deep chain topologies.
			_prev = getattr(self, "enforce_nonchild_outside_head_range", True)
			self.enforce_nonchild_outside_head_range = False
			try:
				self.init_uav_locs()
			finally:
				self.enforce_nonchild_outside_head_range = _prev
		self.uav_controls = np.zeros((self.n_uavs, 3))
		self.remaining_energy = self.max_energy * np.ones(self.n_uavs)
		self.current_time = 0.0
		self.cumulative_reward = 0.0
		if hasattr(self, "_head_heading"):
			self._reset_head_heading()
		if hasattr(self, "_head_perimeter_target_idx"):
			delattr(self, "_head_perimeter_target_idx")
		self.visited = np.zeros(self.n_sensors, dtype=bool)
		self.uav_sensor_coverage = np.zeros(self.n_uavs, dtype=float)
		self.update_counter = 0
		self.filter_failures = 0
		self._last_filtered_dirs = None
		self.connectivity_violations = 0
		self.update_uav_connectivity()
		for uav_id in range(self.n_uavs):
			# self.update_visible_sensors(uav_id)
			self.total_potential(uav_id)

		next_state = self.update_states()
		return next_state


	def reset_env(self, n_sensors):
		"""
		Resets the environment to its initial state.
		"""
		self.n_sensors = n_sensors
		self.init_sensor_locs()
		next_state = self.reset_uav_states()
		return next_state


def simulate(env, model_filename, max_steps=None, use_mean_direction=True, direction_temperature=1.0, reset=True, device="cpu"):
	"""
	Roll out the swarm with a trained follower policy loaded from a checkpoint file.

	model_filename: checkpoint saved by training_utils.save_checkpoint (needs "follower_actor_state_dict").
	max_steps: step limit; defaults to the steps remaining until max_time.
	use_mean_direction: act with the policy mean direction instead of sampling.
	reset: start from a freshly reset environment; otherwise continue from the current state.

	Returns a dict compatible with rollout_visualization (uav_locs, times, reward_per_step, visited_fraction,
	visited_mask, connectivity, plus static scene info) and the final terminated/truncated flags.
	"""
	# imported here because these modules import swarm_env
	import follower_actor as fa
	from mappo_env_interface import obs_adapter

	checkpoint = torch.load(model_filename, map_location=device)
	position_dim = 3
	actor = fa.Actor(
		sensor_feature_dim=2 * position_dim + 1,
		upstream_feature_dim=4 * position_dim + 2,
	)
	actor.load_state_dict(checkpoint["follower_actor_state_dict"])
	actor.to(device).eval()

	raw_obs = env.reset_env(env.n_sensors) if reset else env.update_states()
	if max_steps is None:
		max_steps = int(np.ceil((env.max_time - env.current_time) / env.dt))

	trajectory = {
		"uav_locs": [env.uav_locs.copy()],
		"times": [env.current_time],
		"reward_per_step": [],
		"visited_fraction": [float(np.mean(env.visited))],
		"visited_mask": [np.asarray(env.visited, dtype=bool).copy()],
		"connectivity": [env.uav_connectivity.copy()],
		"sensor_locs": np.asarray(env.sensor_locs, dtype=np.float32).copy(),
		"base_station": np.asarray(env.bs_loc, dtype=np.float32).copy(),
		"u2u_range": float(env.u2u_range),
		"u2u_safe_dist": float(env.u2u_safe_dist),
		"space_min": np.asarray(env.space_min, dtype=np.float32).copy(),
		"space_max": np.asarray(env.space_max, dtype=np.float32).copy(),
		"head_uav_idx": int(env.head_uav_idx),
		"follower_uav_ids": [i for i in range(env.n_uavs) if i != env.head_uav_idx],
	}
	terminated = truncated = False

	for _ in range(max_steps):
		follower_obs = {
			k: v.to(device) for k, v in obs_adapter(env, raw_obs)["follower_actor_obs"].items()
		}
		b, f, n, d = follower_obs["sensor_pos"].shape
		flat = lambda key, dim: follower_obs[key].reshape(b * f, dim)
		with torch.no_grad():
			act_out = actor.act(
				sensor_pos=follower_obs["sensor_pos"].reshape(b * f, n, d),
				agent_pos=flat("agent_pos", d),
				agent_vel=flat("agent_vel", d),
				self_sensor_coverage=flat("self_sensor_coverage", 1),
				remaining_sensor_coverage=flat("remaining_sensor_coverage", 1),
				remaining_time=flat("remaining_time", 1),
				upstream_neighbor_pos=flat("upstream_neighbor_pos", d),
				upstream_neighbor_vel=flat("upstream_neighbor_vel", d),
				u2s_range=follower_obs["u2s_range"].expand(b * f, 1),
				u2u_range=follower_obs["u2u_range"].expand(b * f, 1),
				sensor_mask=follower_obs["sensor_mask"].reshape(b * f, n),
				direction_temperature=direction_temperature,
				use_mean_direction=use_mean_direction,
			)
		env_action = act_out["direction_action"].cpu().numpy()

		raw_obs, reward, terminated, truncated, _ = env.step(env_action)

		trajectory["uav_locs"].append(env.uav_locs.copy())
		trajectory["times"].append(env.current_time)
		trajectory["reward_per_step"].append(float(reward))
		trajectory["visited_fraction"].append(float(np.mean(env.visited)))
		trajectory["visited_mask"].append(np.asarray(env.visited, dtype=bool).copy())
		trajectory["connectivity"].append(env.uav_connectivity.copy())
		if terminated or truncated:
			break

	for key in ("uav_locs", "times", "reward_per_step", "visited_fraction", "visited_mask", "connectivity"):
		trajectory[key] = np.asarray(trajectory[key])
	trajectory["terminated"] = terminated
	trajectory["truncated"] = truncated
	return trajectory
