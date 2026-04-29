import math
import re
from typing import List, Tuple, Optional
import numpy as np


class SpatialMemory:
    """Running spatial map built across steps within a single navigation episode.

    Maintains a 2D occupancy grid in the habitat x-z plane and tracks landmark
    discovery events, so the agent can condition on where it has already been
    rather than only on raw image history.
    """

    # Simple colour + object vocabulary for landmark extraction
    _COLOURS = (
        "red|blue|green|brown|white|black|gray|grey|yellow|orange|pink|purple"
        "|wooden|dark|light|beige|tan|teal|maroon|navy"
    )
    _OBJECTS = (
        "chair|table|desk|couch|sofa|bed|door|wall|shelf|cabinet|television|tv"
        "|plant|lamp|counter|staircase|stairs|hallway|kitchen|bathroom|bedroom"
        "|living room|dining room|window|pillar|column|fireplace|bookcase"
        "|refrigerator|sink|toilet|bathtub|mirror|rug|carpet|painting|picture"
    )

    def __init__(self, grid_resolution: float = 0.25, grid_size: int = 200):
        """
        Args:
            grid_resolution: metres per grid cell.
            grid_size: number of cells per side (grid is grid_size × grid_size).
        """
        self.resolution = grid_resolution
        self.grid_size = grid_size
        # World-space origin maps to the centre cell.
        self.origin = np.array([grid_size // 2, grid_size // 2], dtype=int)
        # Visitation count per cell (clipped to 255).
        self.visited_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        # Full position history in world coordinates (x, y, z).
        self.positions: List[np.ndarray] = []
        # Landmark phrases extracted from the instruction at init time.
        self.landmarks: List[str] = []
        # Landmark phrases that have been confirmed as visited.
        self.visited_landmarks: set = set()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @classmethod
    def extract_landmarks(cls, instruction: str) -> List[str]:
        """Return colour+noun or bare-noun landmark phrases found in the instruction."""
        pattern = rf"\b(?:(?:{cls._COLOURS})\s+)?(?:{cls._OBJECTS})s?\b"
        found = re.findall(pattern, instruction, re.IGNORECASE)
        return list({lm.lower().strip() for lm in found})

    def reset(self):
        """Clear accumulated state (call between rollout episodes on same env)."""
        self.visited_grid[:] = 0
        self.positions.clear()
        self.visited_landmarks.clear()

    def update(self, position: np.ndarray):
        """Record the agent's current 3-D world position (x, y, z)."""
        self.positions.append(position.copy())
        row, col = self._world_to_grid(position)
        self.visited_grid[row, col] = min(int(self.visited_grid[row, col]) + 1, 255)

    def check_landmark_proximity(
        self,
        landmark_candidates: List[Tuple[str, np.ndarray]],
        threshold: float = 2.0,
    ) -> List[str]:
        """Return newly-discovered landmark names within *threshold* metres.

        Args:
            landmark_candidates: list of (name, world_pos) pairs from GT waypoints.
            threshold: discovery radius in metres (x-z plane distance).

        Returns:
            List of landmark names newly added to visited_landmarks.
        """
        if not self.positions:
            return []
        current_xz = self.positions[-1][[0, 2]]
        newly_found: List[str] = []
        for name, lm_pos in landmark_candidates:
            if name in self.visited_landmarks:
                continue
            dist = float(np.linalg.norm(current_xz - np.array(lm_pos)[[0, 2]]))
            if dist <= threshold:
                self.visited_landmarks.add(name)
                newly_found.append(name)
        return newly_found

    def get_summary(self) -> str:
        """Return a compact text description of spatial memory for prompt injection."""
        n_steps = len(self.positions)
        if n_steps == 0:
            return ""

        n_visited_cells = int(self.visited_grid.astype(bool).sum())
        area_m2 = n_visited_cells * (self.resolution ** 2)

        heading_str = ""
        if n_steps >= 2:
            delta = self.positions[-1][[0, 2]] - self.positions[-2][[0, 2]]
            norm = float(np.linalg.norm(delta))
            if norm > 1e-3:
                angle = float(np.degrees(np.arctan2(float(delta[0]), float(delta[1]))))
                heading_str = f", heading ~{angle:.0f}°"

        lm_str = ""
        if self.visited_landmarks:
            lm_str = f" Confirmed landmarks: {', '.join(sorted(self.visited_landmarks))}."

        return (
            f"[Memory] {n_steps} steps, ~{area_m2:.1f}² explored"
            f"{heading_str}.{lm_str}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        col = int(pos[0] / self.resolution) + self.origin[0]
        row = int(pos[2] / self.resolution) + self.origin[1]
        col = int(np.clip(col, 0, self.grid_size - 1))
        row = int(np.clip(row, 0, self.grid_size - 1))
        return row, col
