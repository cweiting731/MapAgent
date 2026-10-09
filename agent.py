"""Starter code for the Unknown-Map Exploration Agent assignment."""

MOVES = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

class Agent:
    def __init__(self):
        """
        Initialize the Agent.

        You can declare any required instance variables here,
        but the state for each new map should be reinitialized in reset().
        """
        # ``reset`` owns map-specific state.  Initialising these attributes
        # here as well makes an accidentally early call to ``act`` easier to
        # diagnose and keeps the object in a consistent state.
        self.rows = 0
        self.cols = 0
        self.start_position = None
        self.internal_map = {}
        self.visited = set()
        self.path = []

    def reset(self, rows, cols, start_position):
        """
        Called once before each new map.

        Args:
            rows: 地圖列數
            cols: 地圖欄數
            start_position: 起點座標 (row, col)
        """
        self.rows = rows
        self.cols = cols
        self.start_position = tuple(start_position)
        # Maps a coordinate to its observed type ("FREE", "WALL", or
        # "GOAL").  Only cells adjacent to a visited cell can be known.
        self.internal_map = {self.start_position: "FREE"}
        self.visited = set()
        # The current DFS branch, from the start through the agent's current
        # position.  It gives a safe route for backtracking from dead ends.
        self.path = []

    def act(self, percept):
        """
        Choose and return one of: UP, DOWN, LEFT, RIGHT.

        Args:
            percept:
                {
                    "position": (row, col),
                    "neighbors": {
                        "UP": "FREE" | "WALL" | "GOAL",
                        "DOWN": "FREE" | "WALL" | "GOAL",
                        "LEFT": "FREE" | "WALL" | "GOAL",
                        "RIGHT": "FREE" | "WALL" | "GOAL",
                    },
                }
        
        Returns:
            One of: UP, DOWN, LEFT, RIGHT
        """
        position = tuple(percept["position"])
        neighbors = percept["neighbors"]

        self.visited.add(position)
        self.internal_map[position] = "FREE"

        # Record every observation.  This lets the agent distinguish a wall
        # from an unvisited, traversable neighbour without probing walls.
        for action, (dr, dc) in MOVES.items():
            neighbor = (position[0] + dr, position[1] + dc)
            self.internal_map[neighbor] = neighbors[action]

        # A goal is always reached as soon as it becomes visible.  Checking
        # this before exploration avoids needless detours beside the goal.
        for action in MOVES:
            if neighbors[action] == "GOAL":
                return action

        # Maintain the DFS branch.  Under the supplied deterministic
        # environment the position is normally its last element.  The small
        # recovery path also makes the state safe if this agent is inspected
        # or reused with an externally repositioned environment.
        if not self.path:
            self.path = [position]
        elif self.path[-1] != position:
            if position in self.path:
                self.path = self.path[: self.path.index(position) + 1]
            else:
                self.path = [position]

        # Descend into an unvisited open cell.  Since its state was reported
        # as FREE, this action cannot collide with a wall.
        for action, (dr, dc) in MOVES.items():
            neighbor = (position[0] + dr, position[1] + dc)
            if neighbors[action] == "FREE" and neighbor not in self.visited:
                self.path.append(neighbor)
                return action

        # All children of this DFS node have been explored.  Return along the
        # recorded branch; this is also a known-free move and cannot collide.
        if len(self.path) > 1:
            parent = self.path[-2]
            self.path.pop()
            return self._action_to(position, parent)

        # A valid map has a reachable goal, so this fallback is not expected
        # in normal grading.  Prefer any known-free action to avoid a wall
        # collision if an invalid/disconnected test map is supplied.
        for action in MOVES:
            if neighbors[action] == "FREE":
                return action
        return next(iter(MOVES))

    @staticmethod
    def _action_to(source, target):
        """Return the action that moves from adjacent ``source`` to ``target``."""
        delta = (target[0] - source[0], target[1] - source[1])
        for action, move in MOVES.items():
            if move == delta:
                return action
        raise ValueError(f"Cells are not adjacent: {source!r}, {target!r}")
