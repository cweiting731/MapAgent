"""Starter code for the Unknown-Map Exploration Agent assignment."""

from collections import deque

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
        self.internalMap = {}
        self.visited = set()

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
        self.internalMap = {self.start_position: "FREE"}
        self.visited = set()

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
        self.internalMap[position] = "FREE"

        # 更新 internal map 地圖資訊
        for action, (row, column) in MOVES.items():
            neighbor = (position[0] + row, position[1] + column)
            self.internalMap[neighbor] = neighbors[action]

        # 找尋鄰居有 GOAL 直接結束
        for action in MOVES.keys():
            if neighbors[action] == "GOAL":
                return action

        # A frontier is a cell known to be FREE but not yet visited.  BFS over
        # the already visited, known-free region finds the closest frontier;
        # therefore the returned move is the first step of a shortest known
        # route to new information.
        action = self._next_action_to_nearest_frontier(position)
        if action is not None:
            return action

        # A valid map has a reachable goal, so this fallback is not expected
        # in normal grading.  Prefer any known-free action to avoid a wall
        # collision if an invalid/disconnected test map is supplied.
        for action in MOVES:
            if neighbors[action] == "FREE":
                return action
        return next(iter(MOVES))

    def _next_action_to_nearest_frontier(self, start):
        """Find the first action on a shortest known path to a frontier.

        BFS only traverses visited cells.  An unvisited FREE cell is accepted
        as an endpoint, never as an intermediate node, so every planned move
        is through space already confirmed to be traversable.
        """
        queue = deque([start])
        came_from = {start: (None, None)}

        while queue:
            current = queue.popleft()
            for action, (row, column) in MOVES.items():
                neighbor = (current[0] + row, current[1] + column)
                if self.internalMap.get(neighbor) != "FREE":
                    continue

                if neighbor not in self.visited:
                    # Reconstruct only the first action, avoiding an
                    # unnecessary persistent plan after new observations.
                    route_actions = [action]
                    cursor = current
                    while cursor != start:
                        parent, parent_action = came_from[cursor]
                        route_actions.append(parent_action)
                        cursor = parent
                    return route_actions[-1]

                if neighbor not in came_from:
                    came_from[neighbor] = (current, action)
                    queue.append(neighbor)

        return None
