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

        # 找尋鄰居有 FREE 並且還沒走過的格子
        action = self._next_action_to_nearest_frontier(position)
        if action is not None:
            return action

        # 防呆
        for action in MOVES:
            if neighbors[action] == "FREE":
                return action
        return next(iter(MOVES))

    def _next_action_to_nearest_frontier(self, start):
        queue = deque([start])
        cameFrom = {start: (None, None)}

        while queue:
            current = queue.popleft()
            for action, (row, column) in MOVES.items():
                neighbor = (current[0] + row, current[1] + column)

                # 鄰居不可走
                if self.internalMap.get(neighbor) != "FREE":
                    continue

                # 可走 還沒走過 找到 Frontier
                if neighbor not in self.visited:
                    # 記錄走到 Frontier 的最後一步
                    routeActions = [action]
                    cursor = current

                    # 沿著 BFS 的父節點一路回推到當前點
                    while cursor != start:
                        parent, parentAction = cameFrom[cursor]
                        routeActions.append(parentAction)
                        cursor = parent
                    # 回傳從當前點出發的第一步
                    return routeActions[-1]

                # 鄰居已走過 並且本次 BFS 尚未搜尋過 加入佇列繼續往外擴展
                if neighbor not in cameFrom:
                    cameFrom[neighbor] = (current, action)
                    queue.append(neighbor)

        # 所有已走訪區域都搜尋完 沒有 Frontier 可以走
        return None
