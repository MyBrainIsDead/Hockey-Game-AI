# importing our dependency
import heapq
from heapq import *
from collections import deque
from logging import exception
from operator import index
from queue import Queue
from typing import Tuple
import numpy as np


# Our game environment state
class GameState:
    def __init__(self, player_state, board, balls, init_trap, traps, trap_state, goals, criterion='criterion',
                 goaled_ball_positions=[], g_cost=0, h_cost=float('inf')):
        self.player_state = player_state
        self.board = board
        self.balls = balls
        self.init_trap = init_trap
        self.traps = traps
        self.trap_state = trap_state
        self.goals = goals
        self.criterion = criterion
        self.goaled_ball_positions = goaled_ball_positions
        self.g_cost = g_cost
        self.h_cost = h_cost

    @property
    def f_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return getattr(self, self.criterion) < getattr(other, self.criterion)

    def __eq__(self, other):
        return self.player_state == other.player_state and \
            self.balls == other.balls and \
            self.goaled_ball_positions == other.goaled_ball_positions and \
            self.trap_state == other.trap_state

    def __hash__(self):
        return hash((self.player_state,
                     tuple(self.balls),
                     tuple(self.goaled_ball_positions), self.trap_state))

    def goal_test(self):
        return len(self.goaled_ball_positions) == len(self.goals)


# Our game enviroment node
class GameNode:
    def __init__(self, state, parent=None, action=None, depth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth

    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


# function to update traps based on the traps_state(trap_loc_index)
def update_traps(trap_loc_index, init_trap):
    new_traps_loc = []
    if trap_loc_index == 1:
        trap_loc_index = 2
    elif trap_loc_index == 2:
        trap_loc_index = 3
    elif trap_loc_index == 3:
        trap_loc_index = 4
    elif trap_loc_index == 4:
        trap_loc_index = 1

    for trap in init_trap:
        x, y = trap

        if trap_loc_index == 1:
            new_traps_loc.append((x, y))

        elif trap_loc_index == 2:
            new_traps_loc.append((x, y - 1))

        elif trap_loc_index == 3:
            new_traps_loc.append((x + 1, y - 1))

        elif trap_loc_index == 4:
            new_traps_loc.append((x + 1, y))
    return new_traps_loc, trap_loc_index


# function to get danger zones ie all possible trap locations
'''def get_danger_zone(traps):
    danger_zone = []
    for trap in traps:
        x, y = trap
        for direction in ((0, 0), (0, -1), (1, -1), (1, 0)):
            danger_zone.append((x + direction[0], y + direction[1]))

    return danger_zone'''


# function to get the minimum distance from one point to another point, the distance is in manhatan
def minimum_distance(src, dest, initial_state):
    # remake our board such that every location x,y = inf
    dist = [[float('inf')] * len(initial_state.board[0]) for _ in range(len(initial_state.board))]
    dist[src[0]][src[1]] = 0

    pq = [(dist[src[0]][src[1]], src[0], src[1])]  # (distance, x, y)

    while pq:
        d, x, y = heappop(pq)
        if x == dest[0] and y == dest[1]:
            cost = 0
            if initial_state.board[x][y].isdigit():
                cost = int(initial_state.board[x][y])

            elif len(initial_state.board[x][y]) == 2:
                cost = int(initial_state.board[x][y][0])

            return d - cost
        # will travel src to find all path and there costs
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(initial_state.board) and 0 <= ny < len(initial_state.board[0]):
                cost = 0
                if initial_state.board[nx][ny].isdigit():
                    cost = int(initial_state.board[nx][ny])

                elif len(initial_state.board[nx][ny]) == 2:
                    cost = int(initial_state.board[nx][ny][0])

                new_dist = d + cost
                if new_dist < dist[nx][ny]:
                    dist[nx][ny] = new_dist
                    heappush(pq, (new_dist, nx, ny))

    return float('inf')  # If no path exists


def heuristic(state, initial_state):
    total_ball_to_goal_cost = 0
    for ball in state.balls:
        if ball in state.goaled_ball_positions: continue
        min_distance = float('inf')
        for goal in state.goals:
            distance = minimum_distance(ball, goal, initial_state)
            min_distance = min(min_distance, distance)
        total_ball_to_goal_cost += min_distance

    max_player_cost = 0
    for ball in state.balls:
        distance = minimum_distance(state.player_state, ball, initial_state)
        max_player_cost = max(max_player_cost, distance)
    return total_ball_to_goal_cost + max_player_cost


def get_successors(state):
    successors = []
    trap_loc_updater = update_traps(state.trap_state, state.init_trap)
    x, y = state.player_state
    new_trap_loc = trap_loc_updater[0]
    new_trap_state = trap_loc_updater[1]
    #danger_zone = get_danger_zone(state.init_trap)

    for movement in [((0, -1), 'left'), ((1, 0), 'down'), ((0, 1), 'right'), ((-1, 0), 'up')]:
        new_cost_g = state.g_cost
        is_moving_allowed = False
        dx, dy = movement[0]
        move = movement[1]
        new_x, new_y = x + dx, y + dy
        # checking to see if the player can move
        if 0 <= new_x < len(state.board) and 0 <= new_y < len(state.board[0]) and (new_x, new_y) not in new_trap_loc and (new_x, new_y) not in state.goaled_ball_positions:
            is_moving_allowed = True

            # getting the cost of the loc new_x,new_y
            if state.board[new_x][new_y].isdigit():
                new_cost_g += int(state.board[new_x][new_y])

            elif len(state.board[new_x][new_y]) == 2:
                new_cost_g += int(state.board[new_x][new_y][0])

            new_ball_positions = state.balls.copy()
            new_goaled_ball = state.goaled_ball_positions.copy()

            for i, ball in enumerate(state.balls):
                bx, by = ball
                # checking if the player is around the ball
                if ball == (new_x, new_y):
                    new_bx, new_by = bx + dx, by + dy

                    # checking if the ball can move
                    if 0 <= new_bx < len(state.board) and 0 <= new_by < len(state.board[0]) and (new_bx, new_by) not in new_trap_loc and (new_bx, new_by) not in state.balls:
                        new_ball_positions[i] = (new_bx, new_by)

                    # checking if the ball has reached the goal
                    if (new_bx, new_by) in state.goals:
                        new_goaled_ball.append((new_bx, new_by))

        if is_moving_allowed:
            successor = GameState((new_x, new_y), state.board, new_ball_positions, state.init_trap, new_trap_loc,
                                  new_trap_state, state.goals, state.criterion, new_goaled_ball, new_cost_g,
                                  state.h_cost)
            successors.append((successor, move))

    successors.reverse()
    return successors


def bfs(initial_state):
    frontier = deque()
    frontier.append(GameNode(state=initial_state))
    explored = set()

    while frontier:
        current_node = frontier.popleft()
        current_state = current_node.state
        if current_state.goal_test():
            return current_node

        explored.add(current_state)
        for successor, action in get_successors(current_state):
            successor_node = GameNode(successor, current_node, action, current_node.depth + 1)
            if successor not in explored and successor_node not in frontier:
                frontier.append(successor_node)


def dfs(initial_state):
    frontier = deque()
    frontier.append(GameNode(state=initial_state))
    explored = set()

    while frontier:
        current_node = frontier.pop()
        current_state = current_node.state

        if current_state.goal_test():
            return current_node

        explored.add(current_state)
        for successor, action in get_successors(current_state):
            successor_node = GameNode(successor, current_node, action, current_node.depth + 1)
            if successor not in explored and successor_node not in frontier:
                frontier.append(successor_node)


def ucs(initial_state):
    initial_state.criterion = 'g_cost'
    frontier = []
    heappush(frontier, GameNode(initial_state))
    explored = {}

    while frontier:
        current_node = heappop(frontier)
        current_state = current_node.state

        if current_state.goal_test():
            return current_node

        if current_state  in explored and current_state.g_cost >= explored[current_state]:continue

        explored[current_state] = current_state.g_cost

        for successor_state, action in get_successors(current_state):
            successor_node = GameNode(successor_state, current_node, action, current_node.depth + 1)

            if successor_state not in explored or successor_state.g_cost < explored[successor_state]:
                heappush(frontier, successor_node)

    return None



def ids(initial_state):
    def dls(node, depth_limit, visited):
        if node.state.goal_test():
            return node  # Goal found
        elif depth_limit == 0:
            return None

        else:
            visited.add(node.state)
            for successor, action in get_successors(node.state):
                successor_node = GameNode(successor, node, action, node.depth + 1)

                if successor not in visited:
                    result = dls(successor_node, depth_limit - 1, visited)
                    if result is not None:
                        return result

            visited.remove(node.state)
        return None

    depth = 0
    while True:
        visited = set()
        result = dls(GameNode(initial_state), depth, visited)
        if result is not None:
            return result
        depth += 1

def a_star(initial_state):
    frontier = []
    initial_state.criterion = 'f_cost'
    heapq.heappush(frontier, GameNode(initial_state))
    explored = set()

    while frontier:
        current_node = heapq.heappop(frontier)
        current_state = current_node.state

        if current_state.goal_test():
            return current_node

        explored.add(current_state)

        for successor_state, action in get_successors(current_state):
            successor_node = GameNode(successor_state, current_node, action, current_node.depth + 1)

            if successor_state not in explored and successor_node not in frontier:
                successor_state.h_cost = heuristic(successor_state, initial_state)
                heapq.heappush(frontier, successor_node)

            elif successor_node in frontier:
                index = frontier.index(successor_node)
                if frontier[index] > successor_node:
                    frontier[index] = successor_node
                    heapq.heapify(frontier)

    return None


def best_first_search(initial_state):
    frontier = []
    initial_state.criterion = 'h_cost'
    heapq.heappush(frontier, GameNode(initial_state))
    explored = set()

    while frontier:
        current_node = heapq.heappop(frontier)
        current_state = current_node.state

        if current_state.goal_test():
            return current_node

        explored.add(current_state)

        for successor_state, action in get_successors(current_state):
            successor_node = GameNode(successor_state, current_node, action, current_node.depth + 1)

            if successor_state not in explored and successor_node not in frontier:
                successor_state.h_cost = heuristic(successor_state, initial_state)
                heapq.heappush(frontier, successor_node)

            elif successor_node in frontier:
                index = frontier.index(successor_node)
                if frontier[index] > successor_node:
                    frontier[index] = successor_node
                    heapq.heapify(frontier)

    return None


def ida_star(initial_state):
    initial_state.criterion = 'f_cost'
    bound = heuristic(initial_state, initial_state)

    while True:
        visited = {}
        result, new_bound = ida_star_iteration(GameNode(initial_state), bound, visited, initial_state)

        if result is not None:
            return result
        if new_bound == float('inf'):
            return None
        bound = new_bound


def ida_star_iteration(node, threshold, visited, initial_state):
    stack = [(node, 0)]
    min_exceeded = float('inf')

    while stack:
        current_node, g_cost = stack.pop()
        current_state = current_node.state

        if current_state.goal_test():
            return current_node, threshold

        if current_state.h_cost == float('inf'):
            current_state.h_cost = heuristic(current_state, initial_state)

        f_cost = g_cost + current_state.h_cost

        if f_cost > threshold:
            min_exceeded = min(min_exceeded, f_cost)
            continue

        if current_state in visited and g_cost >= visited[current_state]:
            continue

        visited[current_state] = g_cost

        for successor_state, action in get_successors(current_state):
            if successor_state not in visited or g_cost + 1 < visited[successor_state]:
                successor_node = GameNode(successor_state, current_node, action, current_node.depth + 1)
                stack.append((successor_node, g_cost + 1))

    return None, min_exceeded


def reconstruct_path(node):
    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    return path[::-1]


def reconstruct_board(node):
    index = []
    while node.parent is not None:
        index.append(node.state.board)
        node = node.parent
    index = np.flip(index)
    for i in index:
        print("***************************************************************")
        print(np.matrix(i.board))
        print("***************************************************************")



if __name__ == "__main__":
    start_board = [
        ["1", "3", "0", "X", "1"],
        ["0P", "1B", "0", "0", "1G"],
        ["0", "4", "0", "X", "1"],
        ["0", "0", "0", "0", "1"],
    ]
    player_pos = (0,0)
    balls_pos = []
    goals_pos = []
    traps_pos = []
    for x in range(len(start_board)):
        for y in range(len(start_board[0])):
            curr_loc = start_board[x][y]
            if curr_loc.isdigit():
                continue
            elif curr_loc =='X':
                traps_pos.append((x,y))
            elif curr_loc[1] == 'P':
                player_pos = (x,y)
            elif curr_loc[1] == 'B':
                balls_pos.append((x,y))
            elif curr_loc[1] == 'G':
                goals_pos.append((x,y))

    init_state = GameState(player_pos, start_board, balls_pos, init_trap=traps_pos,
                           traps=traps_pos, trap_state=1, goals=goals_pos)
    bfs_solution_state = bfs(init_state)
    if bfs_solution_state:
        print("BFS Solution found!")
        print(f"Move: {reconstruct_path(bfs_solution_state)}")
        print(f"depth: {bfs_solution_state.depth}")
        print(f"cost: {bfs_solution_state.state.g_cost}\n")
        print(f"----------------------------------------------")
    else:
        print("No solution found. \n")

    dfs_solution_state = dfs(init_state)
    if dfs_solution_state:
        print("DFS Solution found!")
        print(f"Move: {reconstruct_path(dfs_solution_state)}")
        print(f"depth: {dfs_solution_state.depth}")
        print(f"cost: {dfs_solution_state.state.g_cost}\n")
        print(f"----------------------------------------------")
    else:
        print("No solution found. \n")

    ucs_solution_state = ucs(init_state)
    if ucs_solution_state:
        print("UCS Solution found!")
        print(f"Move: {reconstruct_path(ucs_solution_state)}")
        print(f"depth: {ucs_solution_state.depth}")
        print(f"cost: {ucs_solution_state.state.g_cost}\n")
        print(f"----------------------------------------------")
    else:
        print("No solution found. \n")

    a_star_solution_state = a_star(init_state)
    if a_star_solution_state:
        print("A* Solution found!")
        print(f"Move: {reconstruct_path(a_star_solution_state)}")
        print(f"depth: {a_star_solution_state.depth}")
        print(f"cost: {a_star_solution_state.state.g_cost}\n")
        print(f"----------------------------------------------")
    else:
        print("No solution found. \n")

    best_first_search_solution_state = best_first_search(init_state)
    if best_first_search_solution_state:
        print("BEST FIRST SEARCH Solution found!")
        print(f"Move: {reconstruct_path(best_first_search_solution_state)}")
        print(f"depth: {best_first_search_solution_state.depth}")
        print(f"cost: {best_first_search_solution_state.state.g_cost}\n")
        print(f"----------------------------------------------")
    else:
        print("No solution found. \n")

    ids_solution_state = ids(init_state)
    if ids_solution_state:
        print("IDS Solution found!")
        print(f"Move: {reconstruct_path(ids_solution_state)}")
        print(f"depth: {ids_solution_state.depth}")
        print(f"cost: {ids_solution_state.state.g_cost}\n")
        print(f"----------------------------------------------")
    else:
        print("No solution found. \n")



    ida_star_solution_state = ida_star(init_state)
    if ida_star_solution_state:
        print("IDA* Solution found!")
        print(f"Move: {reconstruct_path(ida_star_solution_state)}")
        print(f"depth: {ida_star_solution_state.depth}")
        print(f"cost: {ida_star_solution_state.state.g_cost}\n")
        print(f"----------------------------------------------")
    else:
        print("No solution found. \n")













