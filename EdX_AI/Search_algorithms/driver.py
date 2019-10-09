import queue

import time

import resource

import sys

import math

#### SKELETON CODE ####

## The Class that Represents the Puzzle

class PuzzleState(object):

    """docstring for PuzzleState"""

    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        if n*n != len(config) or n < 2:

            raise Exception("the length of config is not correct!")

        self.n = n

        self.cost = cost

        self.parent = parent

        self.action = action

        self.dimension = n

        self.config = config

        self.children = []

        for i, item in enumerate(self.config):

            if item == 0:

                self.blank_row = int(i / self.n)

                self.blank_col = i % self.n

                break

    def display(self):

        for i in range(self.n):

            line = []

            offset = i * self.n

            for j in range(self.n):

                line.append(self.config[offset + j])

            print(line)

    def move_left(self):

        if self.blank_col == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):

        if self.blank_col == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):

        if self.blank_row == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):

        if self.blank_row == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):

        """expand the node"""

        # add child nodes in order of UDLR

        if len(self.children) == 0:

            up_child = self.move_up()

            if up_child is not None:

                self.children.append(up_child)

            down_child = self.move_down()

            if down_child is not None:

                self.children.append(down_child)

            left_child = self.move_left()

            if left_child is not None:

                self.children.append(left_child)

            right_child = self.move_right()

            if right_child is not None:

                self.children.append(right_child)

        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters

def writeOutput(state,visited):

    # Get actions
    actions = list()

    curr = state

    while curr.parent is not None:

        actions.append(curr.action)

        curr = curr.parent

    print("path_to_goal: ", actions)

    print("cost_of_path: ", state.cost)

    print("nodes_expanded: ", len(visited))


def bfs_search(initial_state):

    """BFS search"""

    start_time = time.time()

    # make list of visited states
    visited = list()

    # queue of PuzzleStates
    q = queue.Queue()

    # add initial State
    s = initial_state

    visited.append("".join(map(str,s.config)))

    q.put(s)

    found = False

    while not q.empty or not found:

            # Get next state (FIFO)
            curr = q.get()

            # Check state to goal
            if test_goal(curr):

                found = True

            else:

                for child in curr.expand():

                    if "".join(map(str,child.config)) not in visited:

                        q.put(child)

                        visited.append("".join(map(str,child.config)))

    writeOutput(curr,visited)

    print("--- %s seconds ---" % (time.time() - start_time))

def dfs_search(initial_state):

    """DFS search"""

    start_time = time.time()
    # make list of visited states
    visited = list()

    # queue of PuzzleStates
    q = queue.Queue()

    # add initial State
    s = initial_state

    visited.append("".join(map(str,s.config)))

    q.put(s)

    found = False

    while not q.empty or not found:

            # Get next state (FIFO)
            curr = q.get()

            # Check state to goal
            if test_goal(curr):

                found = True

            else:

                for child in curr.expand():

                    if "".join(map(str,child.config)) not in visited:

                        q.put(child)

                        visited.append("".join(map(str,child.config)))

    writeOutput(curr,visited)

    print("--- %s seconds ---" % (time.time() - start_time))

def A_star_search(initial_state):

    """A * search"""

    n = initial_state.n

    start_time = time.time()
    # make list of visited states
    visited = list()

    # queue of PuzzleStates
    q = queue.PriorityQueue()

    # add initial State
    s = initial_state

    visited.append("".join(map(str,s.config)))


    counter = 0

    q.put((0,counter,s))

    found = False


    while not q.empty or not found:

            counter+=1

            # Get next state (FIFO)
            dist,c,curr = q.get()

            # Check state to goal
            if test_goal(curr):

                found = True

            else:

                for child in curr.expand():

                    counter+=1

                    if "".join(map(str,child.config)) not in visited:

                        q.put((calculate_manhattan_dist(child.config,n),counter,child))

                        visited.append("".join(map(str,child.config)))

    writeOutput(curr,visited)

    print("--- %s seconds ---" % (time.time() - start_time))

def calculate_total_cost(state):

    """calculate the total estimated cost of a state"""

    ### STUDENT CODE GOES HERE ###

def calculate_manhattan_dist(conf, n):

    """calculate the manhattan distance of a tile"""

    # define goal states
    goal = []

    for i in range(len(conf)*len(conf)):
        goal.append(i)

    total_dist = int()

    for i in range(n*n):

        a = conf[i]
        a_col = i % n
        a_row = int(i/n)

        b_col = goal.index(a) % n
        b_row = int(goal.index(a) / n)

        dist = abs(b_col - a_col) + abs(b_row - a_row)

        total_dist+=dist

    return(total_dist)


def test_goal(puzzle_state):

    """test the state is the goal state or not"""

    # define goal states
    goal = []

    for i in range(puzzle_state.n*puzzle_state.n):
        goal.append(i)

    if puzzle_state.config == tuple(goal):

        return True

    else:

        return False

# Main Function that reads in Input and Runs corresponding Algorithm

def main():

    sm = sys.argv[1].lower()

    begin_state = sys.argv[2].split(",")

    begin_state = tuple(map(int, begin_state))

    size = int(math.sqrt(len(begin_state)))

    hard_state = PuzzleState(begin_state, size)

    if sm == "bfs":

        bfs_search(hard_state)

    elif sm == "dfs":

        dfs_search(hard_state)

    elif sm == "ast":

        A_star_search(hard_state)

    else:

        print("Enter valid command arguments !")

if __name__ == '__main__':

    main()
