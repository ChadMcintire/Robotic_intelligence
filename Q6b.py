import random
import math
from sys import argv
from queue import PriorityQueue
from multiprocessing import Process, Pool, managers


goal = [1.2, 0.8, 0.5]

# a1 = 0.6
# a2 = 0.4
curTheta1 = 0
curD2 = 0
curD3 = 0
curTheta4 = 0
curTheta5 = 0
curTheta6 = 0
d1 = 0.5      #distance from base to  d2
d6 = 0.5      #distance from theta6 to end effector
curPos = [0.5, 0.5, 0.0]

class RobotCommand:
    """Class that stores robot command and performs related computations"""
    __theta1: float 
    __d2: float 
    __d3: float 
    __theta4: float 
    __theta5: float 
    __theta6: float

    def __init__(self, theta1, d2, d3, theta4, theta5, theta6):
        self.__theta1 = theta1 
        self.__d2 = d2 
        self.__d3 = d3 
        self.__theta4 = theta4 
        self.__theta5 = theta5 
        self.__theta6 = theta6

    def dist(self) -> float:
        """Computes distance away from goal for command"""
        def get_xyz(t1, d2, d3, t4, t5 ,t6):
            """Compute new position"""
            newX = (math.cos(t1) * math.cos(t4) * math.sin(t5) * d6) - (math.sin(t1) * math.cos(t5) * d6) - (math.sin(t1) * d3)
            newY = (math.sin(t1) * math.cos(t4) * math.sin(t5) * d6) + (math.cos(t1) * math.cos(t5) * d6) + (math.cos(t1) * d3)
            newZ = (-1 * (math.sin(t4) * math.sin(t5) * d6)) + d1 + d2
            return [newX, newY, newZ]

        def dist(cur, goal) -> float:
            """Compute distance from goal"""
            return math.sqrt(((cur[0] - goal[0])**2) + ((cur[1] - goal[1])**2) + ((cur[2] - goal[2])**2))

        return dist(get_xyz(self.__theta1, self.__d2, self.__d3, self.__theta4, self.__theta5, self.__theta6), goal)

    def cost(self) -> float:
        """Computes cost of command"""
        return abs(self.__theta1) * 3 + (abs(self.__d2) + abs(self.__d3)) * 2 + abs(self.__theta4) + abs(self.__theta5) + abs(self.__theta6)

    def copy(self):
        """Makes a copy of the command"""
        return RobotCommand(self.__theta1, self.__d2, self.__d3, self.__theta4, self.__theta5, self.__theta6)

    def randomize(self) -> None:
        """Adds random offsets to each param"""
        def get_random_draw() -> float:
            ran = random.random() - 0.5
            # Why divide by 1?
            ran = ran/1
            return ran 
        self.__theta1 += get_random_draw()
        self.__d2 += get_random_draw()

        self.__d3 += get_random_draw()
        self.__theta4 += get_random_draw()
        self.__theta5 += get_random_draw()
        self.__theta6 += get_random_draw()

    def __lt__(self, other):
        """Comparisons are made using the cost."""
        return self.cost() < other.cost()

    def __str__(self):
        return f"(t1: {self.__theta1}, d2: {self.__d2}, d3: {self.__d3}, t4: {self.__theta4}, t5: {self.__theta5}, t6: {self.__theta6}) Dist: {self.dist()} Cost: {self.cost()}"

def runSimulation(args: tuple) -> None:
    iterLimit, innerIterations, tolerance, solutions = args
    command = RobotCommand(0,0,0,0,0,0)
    minCost = 0.1
    for i in range(iterLimit):
        for _ in range(10000):
            newCommand = command.copy()
            newCommand.randomize()
            # If distance is smaller than closest solution and cost is below threshold or best found, update cost theshold and closest solution
            if newCommand.dist() < command.dist() and newCommand.cost() < minCost:
                print(f"Updated position: {command}\n")
                command = newCommand
                minCost = newCommand.cost()

        minCost += 0.1

        if i == iterLimit - 1:
            print("iter limit reached.")
        
        if command.dist() < tolerance:
            print(f"\n\nSolution: {command}\n\nIterations: {i}")
            solutions.put(command)
            break

if __name__ == "__main__":
    #Direction 1
    goal = [1, 1, 1]
    curTheta1 = 0
    curD2 = 0
    curD3 = 0
    curTheta4 = 0
    curTheta5 = 0
    curTheta6 = 0
    d1 = 0.5      #distance from base to  d2
    d6 = 0.5      #distance from theta6 to end effector
    curPos = [0, 0, 0]

    managers.SyncManager.register("PriorityQueue", PriorityQueue)
    manager = managers.SyncManager()
    manager.start()
    solutions = manager.PriorityQueue()
    iterLimit = 100
    innerIterations = 500
    tolerance = 0.5
    processes = 4

    pool = Pool(processes=processes)
    pool.map(runSimulation, [(iterLimit, innerIterations, tolerance, solutions) for _ in range(processes)])
    a = solutions.get()
    print(f"\n\nBest Solution: {a}")


    goal = [-1, 2, 1]
    curTheta1 = a._RobotCommand__theta1
    curD2 = a._RobotCommand__d2
    curD3 = a._RobotCommand__d3
    curTheta4 = a._RobotCommand__theta4
    curTheta5 = a._RobotCommand__theta5
    curTheta6 = a._RobotCommand__theta6
    d1 = 0.5      #distance from base to  d2
    d6 = 0.5      #distance from theta6 to end effector
    curPos = [1, 1, 1]
    managers.SyncManager.register("PriorityQueue", PriorityQueue)
    manager = managers.SyncManager()
    manager.start()
    solutions = manager.PriorityQueue()
    iterLimit = 100
    innerIterations = 500
    tolerance = 0.5
    processes = 4

    pool = Pool(processes=processes)
    pool.map(runSimulation, [(iterLimit, innerIterations, tolerance, solutions) for _ in range(processes)])
    b = solutions.get()
    print(f"\n\nBest Solution: {a}")

    c1 = a.cost()
    c2 = b.cost()
    D1Cost = c1+c2
    pool.close()

    #Direction 2

    goal = [-1, 21, 1]
    curTheta1 = 0
    curD2 = 0
    curD3 = 0
    curTheta4 = 0
    curTheta5 = 0
    curTheta6 = 0
    d1 = 0.5      #distance from base to  d2
    d6 = 0.5      #distance from theta6 to end effector
    curPos = [0, 0, 0]

    managers.SyncManager.register("PriorityQueue", PriorityQueue)
    manager = managers.SyncManager()
    manager.start()
    solutions = manager.PriorityQueue()
    iterLimit = 100
    innerIterations = 500
    tolerance = 0.5
    processes = 4

    pool = Pool(processes=processes)
    pool.map(runSimulation, [(iterLimit, innerIterations, tolerance, solutions) for _ in range(processes)])
    a = solutions.get()
    print(f"\n\nBest Solution: {a}")


    goal = [1, 1, 1]
    curTheta1 = a._RobotCommand__theta1
    curD2 = a._RobotCommand__d2
    curD3 = a._RobotCommand__d3
    curTheta4 = a._RobotCommand__theta4
    curTheta5 = a._RobotCommand__theta5
    curTheta6 = a._RobotCommand__theta6
    d1 = 0.5      #distance from base to  d2
    d6 = 0.5      #distance from theta6 to end effector
    curPos = [-1, 2, 1]
    managers.SyncManager.register("PriorityQueue", PriorityQueue)
    manager = managers.SyncManager()
    manager.start()
    solutions = manager.PriorityQueue()
    iterLimit = 100
    innerIterations = 500
    tolerance = 0.5
    processes = 4

    pool = Pool(processes=processes)
    pool.map(runSimulation, [(iterLimit, innerIterations, tolerance, solutions) for _ in range(processes)])
    b = solutions.get()
    print(f"\n\nBest Solution: {a}")

    c1 = a.cost()
    c2 = b.cost()
    D2Cost = c1+c2
    pool.close()

    print()
    print()
    print()
    print("(0, 0, 0) -> (1, 1, 1) -> (-1, 2, 1) Energy Cost:", D1Cost)
    print("(0, 0, 0) -> (-1, 2, 1) -> (1, 1, 1) Energy Cost:", D2Cost)
