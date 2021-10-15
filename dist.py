import random
import math

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
iterLimit = 10000
curPos = [0.5, 0.5, 0.0]
tolerance = 0.5


def get_xyz(t1, d2, d3, t4, t5 ,t6):
    newX = (math.cos(t1) * math.cos(t4) * math.sin(t5) * d6) - (math.sin(t1) * math.cos(t5) * d6) - (math.sin(t1) * d3)
    newY = (math.sin(t1) * math.cos(t4) * math.sin(t5) * d6) + (math.cos(t1) * math.cos(t5) * d6) + (math.cos(t1) * d3)
    newZ = (-1 * (math.sin(t4) * math.sin(t5) * d6)) + d1 + d2
    return [newX, newY, newZ]
    
def get_random_draw():
    ran = random.random() - 0.5
    ran = ran/1
    return ran
    
def dist(cur, goal):
    return math.sqrt(((cur[0] - goal[0])**2) + ((cur[1] - goal[1])**2) + ((cur[2] - goal[2])**2))
    
for i in range(iterLimit):
    tempTheta1 = curTheta1 + get_random_draw()
    tempD2 = curD2 + get_random_draw()
    tempD3 = curD3 + get_random_draw()
    tempTheta4 = curTheta4 + get_random_draw()
    tempTheta5 = curTheta5 + get_random_draw()
    tempTheta6 = curTheta6 + get_random_draw()

    PotentialNewPos = get_xyz(tempTheta1, tempD2, tempD3, tempTheta4, tempTheta5, tempTheta6)

    print(dist(curPos, goal), dist(PotentialNewPos, goal))
    print(dist(curPos, goal) > dist(PotentialNewPos, goal))

    if dist(curPos, goal) > dist(PotentialNewPos, goal):
        curPos = PotentialNewPos
        curTheta1 = tempTheta1
        curD2 = tempD2
        curD3 = tempD3
        curTheta4 = tempTheta4
        curTheta5 = tempTheta5
        curTheta6 = tempTheta6
        print("Updated position")

    if i == iterLimit - 1:
        print("iter limit reached.")
    
    if dist(curPos, goal) < tolerance:
        print("Success!")
        print("theta 1: ", curTheta1)
        print("theta 1: ", curTheta1)
        print("theta 1: ", curTheta1)
        print("theta 1: ", curTheta1)
        print("theta 1: ", curTheta1)
        print("theta 1: ", curTheta1)
        print("i: ", i)
        break