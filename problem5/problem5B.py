import random
import math

goal = [1.2, 0.8, 0.5]

curTheta1 = -90
curD2 = 0.5
curD3 = 1
curTheta4 = -90
curTheta5 = 90
curTheta6 = 40
d1 = 0.5      #distance from base to  d2
d6 = 0.5      #distance from theta6 to end effector

numMotors = 6

t1Dist = 0
d2Dist = 0
d3Dist = 0
t4Dist = 0
t5dist = 0
t6Dist = 0
totalDist = 0

iterLimit = 1000000
curPos = [1.0, 0.0, 1.5]    # meters
tolerance = 0.01


# calulates new 3D coordinates
def get_xyz(t1, d2, d3, t4, t5, t6):
    newX = (math.cos(t1) * math.cos(t4) * math.sin(t5) * d6) - (math.sin(t1) * math.cos(t5) * d6) - (math.sin(t1) * d3)
    newY = (math.sin(t1) * math.cos(t4) * math.sin(t5) * d6) + (math.cos(t1) * math.cos(t5) * d6) + (math.cos(t1) * d3)
    newZ = (-1 * (math.sin(t4) * math.sin(t5) * d6)) + d1 + d2
    return [newX, newY, newZ]
    
# Generate random number between -0.5 and 0.5
def get_random_draw():
    return random.random() - 0.5
    
# standard 3D distance formula
def dist(cur, goal):
    return math.sqrt(((cur[0] - goal[0])**2) + ((cur[1] - goal[1])**2) + ((cur[2] - goal[2])**2))

# Adds up total motor distance traveled 
def motorsDist(m1, m2, m3, m4, m5, m6):
    return abs(m1) + abs(m2) + abs(m3) + abs(m4) + abs(m5) + abs(m6)

maxDist = 0.01  # max distance to travel per iteration
for i in range(iterLimit):
    for j in range(1000):
        # generate random motor movements
        temp1 = get_random_draw()
        temp2 = get_random_draw()
        temp3 = get_random_draw()
        temp4 = get_random_draw()
        temp5 = get_random_draw()
        temp6 = get_random_draw()
        # get total distance for all motors
        tempDist = motorsDist(temp1, temp2, temp3, temp4, temp5, temp6)
        # potential new position based on previous random motor movements
        tempPos = get_xyz(temp1 + curTheta1, temp2 + curD2, temp3 + curD3, temp4 + curTheta4, temp5 + curTheta5, temp6 + curTheta6)
        
        if dist(tempPos, goal) < dist(curPos, goal) and tempDist < maxDist:
            # update values
            curTheta1 += temp1
            curD2 += temp2
            curD3 += temp3
            curTheta4 += temp4
            curTheta5 += temp5
            curTheta6 += temp6
            curPos = tempPos
            t1Dist += abs(temp1)
            d2Dist += abs(temp2)
            d3Dist += abs(temp3)
            t4Dist += abs(temp4)
            t5dist += abs(temp5)
            t6Dist += abs(temp6)
            totalDist += tempDist

    maxDist += 0.01

    if i == iterLimit - 1:
        print("iter limit reached.")
    
    # if solution found print values
    if(dist(curPos, goal) < tolerance):
        print("Solution found!\n")
        print("Angles -------------------------")
        print("theta 1: ", curTheta1)
        print("D2: ", curD2)
        print("D3: ", curD3)
        print("theta 4: ", curTheta4)
        print("theta 5: ", curTheta5)
        print("theta 6: ", curTheta6)
        print("\nDistances --------------------")
        print("theta 1 dist: ", t1Dist)
        print("d 2 dist: ", d2Dist)
        print("d 3 dist: ", d3Dist)
        print("theta 4 dist: ", t4Dist)
        print("theta 5 dist: ", t5dist)
        print("theta 6 dist: ", t6Dist)
        print("\ntotalDist: ", totalDist)
        break
