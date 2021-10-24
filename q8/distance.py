from math import sqrt

def findPathDistance(rx: list, ry: list) -> float:
    """rx is a list corresponding to x positions, ry is a list corresponding to y posititions"""
    distance = 0
    currX = rx[0]
    currY = ry[0]
    for x, y in zip(rx[1:], ry[1:]):
        distance += sqrt((x-currX) ** 2 + (y-currY) ** 2)
        currX = x
        currY = y
    return distance
  
