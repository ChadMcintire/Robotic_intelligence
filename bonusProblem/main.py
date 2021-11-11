from math import sin, cos
import random
import math

from matplotlib import pyplot as plt

W = 0.33
ts = .01

debug = False


# functions for getting new angle, new x and new y
def get_new_angle(v_left, v_right, phi):
    return phi + (v_right - v_left) / W * ts


def get_new_x(v_left, v_right, phi, X):
    return X + -1 * (v_right + v_left) / 2 * sin(phi) * ts


def get_new_y(v_left, v_right, phi, Y):
    return Y + (v_right + v_left) / 2 * cos(phi) * ts


def dist(cur, goal):
    return math.sqrt((cur[0] - goal[0]) ** 2 + (cur[1] - goal[1]) ** 2)


def get_random_draw(quenching=10.0):
    ran = random.random() - 0.5
    ran = ran / quenching
    return ran


def calc_path_dist(rx, ry):
    distance = 0
    currX = rx[0]
    currY = ry[0]
    for x, y in zip(rx[1:], ry[1:]):
        distance += math.sqrt((x - currX) ** 2 + (y - currY) ** 2)
        currX = x
        currY = y
    return distance


def simulation(goal, iterations, tolerance=1):
    print(f"starting with the goal of ({goal[0]}, {goal[1]}) for {iterations} iterations")
    # store the current set of x and y coords
    x_coords = []
    y_coords = []
    # store the current theta and posistion
    cur_theta = 0
    cur_position = [0, 0]
    # the amount of loops a single iteration receives before timeout.
    iterLimit = 50000

    bestX = []
    bestY = []
    # store a really big number so the first path will be stored
    running_score = 900000000
    bestIter = 0
    for j in range(iterations):
        # each iteration the list of  x and y coords are reset to just hold 0,0 and the current heading is set to 0 and the cur position is set to 0,0
        x_coords = [0]
        y_coords = [0]
        cur_theta = 0
        cur_position = [0, 0]
        for i in range(iterLimit):
            left_velocity = get_random_draw(quenching=1)
            right_velocity = get_random_draw(quenching=1)
            theta_candidate = get_new_angle(left_velocity, right_velocity, cur_theta)
            newPos_candidate = [get_new_x(left_velocity, right_velocity, theta_candidate, cur_position[0]),
                                get_new_y(left_velocity, right_velocity, theta_candidate, cur_position[1])]
            if dist(cur_position, goal) > dist(newPos_candidate, goal) and random.random() < 0.9:
                cur_position = newPos_candidate
                cur_theta = theta_candidate
                x_coords.append(cur_position[0])
                y_coords.append(cur_position[1])
            else:
                pass
            # exit as soon as the path is longer. this is an optimization so we ignore paths that are longer
            if dist(cur_position, goal) < tolerance:
                break
        else:
            if debug:
                print(f"Iteration limit met on iteration {j}")
        # every 500 iterations print a statement to inform user where in the process you are at.
        if j % 500 == 0:
            print(f"Continuing to goal of ({goal[0]}, {goal[1]}) and am at iteration {j} of {iterations}")

        if dist(cur_position, goal) < tolerance:
            x_coords.append(goal[0])
            y_coords.append(goal[1])
            length = calc_path_dist(x_coords, y_coords)
            if debug:
                print(f"Path {j} got to the goal with a length of {length}")
            # if the new path is shorter then the old shortest update the old shortest
            if length < running_score:
                if debug:
                    print(f"new shortest path found on iteration {j} of length {length}")
                bestX = x_coords
                bestY = y_coords
                running_score = length
                bestIter = j
            else:
                if debug:
                    print(f"Iteration {j} is not the largest")
        else:
            if debug:
                print(f"iteration {j} did not make it to the end")
    # plot
    print(f"plotting best iteration ({bestIter} with length {running_score}")
    plt.plot(bestX, bestY)
    plt.axis('scaled')
    plt.title(f"Path found on Iteration {bestIter} of {iterations} and has a length of {running_score}")
    plt.savefig(f'goal_{goal[0]}_{goal[1]}_iterations_{iterations}.png')
    plt.clf()
    print(f"finising  goal of ({goal[0]}, {goal[1]}) for {iterations} iterations")


def main():
    #print("Starting goal (5, 5)")
    #simulation([5, 5], 1000, 1)
    #simulation([5, 5], 10000, 1)
    #simulation([5, 5], 100000, 1)

    #print("Starting goal (10, 2)")
    #simulation([10, 2], 1000, 1)
    #simulation([10, 2], 10000, 1)
    #simulation([10, 2], 100000, 1)

    print("Starting goal (8, -3)")
    simulation([8, -3], 1000, 1)
    simulation([8, -3], 10000, 1)
    simulation([8, -3], 100000, 1)


if __name__ == '__main__':
    main()
