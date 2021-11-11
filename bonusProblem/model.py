from matplotlib import pyplot as plt

import main


def func():
    x_coords_1 = [0]
    y_coords_1 = [0]
    x_coords_2 = [0]
    y_coords_2 = [0]
    cur_theta = 0
    cur_position = [0, 0]
    goal = [5,5]

    right_velocity_1 = .1
    left_velocity_1 = .4
    right_velocity_2 = 0
    left_velocity_2 = -.1


    # good example
    theta_candidate = main.get_new_angle(left_velocity_1, right_velocity_1, cur_theta)
    newPos_candidate = [main.get_new_x(left_velocity_1, right_velocity_1, theta_candidate, cur_position[0]),
                        main.get_new_y(left_velocity_1, right_velocity_1, theta_candidate, cur_position[1])]
    x_coords_1.append(newPos_candidate[0])
    y_coords_1.append(newPos_candidate[1])
    print(f"The old distance was {main.dist(cur_position, goal)}  and the new point is {main.dist(newPos_candidate, goal)} from the goal")


    # bad example
    theta_candidate = main.get_new_angle(left_velocity_2, right_velocity_2, cur_theta)
    newPos_candidate = [main.get_new_x(left_velocity_2, right_velocity_2, theta_candidate, cur_position[0]),
                        main.get_new_y(left_velocity_2, right_velocity_2, theta_candidate, cur_position[1])]
    x_coords_2.append(newPos_candidate[0])
    y_coords_2.append(newPos_candidate[1])
    print(f"The old distance was {main.dist(cur_position, goal)}  and the new point is {main.dist(newPos_candidate, goal)} from the goal")

    plt.plot(x_coords_1, y_coords_1, color="blue")
    plt.plot(x_coords_2, y_coords_2, color="red")
    #plt.axis('scaled')
    plt.savefig('model.png')



if __name__ == '__main__':
    func()