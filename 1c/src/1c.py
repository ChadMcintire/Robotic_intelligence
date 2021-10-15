import matplotlib.pyplot as plt
import numpy as np
import math
import time

def main():
    # size in meters
    L = 4.0
    # velocities in m/s
    v_rear = 20.0
    alpha = math.pi / 6
    t = 10
    # instantaneous curvature
    R = L / math.tan(alpha)

    names = ['delta_t = 1.0', 'delta_t = 0.1', 'delta_t = 0.01']
    delta_t = [1, 0.1, 0.01]

    plot_error(L, v_rear, alpha, t, R, names, delta_t, debug=False)
    plot_comp_time(L, v_rear, alpha, t, names, delta_t)


def plot_error(L, v_rear, alpha, t, R, names, delta_t, debug=False):
    for i in range(len(delta_t)):
        steps = int(t / delta_t[i])
        # for plotting
        all_t = [x*delta_t[i] for x in range(steps + 1)]
        # estimate values for Euler integration
        x_est = np.zeros([steps + 1])
        y_est = np.zeros([steps + 1])
        psi_est = np.zeros([steps + 1])
        # truth values
        truth_x = np.zeros(steps + 1)
        truth_y = np.zeros(steps + 1)
        local_err = np.zeros(steps + 1)
        # Ground Truth circle data
        cx = -R
        cy = 0

        for k in range(1, steps + 1):
            # Euler estimate
            psiVel = (v_rear / L) * math.tan(alpha)
            psi_est[k] = psi_est[k - 1] + (psiVel * delta_t[i])
            xVel = -v_rear * math.sin(psi_est[k - 1])
            yVel = v_rear * math.cos(psi_est[k - 1])
            x_est[k] = x_est[k - 1] + (xVel * delta_t[i])
            y_est[k] = y_est[k - 1] + (yVel * delta_t[i])

            # True value
            truth_psi = (v_rear * float(k) * delta_t[i]) / R
            truth_x[k], truth_y[k] = truth_pos(R, (cx, cy), truth_psi)
            # store value
            local_err[k] = distance((x_est[k], y_est[k]), (truth_x[k], truth_y[k])) / k

            # Debug - track point at each whole second
            if int(k*delta_t[i]) == k*delta_t[i] and debug:
                print("-" * 25)
                print("Time: %s\tk: %s" % (round(k * delta_t[i], 2), k))
                print("Euler\t\t>> Psi: %s,\tx_real: %s,\ty_real: %s" %
                      ('{:.13f}'.format(psi_est[k]), '{:.13f}'.format(x_est[k]), '{:.13f}'.format(y_est[k])))
                print("Truth\t\t>> Psi: %s,\tx_real: %s,\ty_real: %s" %
                      ('{:.13f}'.format(truth_psi), '{:.13f}'.format(truth_x[k]), '{:.13f}'.format(truth_y[k])))
                print("Error (m)\t>> %s" % "{:.13f}".format(local_err[k]))
        # add the line and points for delta_t[i]
        plt.plot(all_t, local_err, label=names[i])
        # todo testing x ticks
        if i == 0:
            plt.ylim([-delta_t[0], max(local_err) + delta_t[0]])

    x_label = "Time (sec)"
    y_label = "Error (m)"
    title = "Local Error of Euler"
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_comp_time(L, v_rear, alpha, t, names, delta_t):
    for i in range(len(delta_t)):
        steps = int(t / delta_t[i])
        # for plotting
        all_t = [x for x in range(t + 1)]
        comp_t = np.zeros(steps + 1)

        # estimate values for Euler integration
        x_est = np.zeros([steps + 1])
        y_est = np.zeros([steps + 1])
        psi_est = np.zeros([steps + 1])

        for k in range(1, steps + 1):
            # compute <repeat> times then take average for more stable compute time
            repeat = 500
            for j in range(0, repeat):
                # Euler estimate
                start = time.time() * 1000

                psiVel = (v_rear / L) * math.tan(alpha)
                psi_est[k] = psi_est[k - 1] + (psiVel * delta_t[i])
                xVel = -v_rear * math.sin(psi_est[k - 1])
                yVel = v_rear * math.cos(psi_est[k - 1])
                x_est[k] = x_est[k - 1] + (xVel * delta_t[i])
                y_est[k] = y_est[k - 1] + (yVel * delta_t[i])

                comp_t[k] += (time.time()*1000) - start

            comp_t[k] = comp_t[k] / repeat
        # sum up values of k on whole second intervals
        sum_comp_t = []
        for v in range(t):
            slice_start = int(v*steps/t)
            slice_end = int((v*steps/t) + steps / t)
            sum_comp_t.append(sum(comp_t[slice_start:slice_end]))

        plt.plot(all_t[1:], sum_comp_t, label=names[i])

    x_label = "Time in Simulation"
    y_label = "Compute Time (ms)"
    title = "Computation Time Comparison"
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


# distance formula between 2 points a and b
def distance(a, b):
    assert isinstance(a, tuple) and isinstance(b, tuple)
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def truth_pos(R, center, theta):
    assert(isinstance(center, tuple))
    x = R * math.cos(theta) + center[0]
    y = R * math.sin(theta) + center[1]
    return x, y


if __name__ == "__main__":
    main()
