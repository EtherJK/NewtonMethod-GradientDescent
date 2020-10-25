import numpy as np

def Newton_method(fun, initial, pace = 0.5, limitation = None):
    # gradient descent to solve the equations fun = 0
    # fun: the function would like to solve
    # g_fun: the gradient funtion of original one
    # initial: the intitial value of parameters
    MAX_ERROR = 1e-7 # the cretirion of solved
    EPSILON = 1e-5 # small value of changing to calculate gradient
    MAX_ITER = int(1e8) # max number of iteration
    MAX_RAPHSON = 10 # max turns of down-hill calculation
    DISPLAY_INTERVAL = 10 # every n steps to display the process

    # judge the direction of descent
    l = len(initial)
    jacobian = np.zeros((l, l), dtype=float)
    x = np.array(initial, dtype=float)
    for iter_times in range(0, MAX_ITER):
        Fx = fun(x)
        previous_deviation = np.sum(abs(Fx))
        if previous_deviation < MAX_ERROR:
            #print(jacobian)
            return x

        for i in range(0, len(x)):
            # calculate the derivative of each equations
            x_dx = x.copy()
            dx = x[i] * EPSILON
            if x[i] == 0:
                dx = EPSILON
            
            # consider the limitation
            if limitation:
                if limitation[i][1]:
                    if dx + x[i] > limitation[i][1]:
                        dx = -dx
            x_dx[i] += dx

            # calculate a column of Jacobian
            dif = (fun(x_dx) - Fx) / dx
            jacobian[:, i] = dif

        # calculate the movement of x    
        inv_jaco = np.linalg.inv(jacobian)
        #inv_jaco = norm(inv_jaco) # if u would like to normalize the jacobian, use this
        direction = - np.dot(inv_jaco, Fx)
        move = pace * direction

        # reconsider the movement of x with the limitation
        move = consider_limitation(x, move, limitation)
        x2 = x + move

        # Newton-Raphson part
        now_deviation = np.sum(abs(fun(x2)))
        if now_deviation > previous_deviation:
            for iter_raphson in range(0, MAX_RAPHSON):
                if now_deviation < previous_deviation:
                    print(iter_raphson)
                    break
                else:
                    move /= 2
                    x2 = x + move
                    now_deviation = np.sum(abs(fun(x2)))

        x = x2

        # display the progress
        if iter_times % DISPLAY_INTERVAL == 0:
            print('Iteration Times: ', iter_times)
            print('Deviation Now: ', now_deviation)
            print('x now: ', x)

def norm(x):
    mx = np.max(np.max(x))
    mn = np.min(np.min(x))
    if abs(mx) > 1 or abs(mn) > 1:
        x /= (mx - mn)
    return x

def consider_limitation(x, dx, limitation):
    TIMES = 2 # if out of limitation, TIMES is how to get dx
    if limitation is None:
        return dx
    new_dx = dx.copy()
    x_dx = x + dx
    for i in range(0, len(limitation)):
        if limitation[i][0] is not None: # if there is lower limit
            if x_dx[i] < limitation[i][0]:
                new_dx[i] = -(x[i] - limitation[i][0]) / TIMES
        if limitation[i][1] is not None: # if there is higher limit
            if x_dx[i] > limitation[i][1]:
                new_dx[i] = (limitation[i][1] - x[i]) / TIMES
    return new_dx

def function_example(x):
    y = np.array([x[0]**2 - 10 * x[0] + x[1]**2 + 8, x[0] * x[1]**2 + x[0] - 10*x[1] + 8])
    return y


if __name__ == '__main__':
    print(Newton_method(function_example, np.array([0, 0])))
    
