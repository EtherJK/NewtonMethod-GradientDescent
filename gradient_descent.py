import numpy as np

def gradient_descent(fun, initial, pace = 0.1, limitation = None):
    # using gradient descent to get the minimum value of function
    # fun: the function would like to solve
    # initial: the intitial value of parameters
    #MAX_ERROR = 1e-7 # the cretirion of solved
    MAX_ITER = int(1e8) # max number of iteration
    DISPLAY_INTERVAL = 1 # every n steps to display the process
    F_judge_old = 1e5 # to judge the balance point
    F_judge_new = 1e4 
    judge_times = 20 # how many steps to judge

    # initial
    x = np.array(initial, dtype=float)
    judge_sum = 0 

    for iter_times in range(0, MAX_ITER):
        F_new = fun(x)

        # old way
        #if abs(F_new - F_old) < MAX_ERROR:
        #    return x
       
        # calculate the gradient
        gradient = calculate_gradient(fun, x, limitation)

        # calculate movement
        #move = - pace * gradient * abs(x) # for the case when x -> 0, dx should be very small, use this
        move = - pace * gradient
        move = consider_limitation(x, move, limitation)
        x += move

        # add to sum, to get average
        judge_sum += F_new

        # display the progress
        if iter_times % DISPLAY_INTERVAL == 0:
            print('Iteration Times: ', iter_times)
            print('Function value: ', F_new)
            print('move Now: ', move)
            print('x Now: ', x)

        # judge the end of iteration
        if iter_times % judge_times == 0:
            if F_judge_new > F_judge_old:
                return x
            F_judge_old = F_judge_new
            F_judge_new = judge_sum / judge_times
            judge_sum = 0



def calculate_gradient(fun, x, limitation):
    EPSILON = 1e-5 # small value of changing to calculate gradient
    gradient = np.zeros(len(x), dtype=float)

    Fx = fun(x)
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
        gradient[i] = (fun(x_dx) - Fx) / dx
    
    return gradient

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
    y = x[0]**2 + x[1]**2 - 2*x[0]*x[1]
    return y


if __name__ == '__main__':
    print(gradient_descent(function_example, np.array([2, -1])))