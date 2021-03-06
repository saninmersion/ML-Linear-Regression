import numpy as np
import matplotlib.pyplot as plt

# <<< Variables >>>

data = np.genfromtxt('./data/experience-salary-dataset.csv', delimiter=",")

initial_m = 0 # initial slope guess
initial_c = 0 # initial y-intercept guess

learning_rate = 0.0001
num_iterations = 1000

# <<< Functions >>>

# Linear Equation Function
def linear_equation(m,x,c):
    return m * x + c

# Plot Function
def plot(points, m, c):
    x = points[:,0]
    y = points[:,1]
    plt.scatter(x, y)
    plt.plot(x, linear_equation(m, x, c), label="fit line for m={} and b={}".format(m,c))
    plt.legend()
    plt.show()

# Error Computing Fucntion
def compute_error(points, m, c):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - linear_equation(m, x, c)) ** 2
    return round(totalError / float(len(points)*100000000), 3)

# sum of squares due to regression
def compute_ssr(data, calculated_m, calculated_c):
    ymean = np.mean(data[:,1])
    totalError = 0

    for i in range(0, len(data)):
        totalError += (linear_equation(calculated_m, data[i,0], calculated_c) - ymean) ** 2

    return totalError 

# sum of squares due to error
def compute_sse(data,calculated_m, calculated_c):
    totalError = 0

    for i in range(0, len(data)):
        totalError += (data[i,1] - linear_equation(calculated_m, data[i,0], calculated_c)) ** 2

    return totalError

def compute_rss(data, calculated_m, calculated_c):
    ssr = compute_ssr(data,calculated_m,calculated_c)
    sse = compute_sse(data,calculated_m,calculated_c)

    # print("ssr={}\nsse={}".format(ssr, sse))
    return round(ssr/float(ssr+sse), 8)


# Step Gradient Function
def step_gradient(points, learningRate, m_current, c_current):
    m_gradient = 0
    c_gradient = 0
    
    N = float(len(points))
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        m_gradient += -(2/N) * x * (y - ((m_current * x) + c_current))
        c_gradient += -(2/N) * (y - ((m_current * x) + c_current))

    new_m = m_current - (learningRate * m_gradient)
    new_c = c_current - (learningRate * c_gradient)
    return [new_m, new_c]

# Gradient Decent 
def calculate_m_b_with_gradient_descent(data, learning_rate, num_iterations, starting_m, starting_c):
    m = starting_m
    c = starting_c

    error = compute_error(data, m , c)
    error_array = np.array([])
    rss_error = compute_rss(data, m, c)

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(131)
    gradient, = ax.plot(c, m)
    ax.set_xlim([0,200])
    ax.set_ylim([0,1000])
    
    bx = fig.add_subplot(132)
    bx.scatter(data[:,0], data[:,1])
    lr, = bx.plot(data[:,0], linear_equation(m, data[:,0], c), label="fit line for y={0}x + {1}".format(m,c))
    bx.legend()
    
    ep = fig.add_subplot(133)
    ep.set_xlim([0,200])
    ep.set_ylim([0,1000])
    eplot = ep.plot(0, error)
    
    plt.ion()
    plt.show()

    for i in range(num_iterations):
        m, c = step_gradient(np.array(data), learning_rate, m, c)
        rss_error = compute_rss(data, m, c)
        error = compute_error(data, m, c)

        error_array = np.append(error_array, error)
        # print("After {0} iterations m = {1}, c = {2}, error = {3}, rss = {4}".format(i, m, c, error, rss_error))

        gradient.set_data(c, m)
        lr.set_data(data[:,0], linear_equation(m, data[:,0], c))
        eplot.set_data(range(i),error_array)
        print("iteration: {}".format(i))
        plt.draw()
        plt.pause(0.0001)

        # plot(x, y, m, b)
        # print("error: {}".format(error[i]))
    
    plot_x = data[:,0]
    plot_y = data[:,1]
    plt.scatter(plot_x, plot_y)
    plt.plot(plot_x, linear_equation(m, plot_x, c), c="orange", label="m={}, c={}, \nerr={} rss={}".format(m,c,error,rss_error))
    plt.legend()
    plt.show()
        
    return [m, c, error, rss_error]

# <<< Output >>

# plot(data,initial_m,initial_b)
[m, b, error, rss] = calculate_m_b_with_gradient_descent(data, learning_rate, num_iterations, initial_m, initial_c)
# compute_rss(data, 1000, 170)
# plot(data, 7092, 26550)
# print(linear_equation(7092, 10,26550))
