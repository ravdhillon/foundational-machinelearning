"""
    @Ravisher Dhillon@
    Linear Regression using Stock Data.
    Using quandl python package, we can get the data for any ticker in CSV format or any other format.
"""
# import pandas as pd
# # quandl is python library where you can find the data and it will get the CSV data for you for any ticker.
# # visit www.quandl.com
# import quandl


# df = quandl.get('WIKI/GOOGL')
# print(df)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

"""
    num_data_points
    variance - how variance this dataset will be
    step
    correlation == +ve or -ve
"""
"""
How do we test our assumption about data and R-square?
1. Variance => we can change the variance and check the value of r_square. and we can visualize the scatter plot to see 
    if the variancec is small, the points are more close to each other and the value of R-Square will be much higher. 
    But if variance is high, the points are pretty scattered and the value of R-Square will be lower.
2. Correlation => if the data is correlated, then R-square has nice value, but if they are not correlated, R-sqaure will be almost 0. 
"""
def create_dataset(num_data_points, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(num_data_points):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation = 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)




def best_fit_slope_and_intercept(xs,  ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) / 
            ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) **2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

xs, ys = create_dataset(40, 40, 2, correlation='pos')

m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)

# create a line that fits the data
regression_line = [(m*x) + b for x in xs]

#Predict
predict_x = 8
predict_y = (m * predict_x) + b

r_sqaured = coefficient_of_determination(ys, regression_line)
print('R-Squared: ', r_sqaured)
# Visualize
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()