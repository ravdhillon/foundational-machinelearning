from math import sqrt

p1 = [2,3]
p2 = [3,5]

# This is not the fast way of calculating the distance
euclidean_distance = sqrt( (p1[0]-p2[0])**2 + (p1[1] - p2[1])**2 )
print(euclidean_distance)