#Write a python program to perform Linear classification using AND and OR logic.

#Code:

from random import choice
from numpy import array, dot, random


def unit_step(x): return 0 if x < 0.5 else 1


training_data = [
    (array([0, 0, 1]), 0),
    (array([0, 1, 1]), 1),
    (array([1, 0, 1]), 1),
    (array([1, 1, 1]), 1),
]
w = random.rand(3)
errors = []
n = 100

try:
    xrange
except NameError:
    xrange = range

for i in xrange(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += error * x
for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))
