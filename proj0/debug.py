#!python

import math

def get_sum_metrics(predictions, metrics=None):
    if metrics is None:
        metrics = []
    for i in range(3):  
        metrics.append(lambda x, i=i: x + i) 

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return sum_metrics


p = 3
m = [lambda y: y**2]

print(get_sum_metrics(p, m))
print(get_sum_metrics(2, m))
