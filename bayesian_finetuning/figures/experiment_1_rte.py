import json
import matplotlib.pyplot as plt
import pandas
import numpy as np
from scipy.interpolate import splrep

finetuned_data = json.load(open('/home/andriy/PycharmProjects/bayesian_finetuning/bayesian_finetuning/bayesian_finetuning/figures/data/finetuned_rte.json'))
random_init_data = json.load(open('/home/andriy/PycharmProjects/bayesian_finetuning/bayesian_finetuning/bayesian_finetuning/figures/data/random_init_rte.json'))

frames_finetuned = [(key, pandas.DataFrame(points)) for key, points in finetuned_data.items()]
frames_random_init = [(key, pandas.DataFrame(points)) for key, points in random_init_data.items()]

fig = plt.figure()
ax1 = fig.add_subplot(111)

value = 'val_loss' # or 'val_loss'
for key, df in frames_finetuned:
    ax1.plot(df['distance'], df[value], c='b', label=f"finetuned_{key}")

for key, df in frames_random_init:
    ax1.plot(df['distance'], df[value], c='g', label=f"rand_{key}")



plt.legend(loc='upper right')
plt.title("Validation ")
plt.xlabel("Distance along random ray from MAP solution")
plt.ylabel(value.capitalize())
plt.show()