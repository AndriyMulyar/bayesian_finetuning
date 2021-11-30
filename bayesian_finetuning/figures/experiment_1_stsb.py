import json
import matplotlib.pyplot as plt
import pandas
import numpy as np
from scipy.interpolate import splrep

finetuned_data = json.load(open('/home/andriy/PycharmProjects/bayesian_finetuning/bayesian_finetuning/bayesian_finetuning/figures/data/finetuned_stsb.json'))
random_init_data = json.load(open('/home/andriy/PycharmProjects/bayesian_finetuning/bayesian_finetuning/bayesian_finetuning/figures/data/random_init_stsb.json'))

frames_finetuned = [(key, pandas.DataFrame(points)) for key, points in finetuned_data.items()]
frames_random_init = [(key, pandas.DataFrame(points)) for key, points in random_init_data.items()]

fig = plt.figure()
ax1 = fig.add_subplot(111)

value = 'val_loss' # or 'val_loss'
for key, df in frames_finetuned:
    ax1.plot(df['distance'][:120], df[value][:120], c='b')

for key, df in frames_random_init:
    ax1.plot(df['distance'][:120], df[value][:120], c='g')



plt.legend(loc='upper right')
plt.xlabel("Distance from MAP solution")
plt.ylabel(value.capitalize())

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),Line2D([0], [0], color='g', lw=4, label='Line')]

ax1.legend(legend_elements, ['Finetuned', 'Random Initialization'])
plt.show()