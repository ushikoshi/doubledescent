import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import json
from reservoir.model import ESN
from tqdm import tqdm
plt.style.use('mystyle.sty')


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%.2f"


def embedding(x, delay):
    # x must be a 3d np.array with trajectory values in the 1st dimension
    # this function returns a 3d np.array containing both x and the delayed values concatenated along the 3rd dimension
    if delay < 0:
        delay = -1 * delay
        x_delayed = x[:-delay, ...]
        y = np.concatenate((x[delay:, ...], x_delayed), axis=2)
    else:
        x_delayed = x[delay:, ...]
        y = np.concatenate((x[:-delay, ...], x_delayed), axis=2)
    return y


def plot_poincare_section(y, period, id='unnamed', n_points=40, phase=0, marker_size=0.25):
    y = y[phase::period, ...]
    plt.scatter(y[-n_points:, :, 0], y[-n_points:, :, 1], color='blue', s=20 * 2 ** marker_size)
    x_min = min(y[-n_points:, :, 0]) - 0.1
    x_max = max(y[-n_points:, :, 0]) + 0.1
    y_min = min(y[-n_points:, :, 1]) - 0.1
    y_max = max(y[-n_points:, :, 1]) + 0.1
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(f"model_{id}.eps")
    plt.savefig(f"model_{id}.png")
    plt.close()


def simulate_model(raw_data, choice, u, y):
    mdl = raw_data['model'].loc[choice]
    mdl_dict = json.loads(mdl)
    mdl = ESN.fromdict(mdl_dict)
    y_sim_test = mdl.simulate(u=u, y=y, burnout=raw_data['burnout_train'].iloc[choice], output_burnout=False, one_step_ahead=False)
    return y_sim_test


# retrieving results dataframe
raw_data = pd.read_parquet('results_df_jul.parquet')

# computing period used for Poincare section sampling
dt_original = np.pi / 600
decimate = 10
dt_decimated = dt_original * decimate
poincare_period = int(2 * np.pi / dt_decimated)

# plotting the Poincare section for a few choices of model
over = raw_data[raw_data['n_features'] > 955]
# over['n_features'].loc[over['mse_test'].nsmallest(50).index.tolist()]
choices = [2544, 5981]  # [5826, 6238, 7490, 6959, 6079]  # [6785, 5404, 5981, 6110, 6315, 6791]
# choices = over['mse_test'].nsmallest(100).index.tolist()[8:]
for choice in tqdm(choices, desc='Processing'):
    y_validation = np.load('y_test_validation_5000.npy')[-840000:, ...]
    u_validation = np.load('u_test_validation_5000.npy')[-840000:, ...]
    y_embedd = embedding(simulate_model(raw_data, choice, u=u_validation, y=y_validation), delay=4)  # creating an embedded space
    plot_poincare_section(y=y_embedd, id=choice, period=poincare_period, n_points=5000, phase=15, marker_size=0.01)  # plotting the Poincare section for the selected model

print('Breakpoint')



