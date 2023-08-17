import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import json
from reservoir.model import ESN
from tqdm import tqdm


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
    x_min = min(y[-n_points:, :, 0]) - 1
    x_max = max(y[-n_points:, :, 0]) + 1
    y_min = min(y[-n_points:, :, 1]) - 1
    y_max = max(y[-n_points:, :, 1]) + 1
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(f"model_{id}.png")
    plt.close()



def plot_alpha_ridge_heatmap_poincares(df, input_scale, density, seed_values, subplots_rows, subplots_cols,
                                       y_test, u_test, sim_settings, path, save_name=None, axlim=None,
                                       number_of_points=500):
    df = df[(df['input_scale'] == input_scale) & (df['density'] == density)]
    df = df[df['seed'].isin(seed_values)]
    df_filtered = df.copy()
    for ind, seed in enumerate(seed_values):
        fig, ax = plt.subplots(subplots_rows, subplots_cols, figsize=(38, 18))
        df = df_filtered[df_filtered['seed'] == seed]
        df = df[['ridge', 'leaky_factor', 'mape_testing', 'model']]
        df = df.sort_values(["leaky_factor", "ridge"], ascending=[False, True])
        for iterate in range(0, subplots_rows * subplots_cols):
            print(iterate)
            mdl = df['model'].iloc[iterate]
            mdl_dict = json.loads(mdl)
            mdl = ESN.fromdict(mdl_dict)
            y_sim_test = mdl.simulate(u=u_test, y=y_test, burnout=sim_settings['burnout_simulate'],
                                      output_burnout=False,
                                      one_step_ahead=sim_settings['osa'])
            phase = 107
            position = np.unravel_index(iterate, (subplots_rows, subplots_cols))
            ax[position] = plot_poincare_section(x=y_sim_test[:, ...],
                                                 number_of_points=number_of_points,
                                                 phase=107,
                                                 save_name=f'{ind:d}_{position[0]}_{position[1]}_{phase:d}_{save_name}.jpg',
                                                 sim_settings=sim_settings,
                                                 delay=sim_settings['embedd_delay'],
                                                 path=path,
                                                 marker_size=0.1,
                                                 ax=ax[position],
                                                 axlim=None,
                                                 heatmap=True)
        plt.tight_layout()
        if save_name is not None:
            new_save = f'seed{seed}_{save_name}'
            plt.savefig(os.path.join(path, new_save))
        plt.close()
    return None


def simulate_model(raw_data, choice, u, y):
    mdl = raw_data['model'].iloc[choice]
    mdl_dict = json.loads(mdl)
    mdl = ESN.fromdict(mdl_dict)
    y_sim_test = mdl.simulate(u=u, y=y, burnout=raw_data['burnout_train'].iloc[choice], output_burnout=False, one_step_ahead=False)
    return y_sim_test


# retrieving results dataframe
raw_data = pd.read_parquet('results_df.parquet')

# computing period used for Poincare section sampling
dt_original = np.pi / 600
decimate = 10
dt_decimated = dt_original * decimate
poincare_period = int(2 * np.pi / dt_decimated)

# testing best phase for Poincare sections
phases = range(1, 120)
for phase in tqdm(phases, desc="Processing"):
    choice = 297
    y_validation = np.load('y_test_validation_5000.npy')[-120000:, ...]
    u_validation = np.load('u_test_validation_5000.npy')[-120000:, ...]
    y_embedd = embedding(simulate_model(raw_data, choice, u=u_validation, y=y_validation), delay=4)  # creating an embedded space
    plot_poincare_section(y=y_embedd, id=f"297_{phase}", period=poincare_period, n_points=500, phase=phase, marker_size=0.25)  # plotting the Poincare section for the selected model

print('Breakpoint')



