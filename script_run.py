from reservoir.model import ESN
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
plt.style.use('mystyle.sty')


def train_model(hp, u_train, y_train, burnout_train, state_transform='quadratic'):

    mdl = ESN(nus=[[1]], nys=[[1]], n_features=hp['n_features'], burnout_train=burnout_train,
              input_scale=hp['input_scale'], density=hp['density'],
              spectral_radius=hp['spectral_radius'], state_transform=state_transform,
              use_iterative=False, ridge=hp['ridge'], random_state=hp['seed'],
              leaky_factor=hp['leaky_factor'])  # initializing model
    
    mdl.fit(u_train, y_train)  # training model

    return mdl


def mape(y, y_hat):
    return float((100 * sum(abs(y[:, 0, 0] - y_hat[:, 0, 0]))) / (len(y) * abs(max(y) - min(y))))


def mse(y_true, y_pred):
    """Mean square error between samples."""
    n = y_pred.shape[0]
    return np.mean((y_true[-n:, ...] - y_pred) ** 2)


def run_mc(hyperparameters, u_train, y_train, u_test, y_test, burnout_train, burnout_simulate, osa=False, state_transform='quadratic'):
    
    df = pd.DataFrame(columns=['n_features', 'spectral_radius', 'leaky_factor', 'ridge', 'input_scale', 'density', 'seed', 'burnout_train', 'burnout_simulate',
                               'state_transform', 'model', 'mape_train', 'mse_train', 'mape_test', 'mse_test'])

    for hp in tqdm(hyperparameters, desc="Processing"):
        row = list(hp.values())

        # model training for each seed
        mdl = train_model(hp, u_train=u_train, y_train=y_train, state_transform=state_transform, burnout_train=burnout_train)
        # model testing for each seed
        y_sim_test = mdl.simulate(u_test, y_test, burnout=burnout_simulate, output_burnout=False, one_step_ahead=osa)
        # appending model object
        row.append(mdl)
        # appending training metrics
        row.append(
            mape(y_train[mdl.order + mdl.burnout_train:, ...], mdl.y_sim_train))
        row.append(
            mse(y_train[mdl.order + mdl.burnout_train:, ...], mdl.y_sim_train))
        # appending testing metrics
        if osa:
            y_test_new = y_test[mdl.burnout_train:-1, ...]
        else:
            y_test_new = y_test[mdl.burnout_train:, ...]
        row.append(mape(y_test_new, y_sim_test))
        row.append(mse(y_test_new, y_sim_test))

        df.loc[len(df)] = row
        df.loc[len(df)-1, 'seed'] = mdl.random_state

    df_complete = model_to_dict(df)

    return df_complete


def list_to_dict(hp_lists):
    
    hp_dicts = [dict(hp) for hp in hp_lists]
    return hp_dicts


def model_to_dict(raw_data):
    keys_to_remove = ['nys', 'nus']
    models_dict = raw_data['model'].copy()
    models_dict = [model.__dict__ for model in models_dict]

    for model in models_dict:
        model['n_features'] = model.pop('n_features')
        model['burnout_train'] = model.pop('burnout_train')
        model['w_xy'] = model['w_xy']

    models_json = [json.dumps(model) for model in models_dict]

    raw_data['model'] = models_json
    return raw_data


if __name__ == '__main__':

    # Define the range and number of points
    start_value = 75
    end_value = 75500
    num_points = 50

    # Generate logarithmically spaced values
    n_features_grid = np.logspace(np.log10(start_value), np.log10(end_value), num=num_points, dtype=int)

    # # defining the n_features grid
    # features_aux = [int(round(i)) for i in np.geomspace(1, 100000, 199)]
    # features_aux.append(955)
    # aux = []
    # for x in features_aux:
    #     if 50 <= x <= 4000:
    #         aux.append(x)
    # n_features_grid = list(set(aux))
    n_features_grid.sort()

    # defining other hyperparameters grid
    ridge_grid = [1e-5, 1e-7, 1e-10, 1e-12]
    spectral_radius_grid = [0.1]
    input_scale_grid = [0.1]
    leaky_factor_grid = [0.3]
    density_grid = [0.05]
    
    # defining seeds (for Monte Carlo on the initialization)
    seed_grid = list(range(1, 101))

    # defining settings
    burnout_train = 200
    burnout_simulate = 200
    state_transform = 'quadratic'
    
    # listing hyperparameter combinations
    hp_lists = list(product(n_features_grid, spectral_radius_grid, leaky_factor_grid, ridge_grid, input_scale_grid, density_grid, seed_grid, [burnout_train],
                            [burnout_simulate], [state_transform]))

    # gathering list of hyperparameters dictionaries
    keys = ['n_features', 'spectral_radius', 'leaky_factor', 'ridge', 'input_scale', 'density', 'seed', 'burnout_train', 'burnout_simulate', 'state_transform']
    hp_dicts = [dict(zip(keys, hp)) for hp in hp_lists]

    # defining training dataset
    y_train_complete = np.load('y_train.npy')
    # y_train = y_train_complete[:955, ...]  # taking only the first half as the training series
    # std_dev = 0.05 * y_train.std()  # computing 5% of the standar deviation of the training series
    # gaussian_noise = np.random.normal(loc=0, scale=std_dev, size=y_train.shape)  # computing gaussian noise
    # y_train = y_train + gaussian_noise  # adding noise to the output measurement
    # np.save('y_train_noise2.npy', y_train)
    y_train = np.load('y_train_noise.npy')
    u_train_complete = np.load('u_train.npy')
    u_train = u_train_complete[:955, ...]  # taking only the first half as the training series

    # defining testing dataset
    y_test = y_train_complete[955:, ...]  # taking the second half as the testing series
    u_test = u_train_complete[955:, ...]  # taking the second half as the testing series

    results_df = run_mc(hyperparameters=hp_dicts, u_train=u_train, y_train=y_train, u_test=u_test, y_test=y_test,
                                 burnout_train=burnout_train, burnout_simulate=burnout_simulate, osa=False, state_transform='quadratic')
    
    results_df.to_parquet('results_df.parquet')

    print('Debug')
