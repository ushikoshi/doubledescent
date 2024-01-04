import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('mystyle.sty')


def get_quantiles(xaxis, r, quantileslower=0.05, quantilesupper=0.95):
    new_xaxis, inverse, counts = np.unique(xaxis, return_inverse=True, return_counts=True)

    r_values = np.zeros([len(new_xaxis), max(counts)])
    secondindex = np.zeros(len(new_xaxis), dtype=int)
    for n in range(len(xaxis)):
        i = inverse[n]
        j = secondindex[i]
        r_values[i, j] = r[n]
        secondindex[i] += 1
    m = np.median(r_values, axis=1)
    lerr = m - np.quantile(r_values, quantileslower, axis=1)
    uerr = np.quantile(r_values, quantilesupper, axis=1) - m
    return new_xaxis, m, lerr, uerr


def plot_errorbar(ax, x, y, lerr, uerr, color, lbl=''):
    l, = ax.plot(x, y, '-o', ms=4, color=color, label=lbl)
    ax.plot(x, y - lerr, alpha=0.6, color=l.get_color() )
    ax.plot(x, y + uerr, alpha=0.6, color=l.get_color())
    ax.fill_between(x.astype(float), y - lerr, y + uerr, alpha=0.3, color=l.get_color())


# retrieving results dataframe
raw_data = pd.read_parquet('results_df_jul.parquet')

# plotting using Antonio's code
# plt.style.use('mystyle.mplsty')
print("\n\nPlotting double descent curve with boxplots...\n")
plt.close('all')
fig, ax = plt.subplots(figsize=(19, 9))
ridge_grid = [1e-5, 1e-7]
colors = ['red', 'blue']
for i, ridge in enumerate(ridge_grid):
    df = raw_data[raw_data['ridge'] == ridge].copy().reset_index(drop=True)
    df['proportion'] = df['n_features']/755
    new_xaxis, m_test, lerr_test, uerr_test = get_quantiles(df['proportion'], df['mse_test'])
    plot_errorbar(ax, new_xaxis, m_test, lerr_test, uerr_test, color=colors[i], lbl=str(ridge))
    ax.axvline(1, ls='--', color='black')
    ax.set_xscale('log')
    # ax.set_ylim(0, 5)
    # ax.set_xlabel('N / train_len')
    ax.set_ylabel('MSE')
    plt.legend()
plt.tight_layout()
plt.savefig('teste.pdf')
# plt.show()

print('Breakpoint')



