import os
import matplotlib.pyplot as plt
from src.utils.model import get_unique_filename


def save_plot(df, plot_name, plot_dir):
    unique_filename=get_unique_filename(plot_name)
    path_to_plot=os.path.join(plot_dir, unique_filename)
    df.plot(figsize=(10,7))
    plt.grid(True)
    plt.savefig(path_to_plot)
    plt.show()