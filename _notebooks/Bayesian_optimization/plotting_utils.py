import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def illustrate_1d_gpr(f, gpr, x_plot, EI, LCB, num_sample_points=None):
    x_sample = gpr.X_train_
    y_sample = gpr.y_train_
    (mean, stdev) = gpr.predict(x_plot, return_std = True)
    mean = mean.flatten()
    
    fig = plt.figure(figsize = (8, 9))
    ax1 = fig.add_subplot(311)
    ax1.plot(x_plot, f(x_plot), "k-", label = "Ground Truth", zorder = 0)
    ax1.plot(x_plot, mean, "r--", label = "Predicted Mean", zorder = 1)

    if num_sample_points == None:
        ax1.plot(x_sample, y_sample, "ro", label = "Initial Sampled Points", zorder = 2)
    else:
        ax1.plot(x_sample[:num_sample_points], y_sample[:num_sample_points], "ro", label = "Initial Sampled Points", zorder = 2)
        ax1.plot(x_sample[num_sample_points:], y_sample[num_sample_points:], "bo", label = "New Sampled Points", zorder = 2)

    ax1.fill_between(x_plot.flatten(), mean - 1.96*stdev, mean + 1.96*stdev, color = "r", alpha = 0.25)
    ax1.set_xlabel(r"$x$", fontsize = 15)
    ax1.set_ylabel(r"$f(x)$", fontsize = 15)
    ax1.set_title("Gaussian Process Regression", fontsize = 15)
    ax1.legend(fontsize = 10)
    ax1.grid(True)
    
    ax2 = fig.add_subplot(312)
    ax2.plot(x_plot, EI(x_plot, gpr), "k-")
    ax2.set_xlabel(r"$x$", fontsize = 15)
    ax2.set_ylabel(r"$EI(x)$", fontsize = 15)
    ax2.set_title(r"Expected Improvement, $\delta=%.2f$" % EI.params["delta"], fontsize = 15)
    ax2.grid(True)
    
    ax3 = fig.add_subplot(313)
    ax3.plot(x_plot, LCB(x_plot, gpr), "k-")
    ax3.set_xlabel(r"$x$", fontsize = 15)
    ax3.set_ylabel(r"$LCB(x)$", fontsize = 15)
    ax3.set_title(r"Lower Confidence Bound, $\sigma=%.2f$" % LCB.params["sigma"], fontsize = 15)
    ax3.grid(True)
    plt.tight_layout()
    
    return (fig, (ax1, ax2, ax3))

def main(*args, **kwargs):
    pass

if __name__ == "__main__":
    main()