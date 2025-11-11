import torch
import numpy as np
import matplotlib.pyplot as plt

# Number of body parts
V = 10 

# Mouse
body_parts = {0: "nose", 1: "left ear", 2: "right ear", 3: "left front paw", 4: "right front paw", 
              5: "left hind paw", 6: "right hind paw", 7: "tail tip", 8: "head center", 9: "tail base"} 
edges = ["01", "02", "81", "82", "83", "84", "89", "95", "96", "97"]

def paired_euclidean_distances(tensor1, tensor2):
    return(torch.norm(tensor1 - tensor2, dim=-1))

def body(tensor, n_batch, t):
    """
    Returns the body parts' coordinates and the skeleton of the mouse for one instant.

    Parameters
    ----------
    tensor : array-like of shape (N, T, V, C)
        The ground-truth tracks.
    n_batch : int
        The index (in the batch of sequences) of the sequence to retain.
    t : int
        The index (in the retained sequence) of the instant.
        
    Returns
    ----------
    parts : tuple
        The body parts' coordinates.
    skeleton : tuple
        The skeleton of the mouse consists in a collection of the mouse's edges in the form ([x[i], x[j]], [y[i], y[j]], [z[i], z[j]]) 
        for i and j two linked body parts.
    """
    # Parts
    nose = tensor[n_batch, t, 0, :]
    l_ear = tensor[n_batch, t, 1, :]
    r_ear = tensor[n_batch, t, 2, :]
    l_f_paw = tensor[n_batch, t, 3, :]
    r_f_paw = tensor[n_batch, t, 4, :]
    l_h_paw = tensor[n_batch, t, 5, :]
    r_h_paw = tensor[n_batch, t, 6, :]
    tail_tip = tensor[n_batch, t, 7, :]
    head_center = tensor[n_batch, t, 8, :]
    tail_base = tensor[n_batch, t, 9, :]

    parts = (nose, l_ear, r_ear, l_f_paw, r_f_paw, l_h_paw, r_h_paw, tail_tip, head_center, tail_base)

    # Skeleton
    d_01 = ([nose[0], l_ear[0]], [nose[1], l_ear[1]], [nose[2], l_ear[2]])
    d_02 = ([nose[0], r_ear[0]], [nose[1], r_ear[1]], [nose[2], r_ear[2]])
    d_81 = ([head_center[0], l_ear[0]], [head_center[1], l_ear[1]], [head_center[2], l_ear[2]])
    d_82 = ([head_center[0], r_ear[0]], [head_center[1], r_ear[1]], [head_center[2], r_ear[2]])
    d_83 = ([head_center[0], l_f_paw[0]], [head_center[1], l_f_paw[1]], [head_center[2], l_f_paw[2]])
    d_84 = ([head_center[0], r_f_paw[0]], [head_center[1], r_f_paw[1]], [head_center[2], r_f_paw[2]])
    d_89 = ([head_center[0], tail_base[0]], [head_center[1], tail_base[1]], [head_center[2], tail_base[2]])
    d_95 = ([tail_base[0], l_h_paw[0]], [tail_base[1], l_h_paw[1]], [tail_base[2], l_h_paw[2]])
    d_96 = ([tail_base[0], r_h_paw[0]], [tail_base[1], r_h_paw[1]], [tail_base[2], r_h_paw[2]])
    d_97 = ([tail_base[0], tail_tip[0]], [tail_base[1], tail_tip[1]], [tail_base[2], tail_tip[2]])

    skeleton = (d_01, d_02, d_81, d_82, d_83, d_84, d_89, d_95, d_96, d_97)
    
    return parts, skeleton


def plot_tracks(tensor_gt, tensor_pred, n_batch, fig_name=None):
    """
    Plots every body part's tracks individually over one sequence.

    Parameters
    ----------
    tensor_gt : array-like of shape (N, T, V, C)
        The ground-truth tracks.
    tensor_pred : array-like of shape (N, T, V, C)
        The predicted tracks.
    n_batch : int
        The index (in the batch of sequences) of the sequence to plot.
    fig_name : str, optional
        The filename to use when saving the figures as SVG files (one file per body part). If None, the images are not saved.
    """
    for i in range(V):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='3d')

        # Ground truth
        track = tensor_gt[n_batch, :, i, :] 
        ax.plot(track[:, 0], track[:, 1], track[:, 2], color ="red", label="Ground truth")

        # Prediction
        track = tensor_pred[n_batch, :, i, :]
        ax.plot(track[:, 0], track[:, 1], track[:, 2],color ="blue", alpha=0.7, label="Prediction")

        # PLot's parameters
        plt.title(body_parts[i])
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(0, 50)
        ax.grid(True)
        plt.tight_layout()
        
        plt.show()

        if fig_name is not None:
            # Save as SVG file
            plt.savefig(".\{}-{}.svg".format(fig_name, body_parts[i]))


def plot_mouse(parts, skeleton, bounds=None, z_diff=None, fig_name=None):
    """
    Plots the mouse's parts and skeleton from a top-side view.

    Parameters
    ----------
    parts : tuple
        The body parts' coordinates.
    skeleton : tuple
        The skeleton of the mouse consists in a collection of the mouse's edges in the form ([x[i], x[j]], [y[i], y[j]], [z[i], z[j]]) 
        for i and j two linked body parts.
    bounds : tuple, optional
        The X-axis and Y-axis bounds within which the figure is plotted. If None, those bounds are calculated to be adjusted to the plotted elements. 
    z_diff : float, optional
        The average difference between the ground-truth and predicted cooridnates on the Z-axis to display on the figure.
    fig_name : str, optional
        The filename to use when saving the figure as an SVG image. If None, the image is not saved.
        
    Returns
    ----------
    bounds : tuple, optional
        The bounds of the figure adjusted to the plotted elements.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    
    display = []

    # Initialize bounds
    if bounds is None:
        x_low, x_high = parts[0][0], parts[0][0]
        y_low, y_high = parts[0][1], parts[0][1]

    # Display body parts with colors
    for i, part in enumerate(parts):
        if i in [0, 7]: # nose and tail tip
            color = "y"
        if i in [1, 3, 5]: # left side
            color = "r"
        if i in [2, 4, 6]: # right side
            color = "g"
        if i in [8, 9]: # head center and tail base
            color = "grey"

        if np.any(np.asarray(part) >= 0): # a not-masked body part has all coordinates >= 0
            display += [i] # save index of displayed body part
            ax.scatter(*part, color=color, s=100)

            # Adjust bounds
            if bounds is None:
                if part[0] < x_low:
                    x_low = part[0]

                if part[0] > x_high:
                    x_high = part[0]
    
                if part[1] < y_low:
                    y_low = part[1]
    
                if part[1] > y_high:
                    y_high = part[1]
                
    s = dict(zip(edges, skeleton))
    
    for k in edges:
        i, j = k[0], k[1]

        if int(i) in display and int(j) in display: # if both edge's parts are displayed
            ax.plot(*s[k], color='k', linewidth=2)

    # Plot's parameters
    ax.view_init(azim=-90, elev=90)  # elevation and azimuth
    ax.set_xlabel("X", fontsize=17)
    ax.set_ylabel("Y", fontsize=17)
    ax.set_zticks([])  
    ax.grid(True)

    if bounds is not None:
        x_low, x_high, y_low, y_high = bounds

    # Plot's bounds
    ax.set_xlim(x_low - 20, x_high + 20)
    ax.set_ylim(y_low - 20, y_high + 20)
    ax.set_zlim(0, 50)        

    if z_diff is not None:
        ax.set_title("Z-diff: {}mm".format(round(z_diff, 2)), fontsize=25)
        ax.title.set_position((0.5, 0.1)) 
    
    if fig_name is not None:
        # Save as SVG file
        plt.savefig(".\{}.svg".format(fig_name))
        
    if bounds is None:
        return (x_low, x_high, y_low, y_high)