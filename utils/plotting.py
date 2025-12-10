import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # Needed for 3D projection
import os

def plot_3d_surface(
    f,
    domain,
    title="3D Surface Plot",
    best_points=None,
    filename=None,
    grid_size=80
):
    """
    Plot a 3D surface for f(x1, x2) with optional points on top.

    Parameters
    ----------
    f : callable
        Objective function f(x1, x2).
    domain : list of tuples
        Domain for each variable: [(x1_min, x1_max), (x2_min, x2_max)].
    title : str
        Title of the plot.
    best_points : list of tuples or np.ndarray
        Points to highlight, e.g. [(x1, x2, f), ...].
    filename : str
        If provided, saves the plot to results/plots/filename.
    grid_size : int
        Resolution of the grid for the surface.
    """

    # Create meshgrid in the domain
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]

    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Create 3D figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=14)

    # Plot surface
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.6,
                    cmap="viridis", edgecolor='none')

    # Plot best points if provided
    if best_points is not None:
        best_points = np.array(best_points)
        ax.scatter(
            best_points[:, 0],
            best_points[:, 1],
            best_points[:, 2],
            c='r',
            s=60,
            label='Solutions'
        )
        ax.legend()

    # Axes labels
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("f(x₁, x₂)")

    # Save file if requested
    if filename:
        filepath = f"results/plots/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[plotting] Saved plot to {filepath}")

    plt.close(fig)  # Close figure to avoid memory buildup

def plot_3d_route(points, route, title="TSP route", filename=None):
    """
    Plot a 3D route given `points` (Nx3 array) and `route` (sequence of indices).
    Draws lines following the route and closing the route back to the origin.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pts = np.array(points)
    route = list(route)

    # Build full route with return to start
    route_full = route + [route[0]]

    X = pts[ [route_full], : ][0][:, 0]
    Y = pts[ [route_full], : ][0][:, 1]
    Z = pts[ [route_full], : ][0][:, 2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, linestyle='-', linewidth=1.5, marker='o', markersize=4)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=20, c='k', alpha=0.6)

    # annotate order optionally (commented out by default to avoid clutter)
    # for i, idx in enumerate(route):
    #     ax.text(pts[idx,0], pts[idx,1], pts[idx,2], str(i), color='red')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if filename:
        filepath = f"results/plots/{filename}"
        os.makedirs("results/plots", exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[plotting] Saved TSP route plot to {filepath}")

    plt.close(fig)
