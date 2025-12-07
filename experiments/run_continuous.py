from utils.plotting import plot_3d_surface
from problems.continuous.problem1 import f, domain

def run_problem1_visualization():
    best_points = []  # fill with (x1, x2, f)

    # Example point
    x1, x2 = 1.2, -0.4
    best_points.append((x1, x2, f(x1, x2)))

    plot_3d_surface(
        f=f,
        domain=domain,
        title="Problem 1 - Visualization",
        best_points=best_points,
        filename="problem1_surface.png"
    )
    print("3D surface plot for Problem 1 saved as 'problem1_surface.png'")