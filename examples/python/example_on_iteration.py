#!/usr/bin/env python3
"""
Multi-Period (52-week) Budget Allocation Optimization with Seasonal Risk/Return,
Live Updating via Callback and Matplotlib GUI.
"""

import clarabel
import numpy as np
import threading
import time
import re
from scipy import sparse
from scipy.sparse import block_diag, vstack
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import queue

# Global parameters for parsing and reshaping.
GLOBAL_WEEKS = 52
GLOBAL_N = 300

# Global counter for iterations.
iteration_counter = 0

# Global list for investment names; will be generated in main().
investment_names = []

# Global flags and storage for final solution.
solver_finished = False
final_data = None

# A thread-safe queue to pass live update data to the chart updater.
live_queue = queue.Queue()

# Global variable for the animation object.
animation_object = None

# Store all iteration data for navigation.
iteration_data = []
current_iteration_index = 0

def get_season_name(phase):
    """Map a phase (in radians) to a season name."""
    phase = phase % (2 * np.pi)
    if 0 <= phase < np.pi / 2:
        return "Spring"
    elif np.pi / 2 <= phase < np.pi:
        return "Summer"
    elif np.pi <= phase < 3 * np.pi / 2:
        return "Autumn"
    else:
        return "Winter"


def generate_investment_names(phases):
    """Generate a fixed name for each investment based on its phase.
    If a base name is already used, add a numeric suffix (e.g., 'Summer 2').
    """
    names = []
    counts = {}
    for p in phases:
        base = get_season_name(p)
        cnt = counts.get(base, 0) + 1
        counts[base] = cnt
        if cnt == 1:
            names.append(base)
        else:
            names.append(f"{base} {cnt}")
    return names


def build_problem(N, weeks, B_total, gamma, base_r):
    """
    Build the multi-period problem.
    Returns: P_total, q_total, A_total, b_total, cones, r_weeks, P_weeks, phases
    """
    total_vars = weeks * N
    r_weeks = []
    q_list = []
    P_weeks = []

    # Generate seasonal parameters for each investment.
    phases = np.random.uniform(0, 2 * np.pi, size=N)
    amplitude = np.random.uniform(0.05, 0.2, size=N)

    for w in range(weeks):
        seasonal_factor = np.sin(2 * np.pi * ((w + 1) / weeks) + phases)
        r_w = base_r * (1 + amplitude * seasonal_factor) + np.random.normal(
            0, 0.005, size=N
        )
        r_weeks.append(r_w)
        q_list.append(-r_w)

        M = np.random.randn(N, N)
        D = np.diag(np.logspace(-3, 3, N))
        P_w_dense = M.T @ M + D
        P_w = sparse.csc_matrix(P_w_dense)
        P_weeks.append(P_w)

    q_total = np.concatenate(q_list)
    P_total = block_diag(P_weeks, format="csc")

    rows_budget = 1
    rows_nonneg = total_vars
    rows_turnover = (weeks - 1) * N
    total_rows = rows_budget + rows_nonneg + 2 * rows_turnover

    A_budget = sparse.csc_matrix(np.ones((1, total_vars)))
    b_budget = np.array([B_total])

    A_nonneg = -sparse.eye(total_vars, format="csc")
    b_nonneg = np.zeros(total_vars)

    rows_ll, cols_ll, data_ll = [], [], []
    for w in range(1, weeks):
        for i in range(N):
            row = (w - 1) * N + i
            idx_w = (w - 1) * N + i
            idx_wp1 = w * N + i
            rows_ll.extend([row, row])
            cols_ll.extend([idx_w, idx_wp1])
            data_ll.extend([1 - gamma, -1.0])
    A_turnover_lower = sparse.coo_matrix(
        (data_ll, (rows_ll, cols_ll)), shape=((weeks - 1) * N, total_vars)
    ).tocsc()
    b_turnover_lower = np.zeros((weeks - 1) * N)

    rows_lu, cols_lu, data_lu = [], [], []
    for w in range(1, weeks):
        for i in range(N):
            row = (w - 1) * N + i
            idx_w = (w - 1) * N + i
            idx_wp1 = w * N + i
            rows_lu.extend([row, row])
            cols_lu.extend([idx_w, idx_wp1])
            data_lu.extend([-(1 + gamma), 1.0])
    A_turnover_upper = sparse.coo_matrix(
        (data_lu, (rows_lu, cols_lu)), shape=((weeks - 1) * N, total_vars)
    ).tocsc()
    b_turnover_upper = np.zeros((weeks - 1) * N)

    A_total = vstack(
        [A_budget, A_nonneg, A_turnover_lower, A_turnover_upper], format="csc"
    )
    b_total = np.concatenate([b_budget, b_nonneg, b_turnover_lower, b_turnover_upper])

    cone_budget = clarabel.ZeroConeT(1)
    cone_nonneg = clarabel.NonnegativeConeT(total_vars)
    cone_turnover_lower = clarabel.NonnegativeConeT((weeks - 1) * N)
    cone_turnover_upper = clarabel.NonnegativeConeT((weeks - 1) * N)
    cones = [cone_budget, cone_nonneg, cone_turnover_lower, cone_turnover_upper]

    return P_total, q_total, A_total, b_total, cones, r_weeks, P_weeks, phases


def update_chart(frame):
    """
    Update the chart based on the current iteration index.
    Use the left and right arrow keys to navigate through iterations.
    Press 'Escape' to close the chart.
    """
    global solver_finished, final_data, current_iteration_index

    ax.clear()

    # Add navigation instructions as text in the matplotlib window
    ax.text(
        0.5, -0.25,  # Adjusted padding to prevent clipping
        "Use Left/Right arrows to navigate iterations. Press 'Escape' to exit.",
        fontsize=10,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="gray",
    )

    if 0 <= current_iteration_index < len(iteration_data):
        # if last iteration, and final data is available, show final data
        if current_iteration_index == len(iteration_data) - 1 and final_data is not None:
            iter_num, final_names, final_allocs = final_data
            bars = ax.bar(final_names, final_allocs, color="lightgreen")
            ax.set_title("Final Top 10 Investments")
            ax.set_ylabel("Total Annual Allocation")
            ax.set_xlabel("Investment")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )
            # Add navigation text for previous iteration
            if len(iteration_data) > 1:
                previus_iter_num = iteration_data[current_iteration_index - 1][0]
                ax.text(
                    0.01, 1.02,
                    f"< Previous: Iteration {previus_iter_num}",
                    fontsize=10,
                    ha="left",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )
            return ax

        iter_num, top_names, top_allocations = iteration_data[current_iteration_index]
        bars = ax.bar(top_names, top_allocations, color="skyblue")
        ax.set_title(f"Top 10 Investments at Iteration {iter_num}/{iteration_counter}")
        ax.set_ylabel("Total Annual Allocation")
        ax.set_xlabel("Investment")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        # Add navigation text for previous and next iterations
        if current_iteration_index > 0:
            previus_iter_num = iteration_data[current_iteration_index - 1][0]
            ax.text(
                0.01, 1.02,
                f"< Previous: Iteration {previus_iter_num}",
                fontsize=10,
                ha="left",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )
        if current_iteration_index < len(iteration_data) - 1:
            if current_iteration_index == len(iteration_data) - 2 and solver_finished:
                next_text = "Next: Final >"
            else:
                next_iter_num = iteration_data[current_iteration_index + 1][0]
                next_text = f"Next: Iteration {next_iter_num} >"
            ax.text(
                0.99, 1.02,
                next_text,
                fontsize=10,
                ha="right",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )

    # Adjust bottom padding to prevent label clipping
    plt.subplots_adjust(bottom=0.2)
    
    # Adjust top padding to prevent navigation text clipping
    plt.subplots_adjust(top=0.9)
    return ax


def on_key(event):
    """
    Handle keyboard events to navigate through iterations.
    """
    global current_iteration_index

    if event.key == "right":
        current_iteration_index = min(current_iteration_index + 1, len(iteration_data) - 1)
    elif event.key == "left":
        current_iteration_index = max(current_iteration_index - 1, 0)
    elif event.key == "escape":
        plt.close(fig)
    update_chart(None)
    fig.canvas.draw()


def run_solver_thread(P_total, q_total, A_total, b_total, cones, settings, weeks, N):
    """
    Run the solver (blocking call) and then compute final top-10 investments.
    """
    global solver_finished, final_data, iteration_counter, iteration_data, fig,current_iteration_index
    result = clarabel.DefaultSolver(
        P_total, q_total, A_total, b_total, cones, settings
    ).solve()
    solver_finished = True
    try:
        x_opt = result.x
    except AttributeError:
        x_opt = None
    if x_opt is None:
        final_data = ("No solution", [], [])
        return
    allocation_matrix = np.array(x_opt).reshape((weeks, N))
    total_allocation_per_investment = allocation_matrix.sum(axis=0)
    top_indices = np.argsort(total_allocation_per_investment)[::-1][:10]
    final_names = []
    final_allocs = []
    for idx in top_indices:
        final_names.append(investment_names[idx])
        final_allocs.append(total_allocation_per_investment[idx])
    final_data = (iteration_counter, final_names, final_allocs)

    # Add the final iteration data to the iteration_data list for navigation
    iteration_data.append(final_data)
    
    # Set the current iteration index to the last iteration
    current_iteration_index = len(iteration_data) - 1


def main():
    global investment_names, solver_finished, final_data, animation_object

    N = 400
    weeks = 52
    B_total = 1e6
    gamma = 0.80

    global GLOBAL_WEEKS, GLOBAL_N
    GLOBAL_WEEKS = weeks
    GLOBAL_N = N

    np.random.seed(42)
    base_r = np.random.uniform(0.05, 0.30, N)

    print("Building multi-period problem with seasonal risk/return profiles...")
    P_total, q_total, A_total, b_total, cones, r_weeks, P_weeks, phases = build_problem(
        N, weeks, B_total, gamma, base_r
    )
    investment_names = generate_investment_names(phases)

    settings = clarabel.DefaultSettings()
    settings.verbose = True
    settings.max_iter = 1000

    def callback(variables, iter):
        global iteration_counter, iteration_data, current_iteration_index
        iteration_counter = iter
        
        # variables is a tuple of (x, y, z) where x is the current solution
        x = variables[0]

        total_allocation_per_investment = np.array(x).reshape((weeks, N)).sum(axis=0)
        top_indices = np.argsort(total_allocation_per_investment)[::-1][:10]
        top_names = []
        top_allocations = []
        for idx in top_indices:
            top_names.append(investment_names[idx])
            top_allocations.append(total_allocation_per_investment[idx])
        iteration_data.append((iter, top_names, top_allocations))

        # Update the current iteration index if it was on the last iteration
        if current_iteration_index == len(iteration_data) - 2:
            current_iteration_index = len(iteration_data) - 1

    settings.on_iteration = callback

    # Start the solver in a separate thread.
    solver_thread = threading.Thread(
        target=lambda: run_solver_thread(
            P_total, q_total, A_total, b_total, cones, settings, weeks, N
        ),
        daemon=True,
    )
    solver_thread.start()

    # Create a figure and axis for the chart.
    global fig, ax
    fig, ax = plt.subplots(figsize=(10, 6))  # Set larger window size

    # Connect the keyboard event handler.
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Set up matplotlib animation to update every 500 ms.
    animation_object = animation.FuncAnimation(fig, update_chart, interval=500, save_count=1000)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
