#!/usr/bin/env python3
"""
Multi-Period (52-week) Budget Allocation Optimization with Seasonal Risk/Return,
Live Updating via Callback and Tkinter GUI in Dark Mode.

Flow:
  [Initial Screen] --button clicked--> [Initializing Solver... Screen]
       --build_problem() completes--> [Waiting for iteration step Screen]
       --First iteration callback--> [Live Dashboard]
       --Final data available--> [Final Top 10 Investments with green bars]
       
Note: No artificial delays are added.
"""

import clarabel
import numpy as np
import threading
import tkinter as tk
import sys
from scipy import sparse
from scipy.sparse import block_diag, vstack

# -----------------------------------------------------------------------------
# Model Class
# -----------------------------------------------------------------------------
class BudgetOptimizationModel:
    """Encapsulates the creation of the optimization problem and investment names."""
    def __init__(self, N=400, weeks=52, B_total=1e6, gamma=0.80):
        self.N = N
        self.weeks = weeks
        self.B_total = B_total
        self.gamma = gamma
        self.investment_names = []
        self.phases = None
        self.P_total = None
        self.q_total = None
        self.A_total = None
        self.b_total = None
        self.cones = None
        self.r_weeks = None
        self.P_weeks = None

    @staticmethod
    def get_season_name(phase):
        phase = phase % (2 * np.pi)
        if 0 <= phase < np.pi / 2:
            return "Spring"
        elif np.pi / 2 <= phase < np.pi:
            return "Summer"
        elif np.pi <= phase < 3 * np.pi / 2:
            return "Autumn"
        else:
            return "Winter"

    def generate_investment_names(self, phases):
        names = []
        counts = {}
        for p in phases:
            base = self.get_season_name(p)
            cnt = counts.get(base, 0) + 1
            counts[base] = cnt
            if cnt == 1:
                names.append(base)
            else:
                names.append(f"{base} {cnt}")
        self.investment_names = names

    def build_problem(self):
        N, weeks, B_total, gamma = self.N, self.weeks, self.B_total, self.gamma
        total_vars = weeks * N
        r_weeks = []
        q_list = []
        P_weeks = []

        # Generate seasonal parameters.
        self.phases = np.random.uniform(0, 2 * np.pi, size=N)
        amplitude = np.random.uniform(0.05, 0.2, size=N)
        base_r = np.random.uniform(0.05, 0.30, N)

        for w in range(weeks):
            seasonal_factor = np.sin(2 * np.pi * ((w + 1) / weeks) + self.phases)
            r_w = base_r * (1 + amplitude * seasonal_factor) + np.random.normal(0, 0.005, size=N)
            r_weeks.append(r_w)
            q_list.append(-r_w)

            M = np.random.randn(N, N)
            D = np.diag(np.logspace(-3, 3, N))
            P_w_dense = M.T @ M + D
            P_w = sparse.csc_matrix(P_w_dense)
            P_weeks.append(P_w)

        q_total = np.concatenate(q_list)
        P_total = block_diag(P_weeks, format="csc")

        # Construct constraint matrices.
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
        A_turnover_lower = sparse.coo_matrix((data_ll, (rows_ll, cols_ll)),
                                             shape=((weeks - 1) * N, total_vars)).tocsc()
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
        A_turnover_upper = sparse.coo_matrix((data_lu, (rows_lu, cols_lu)),
                                             shape=((weeks - 1) * N, total_vars)).tocsc()
        b_turnover_upper = np.zeros((weeks - 1) * N)

        A_total = vstack([A_budget, A_nonneg, A_turnover_lower, A_turnover_upper],
                         format="csc")
        b_total = np.concatenate([b_budget, b_nonneg, b_turnover_lower, b_turnover_upper])

        cone_budget = clarabel.ZeroConeT(1)
        cone_nonneg = clarabel.NonnegativeConeT(total_vars)
        cone_turnover_lower = clarabel.NonnegativeConeT((weeks - 1) * N)
        cone_turnover_upper = clarabel.NonnegativeConeT((weeks - 1) * N)
        cones = [cone_budget, cone_nonneg, cone_turnover_lower, cone_turnover_upper]

        # Store the built problem.
        self.P_total = P_total
        self.q_total = q_total
        self.A_total = A_total
        self.b_total = b_total
        self.cones = cones
        self.r_weeks = r_weeks
        self.P_weeks = P_weeks

        # Generate investment names.
        self.generate_investment_names(self.phases)

# -----------------------------------------------------------------------------
# Solver Class
# -----------------------------------------------------------------------------
class BudgetOptimizerSolver:
    """
    Runs the optimization solver in a separate thread and handles iteration callbacks.
    """
    def __init__(self, model: BudgetOptimizationModel, max_iter=1000):
        self.model = model
        self.iteration_counter = 0
        self.iteration_data = []  # List of tuples: (iteration, names, allocations)
        self.final_data = None
        self.solver_finished = False
        self.callback_lock = threading.Lock()
        self.update_ui_callback = None  # To be set by the dashboard
        self.initialized = False  # Will be set to True once build_problem() finishes

        self.settings = clarabel.DefaultSettings()
        self.settings.verbose = True
        self.settings.max_iter = max_iter
        self.settings.on_iteration = self.callback

    def callback(self, variables, iter):
        """
        Called at each iteration of the solver.
        Appends the top 10 investments and immediately triggers a UI update.
        """
        with self.callback_lock:
            self.iteration_counter = iter
            x = variables[0]
            weeks, N = self.model.weeks, self.model.N
            allocation_matrix = np.array(x).reshape((weeks, N))
            total_allocation = allocation_matrix.sum(axis=0)
            top_indices = np.argsort(total_allocation)[::-1][:10]
            top_names = [self.model.investment_names[i] for i in top_indices]
            top_allocations = [total_allocation[i] for i in top_indices]
            self.iteration_data.append((iter, top_names, top_allocations))
        if self.update_ui_callback:
            self.update_ui_callback()

    def run_solver(self):
        """
        Runs the solver and computes the final top-10 investments.
        """
        P_total = self.model.P_total
        q_total = self.model.q_total
        A_total = self.model.A_total
        b_total = self.model.b_total
        cones = self.model.cones
        weeks, N = self.model.weeks, self.model.N

        result = clarabel.DefaultSolver(P_total, q_total, A_total, b_total, cones, self.settings).solve()
        self.solver_finished = True

        try:
            x_opt = result.x
        except AttributeError:
            x_opt = None

        if x_opt is None:
            self.final_data = ("No solution", [], [])
            return

        allocation_matrix = np.array(x_opt).reshape((weeks, N))
        total_allocation = allocation_matrix.sum(axis=0)
        top_indices = np.argsort(total_allocation)[::-1][:10]
        final_names = [self.model.investment_names[i] for i in top_indices]
        final_allocations = [total_allocation[i] for i in top_indices]
        self.final_data = (self.iteration_counter, final_names, final_allocations)
        self.iteration_data.append(self.final_data)
        if self.update_ui_callback:
            self.update_ui_callback()

    def start(self):
        """Starts the solver in a daemon thread and marks it as initialized."""
        self.initialized = True
        if self.update_ui_callback:
            self.update_ui_callback()
        solver_thread = threading.Thread(target=self.run_solver, daemon=True)
        solver_thread.start()
        return solver_thread

# -----------------------------------------------------------------------------
# Tkinter Dashboard Class (Dark Mode)
# -----------------------------------------------------------------------------
class TkinterDashboard:
    """
    A dark-themed dashboard that displays a live-updating bar chart of the top 10 investments.
    
    Flow:
      - Immediately after the start screen is hidden, the dashboard window appears.
      - Initially:
            * If the model has investment names available (i.e. build_problem() finished),
              it shows the first 10 investments with zero allocations (gray bars) and
              the title "Waiting for iteration step".
            * Otherwise, it displays "Initializing Solver..." in the center.
      - Once iteration callbacks arrive, the chart updates.
      - When final data is available, the bars turn green and the title becomes "Final Top 10 Investments".
      - If the user is on the final screen but then navigates away, keyboard navigation is enabled.
    """
    def __init__(self, solver: BudgetOptimizerSolver, width=800, height=600):
        self.solver = solver
        self.width = width
        self.height = height
        self.current_index = 0
        self.last_data_count = 0  # Tracks previous length of iteration_data
        self.force_final = True  # Force final view if solver finished and user hasn't navigated

        self.root = tk.Tk()
        self.root.title("Budget Allocation Optimization")
        self.root.configure(bg="#1e1e1e")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height,
                                bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack()

        self.root.bind("<Key>", self.on_key)
        self.root.bind("<<UpdateChart>>", lambda event: self.draw_chart())
        self.solver.update_ui_callback = lambda: self.root.event_generate("<<UpdateChart>>")
        
        # Immediately draw the initial state.
        self.draw_chart()

    def on_close(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        sys.exit(0)

    def draw_chart(self):
        try:
            self.canvas.delete("all")
        except Exception:
            return
        data_list = self.solver.iteration_data

        # If build_problem is not finished, show "Initializing Solver..."
        if not self.solver.initialized:
            title = "Initializing Solver..."
            self.canvas.create_text(self.width/2, self.height/2,
                                     text=title, fill="lightgray", font=("Arial", 16))
            self.canvas.create_text(self.width/2, self.height-20,
                                     text="Use Left/Right arrow keys to navigate iterations. Press 'Escape' to exit.",
                                     fill="lightgray", font=("Arial", 10))
            return
        else:
            # Build finished. If no iteration data has arrived yet, show waiting state.
            if not data_list:
                names = self.solver.model.investment_names[:10] if self.solver.model.investment_names else []
                allocations = [0] * len(names)
                title = "Waiting for iteration step"
                bar_fill = "#555555"
            else:
                # Iteration data is available.
                if self.solver.solver_finished and self.force_final:
                    self.current_index = len(data_list) - 1
                elif self.current_index == self.last_data_count - 1:
                    self.current_index = len(data_list) - 1
                self.last_data_count = len(data_list)
                if self.current_index == len(data_list) - 1 and self.solver.final_data is not None:
                    iter_num, names, allocations = self.solver.final_data
                    title = "Final Top 10 Investments"
                    bar_fill = "green"
                else:
                    iter_num, names, allocations = data_list[self.current_index]
                    title = f"Top 10 Investments at Iteration {iter_num}/{self.solver.iteration_counter}"
                    bar_fill = "#007acc"

        self.canvas.create_text(self.width / 2, self.height - 20,
            text="Use Left/Right arrow keys to navigate iterations. Press 'Escape' to exit.",
            fill="lightgray", font=("Arial", 10))

        if names:
            margin_left = 50
            margin_right = 50
            margin_top = 50
            margin_bottom = 80
            num_bars = 10
            bar_space = (self.width - margin_left - margin_right) / num_bars
            max_value = max(allocations) if allocations else 1
            scale = (self.height - margin_top - margin_bottom) / (max_value if max_value != 0 else 1)

            for i in range(num_bars):
                x0 = margin_left + i * bar_space + 10
                x1 = margin_left + (i + 1) * bar_space - 10
                y1 = self.height - margin_bottom
                y0 = y1 - allocations[i] * scale
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=bar_fill, outline="white")
                self.canvas.create_text((x0 + x1) / 2, y0 - 10,
                                         text=f"{allocations[i]:.2f}", font=("Arial", 10), fill="white")
                self.canvas.create_text((x0 + x1) / 2, y1 + 10,
                                         text=names[i], font=("Arial", 10), anchor="n", fill="white")
            self.canvas.create_text(self.width / 2, margin_top / 2,
                                     text=title, font=("Arial", 14, "bold"), fill="white")
            if self.current_index > 0:
                prev_iter = data_list[self.current_index - 1][0]
                self.canvas.create_text(10, margin_top / 2,
                                         text=f"< Previous: Iteration {prev_iter}",
                                         font=("Arial", 10), anchor="w", fill="lightgray")
            if self.current_index < len(data_list) - 1:
                if self.current_index == len(data_list) - 2 and self.solver.solver_finished:
                    next_text = "Next: Final >"
                else:
                    next_iter = data_list[self.current_index + 1][0]
                    next_text = f"Next: Iteration {next_iter} >"
                self.canvas.create_text(self.width - 10, margin_top / 2,
                                         text=next_text, font=("Arial", 10), anchor="e", fill="lightgray")
        else:
            self.canvas.create_text(self.width / 2, self.height - 20,
                                     text="Use Left/Right arrow keys to navigate iterations. Press 'Escape' to exit.",
                                     fill="lightgray", font=("Arial", 10))

    def on_key(self, event):
        # Cancel force_final if user navigates.
        self.force_final = False
        if event.keysym == "Right":
            self.current_index = min(self.current_index + 1, len(self.solver.iteration_data) - 1)
        elif event.keysym == "Left":
            self.current_index = max(self.current_index - 1, 0)
        elif event.keysym == "Escape":
            self.on_close()
            return
        try:
            self.draw_chart()
        except Exception:
            pass

    def start(self):
        self.root.mainloop()

# -----------------------------------------------------------------------------
# Start Screen Class (Dark Mode)
# -----------------------------------------------------------------------------
class StartScreen:
    """
    A dark-themed start screen with a button to initiate the solver.
    When the button is pressed, the start screen is replaced with the dashboard.
    """
    def __init__(self, model: BudgetOptimizationModel, solver: BudgetOptimizerSolver, width=800, height=600):
        self.model = model
        self.solver = solver
        self.width = width
        self.height = height
        self.root = tk.Tk()
        self.root.title("Budget Allocation Optimization - Start")
        self.root.configure(bg="#1e1e1e")
        self.frame = tk.Frame(self.root, width=self.width, height=self.height, bg="#1e1e1e")
        self.frame.pack_propagate(False)
        self.frame.pack()

        self.label = tk.Label(self.frame,
                              text="Welcome to Budget Allocation Optimization",
                              font=("Arial", 16), bg="#1e1e1e", fg="white")
        self.label.pack(pady=20)

        self.start_button = tk.Button(self.frame,
                                      text="Start Solver",
                                      font=("Arial", 14),
                                      command=self.start_solver,
                                      relief=tk.FLAT,
                                      highlightthickness=0,
                                      bd=0)
        self.start_button.pack(pady=20)

    def start_solver(self):
        # Build the optimization problem and start the solver in a background thread.
        def build_and_start():
            self.model.build_problem()
            self.solver.start()
        threading.Thread(target=build_and_start, daemon=True).start()
        # Immediately hide the start screen.
        self.root.withdraw()
        # Launch the dashboard immediately.
        dashboard = TkinterDashboard(self.solver, width=self.width, height=self.height)
        dashboard.start()

    def start(self):
        self.root.mainloop()

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    np.random.seed(42)
    model = BudgetOptimizationModel(N=400, weeks=52, B_total=1e6, gamma=0.80)
    solver = BudgetOptimizerSolver(model, max_iter=1000)
    start_screen = StartScreen(model, solver, width=800, height=600)
    start_screen.start()

if __name__ == "__main__":
    main()
