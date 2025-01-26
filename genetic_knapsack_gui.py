import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from genetic_knapsack import GeneticKnapsack, Item
import random
from typing import List

class KnapsackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Knapsack Solver")
        self.root.geometry("800x600")
        
        # Create main frames
        self.input_frame = ttk.LabelFrame(root, text="Parameters", padding="10")
        self.input_frame.pack(fill="x", padx=5, pady=5)
        
        self.items_frame = ttk.LabelFrame(root, text="Items", padding="10")
        self.items_frame.pack(fill="x", padx=5, pady=5)
        
        self.results_frame = ttk.LabelFrame(root, text="Results", padding="10")
        self.results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self._create_input_widgets()
        self._create_items_section()
        self._create_results_section()
        
    def _create_input_widgets(self):
        # Parameters inputs
        params = [
            ("Capacity:", "capacity", 20),
            ("Population Size:", "population_size", 100),
            ("Generations:", "generations", 100),
            ("Mutation Rate:", "mutation_rate", 0.1),
            ("Tournament Size:", "tournament_size", 3)
        ]
        
        self.param_vars = {}
        for i, (label, var_name, default) in enumerate(params):
            ttk.Label(self.input_frame, text=label).grid(row=i, column=0, padx=5, pady=2)
            var = tk.StringVar(value=str(default))
            self.param_vars[var_name] = var
            ttk.Entry(self.input_frame, textvariable=var).grid(row=i, column=1, padx=5, pady=2)
            
        ttk.Button(self.input_frame, text="Solve", command=self.solve).grid(row=len(params), column=0, columnspan=2, pady=10)
        
    def _create_items_section(self):
        # Buttons frame
        button_frame = ttk.Frame(self.items_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(button_frame, text="Add Item", command=self.add_item).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Remove Item", command=self.remove_item).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Generate Random", command=self.show_random_dialog).pack(side="left", padx=5)
        
        # Tree view setup
        self.items_tree = ttk.Treeview(self.items_frame, columns=("Weight", "Value"), show="headings")
        self.items_tree.heading("Weight", text="Weight")
        self.items_tree.heading("Value", text="Value")
        self.items_tree.pack(fill="x", pady=5)
        
    def _create_results_section(self):
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.results_frame)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        
        # Results text
        self.results_text = tk.Text(self.results_frame, height=10, width=50)
        self.results_text.pack(side="right", fill="y")
        
    def add_item(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Item")
        
        weight_var = tk.StringVar()
        value_var = tk.StringVar()
        
        ttk.Label(dialog, text="Weight:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(dialog, textvariable=weight_var, justify="center").grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Value:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(dialog, textvariable=value_var, justify="center").grid(row=1, column=1, padx=5, pady=5)
        
        def save():
            try:
                weight = float(weight_var.get())
                value = float(value_var.get())
                self.items_tree.insert("", "end", values=(f"{weight:.2f}", f"{value:.2f}"))
                dialog.destroy()
            except ValueError:
                tk.messagebox.showerror("Error", "Invalid input")
                
        ttk.Button(dialog, text="Save", command=save).grid(row=2, column=0, columnspan=2, pady=10)
        
    def remove_item(self):
        selected = self.items_tree.selection()
        for item in selected:
            self.items_tree.delete(item)
            
    def show_random_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Generate Random Items")
        dialog.geometry("300x400")
        dialog.resizable(False, False)
        
        # Input variables
        vars = {
            'count': tk.StringVar(value='5'),
            'min_weight': tk.StringVar(value='1'),
            'max_weight': tk.StringVar(value='10'),
            'min_value': tk.StringVar(value='1'),
            'max_value': tk.StringVar(value='20')
        }
        
        # Create input fields
        fields = [
            ("Number of items:", 'count'),
            ("Minimum weight:", 'min_weight'),
            ("Maximum weight:", 'max_weight'),
            ("Minimum value:", 'min_value'),
            ("Maximum value:", 'max_value')
        ]
        
        for i, (label, var_name) in enumerate(fields):
            ttk.Label(dialog, text=label).pack(pady=(10,0))
            ttk.Entry(dialog, textvariable=vars[var_name], justify='center').pack()

        def generate_items():
            try:
                count = int(vars['count'].get())
                min_w = float(vars['min_weight'].get())
                max_w = float(vars['max_weight'].get())
                min_v = float(vars['min_value'].get())
                max_v = float(vars['max_value'].get())
                
                if count <= 0 or min_w >= max_w or min_v >= max_v:
                    raise ValueError("Invalid input ranges")
                
                for _ in range(count):
                    weight = round(random.uniform(min_w, max_w), 2)
                    value = round(random.uniform(min_v, max_v), 2)
                    self.items_tree.insert("", "end", values=(f"{weight:.2f}", f"{value:.2f}"))
                
                dialog.destroy()
                messagebox.showinfo("Success", f"Generated {count} random items")
                
            except ValueError as e:
                messagebox.showerror("Error", str(e))
        
        ttk.Button(dialog, text="Generate", command=generate_items).pack(pady=20)
        
    def solve(self):
        # Get parameters
        try:
            params = {name: float(var.get()) for name, var in self.param_vars.items()}
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid parameter values")
            return
            
        # Get items
        items = []
        for item in self.items_tree.get_children():
            weight, value = self.items_tree.item(item)["values"]
            items.append(Item(float(weight), float(value)))
            
        if not items:
            tk.messagebox.showerror("Error", "No items added")
            return
            
        # Solve
        solver = GeneticKnapsack(
            items=items,
            capacity=params["capacity"],
            population_size=int(params["population_size"]),
            generations=int(params["generations"]),
            mutation_rate=params["mutation_rate"],
            tournament_size=int(params["tournament_size"])
        )
        
        solution, fitness = solver.solve()
        
        # Update results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Best fitness: {fitness}\n\nSelected items:\n")
        for i, selected in enumerate(solution):
            if selected:
                self.results_text.insert(tk.END, 
                    f"Item {i}: Weight={items[i].weight}, Value={items[i].value}\n")
                    
        # Update plot
        self.ax.clear()
        self.ax.plot(solver.best_fitness_history)
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Best Fitness")
        self.ax.set_title("Fitness History")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = KnapsackGUI(root)
    root.mainloop()