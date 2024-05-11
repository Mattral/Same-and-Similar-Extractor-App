import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import time  # Used to simulate delay for demonstration

class DataComparerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Comparison Tool")

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 10), padding=6)
        style.configure('TLabel', font=('Helvetica', 12))
        style.configure('TFrame', background='light gray')
        style.configure('Horizontal.TProgressbar', background='green')

        # Setup frames
        self.frame = ttk.Frame(self.root, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Widgets
        self.load_button_1 = ttk.Button(self.frame, text="Load Warehouse Data", command=lambda: self.load_data(1))
        self.load_button_1.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.load_button_2 = ttk.Button(self.frame, text="Load Industry Data", command=lambda: self.load_data(2))
        self.load_button_2.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.column_dropdown_1 = ttk.Combobox(self.frame, state="readonly", width=30)
        self.column_dropdown_1.grid(row=1, column=0, padx=5, pady=5)

        self.column_dropdown_2 = ttk.Combobox(self.frame, state="readonly", width=30)
        self.column_dropdown_2.grid(row=1, column=1, padx=5, pady=5)

        self.compare_button = ttk.Button(self.frame, text="Compare", command=self.compare_data)
        self.compare_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.progress = ttk.Progressbar(self.frame, style='Horizontal.TProgressbar', length=200, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=2, pady=10, sticky='ew')

        self.result_frame = ttk.LabelFrame(self.root, text="Results", padding=5)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_tree = ttk.Treeview(self.result_frame, columns=("Warehouse", "Industry"), show="headings")
        self.result_tree.heading("Warehouse", text="Warehouse Data")
        self.result_tree.heading("Industry", text="Industry Data")
        self.result_tree.pack(fill=tk.BOTH, expand=True)

        # Data
        self.data1 = None
        self.data2 = None

    def load_data(self, file_number):
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")])
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')  # Customize as needed
                elif file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    raise ValueError("Unsupported file format")

                if file_number == 1:
                    self.data1 = df
                    self.column_dropdown_1['values'] = list(df.columns)
                    self.column_dropdown_1.set('')
                else:
                    self.data2 = df
                    self.column_dropdown_2['values'] = list(df.columns)
                    self.column_dropdown_2.set('')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load the file: {e}")

 
    def compare_data(self):
        column1 = self.column_dropdown_1.get()
        column2 = self.column_dropdown_2.get()
        if self.data1 is not None and self.data2 is not None and column1 and column2:
            df1_col = self.data1[column1].astype(str).str.strip()
            df2_col = self.data2[column2].astype(str).str.strip()

            common = df1_col[df1_col.isin(df2_col)].unique()

            self.result_tree.delete(*self.result_tree.get_children())  # Clear previous results
            self.progress['value'] = 0
            self.progress['maximum'] = len(common)

            for item in common:
                matches_in_df1 = df1_col[df1_col == item].index
                matches_in_df2 = df2_col[df2_col == item].index
                for match_index in matches_in_df1:
                    for index in matches_in_df2:
                        self.result_tree.insert("", "end", values=(
                            f"Row {match_index + 2}: {item}",  # +2 to account for zero-index and header
                            f"Row {index + 2}: {item}"
                        ))
                self.progress['value'] += 1
                self.root.update_idletasks()
                time.sleep(0.05)  # Simulate delay for demonstration purposes

            if len(common) == 0:
                messagebox.showinfo("No Matches", "No exact matches found between the selected columns.")


if __name__ == "__main__":
    root = tk.Tk()
    app = DataComparerApp(root)
    root.mainloop()

