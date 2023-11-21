import csv
import tkinter as tk
from tkinter import filedialog, font
import argparse

class CsvEditor:
    def __init__(self, root, start_row, label_column_name, category_column_name):
        self.root = root
        self.current_row = start_row
        self.label_column_name = label_column_name
        self.category_column_name = category_column_name
        self.root.title("CSV Editor")

        # Define font style and size
        self.custom_font = font.Font(family="Helvetica", size=28)

        # Load CSV data
        self.filename = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        with open(self.filename, newline='') as file:
            self.data = list(csv.DictReader(file))
            self.headers = self.data[0].keys()

        # GUI Components
        self.label = tk.Label(root, text=self.data[self.current_row]['statement'], font=self.custom_font, wraplength= 1200)
        self.label.pack(pady=20)

        self.textbox_label = tk.Label(root, text="Enter Category:", font=self.custom_font)
        self.textbox_label.pack(pady=10)
        # Entry widget
        self.textbox = tk.Entry(root, font=self.custom_font)
        self.textbox.pack(pady=10)
        self.textbox.bind('<Return>', self.on_enter)

        self.button_p = tk.Button(root, text="P", command=lambda: self.update_csv("P"), width=10, height=2)
        self.button_p.pack(side=tk.LEFT, padx=10)

        self.button_i = tk.Button(root, text="I", command=lambda: self.update_csv("I"), width=10, height=2)
        self.button_i.pack(side=tk.LEFT, padx=10)

        self.button_h = tk.Button(root, text="H", command=lambda: self.update_csv("H"), width=10, height=2)
        self.button_h.pack(side=tk.LEFT, padx=10)

        # Save button
        self.save_button = tk.Button(root, text="Save", command=self.save_changes, width=10, height=2)
        self.save_button.pack(pady=20)

    def on_enter(self, event):
        # Write content of the textbox into the specified column
        self.data[self.current_row][self.category_column_name] = self.textbox.get()
        # Save changes to the CSV
        self.save_changes()
        # Advance to the next row
        self.advance_row()

    def update_csv(self, value):
        # Update the specified column with the value P, I, or H
        self.data[self.current_row][self.label_column_name] = value

    def advance_row(self):
        self.current_row += 1
        if self.current_row < len(self.data):
            self.label.config(text=self.data[self.current_row]['statement'])
            self.textbox.delete(0, tk.END)
        else:
            self.root.quit()

    def save_changes(self):
        with open(self.filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(self.data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit a CSV file with a GUI.")
    parser.add_argument('--start-row', type=int, default=0, help='Row number to start with (0-indexed).')
    parser.add_argument('--label-column-name', default='new_possibility_yury', help='Name of the label column.')
    parser.add_argument('--category-column-name', default='category_yury', help='Name of the category column.')

    args = parser.parse_args()
    root = tk.Tk()
    app = CsvEditor(root, start_row=args.start_row, label_column_name=args.label_column_name, category_column_name=args.category_column_name)
    root.mainloop()
