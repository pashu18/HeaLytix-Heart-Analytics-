import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from fpdf import FPDF
from PIL import Image, ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import os

# Load the dataset
file_path = 'heart.csv'  
df = pd.read_csv(file_path)

# Drop the target column to get the feature matrix X
X = df.drop('target', axis=1)
y = df['target']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier with hyperparameter tuning
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Initialize the Tkinter window
root = tk.Tk()
root.title("Heart Attack Prediction System")
root.geometry("1200x800")  # Updated size to accommodate graphs and fields side by side

# Create a frame for the header section
header_frame = tk.Frame(root, bg="#de6262")
header_frame.pack(fill="x")

# Header labels and input fields
info_frame = tk.Frame(header_frame, bg="#de6262")
info_frame.pack(side="right", padx=20)

# Registration, Date, Name, Birth Year fields
tk.Label(info_frame, text="Registration No.", bg="#de6262", fg="white").pack(anchor="w")
registration_entry = tk.Entry(info_frame)
registration_entry.pack(anchor="w")

# Auto-fill date field with current date
tk.Label(info_frame, text="Date", bg="#de6262", fg="white").pack(anchor="w")
date_entry = tk.Entry(info_frame)
date_entry.insert(0, datetime.now().strftime("%d/%m/%Y"))
date_entry.pack(anchor="w")

# Patient Name field (default to "unknown")
tk.Label(info_frame, text="Patient Name", bg="#de6262", fg="white").pack(anchor="w")
name_entry = tk.Entry(info_frame)
name_entry.insert(0, "unknown")
name_entry.pack(anchor="w")

tk.Label(info_frame, text="Birth Year", bg="#de6262", fg="white").pack(anchor="w")
birth_year_entry = tk.Entry(info_frame)
birth_year_entry.pack(anchor="w")

# Main input section with grid layout
main_frame = tk.Frame(root, bg="#f5f5f5")
main_frame.pack(fill="both", expand=True)

# Input fields on the left
input_frame = tk.Frame(main_frame, bg="#f5f5f5")
input_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nw")

fields = [
    ("Age", "Numerical (e.g., 45)"),
    ("Sex", ["Male", "Female"]),
    ("Chest Pain Type", ["Type 0", "Type 1", "Type 2", "Type 3"]),
    ("Resting Blood Pressure", "Numerical (e.g., 120)"),
    ("Serum Cholesterol", "Numerical (e.g., 200)"),
    ("Fasting Blood Sugar", ["No", "Yes"]),
    ("Resting ECG", ["Type 0", "Type 1", "Type 2"]),
    ("Max Heart Rate", "Numerical (e.g., 150)"),
    ("Exercise Induced Angina", ["No", "Yes"]),
    ("Oldpeak", "Numerical (e.g., 1.5)"),
    ("Slope", ["Type 0", "Type 1", "Type 2"]),
    ("Number of Major Vessels", ["0", "1", "2", "3"]),
    ("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"]),
]

entries = []
for field, desc in fields:
    frame = tk.Frame(input_frame, bg="#f5f5f5")
    frame.pack(fill="x", padx=5, pady=5)
    label = tk.Label(frame, text=field, bg="#f5f5f5")
    label.pack(side="left")
    
    if isinstance(desc, list):
        entry = ttk.Combobox(frame, values=desc)
        entry.set(desc[0])  # Set default value
    else:
        entry = tk.Entry(frame)
    
    entry.pack(side="left", padx=10)
    entries.append(entry)

# Function to validate input fields
def validate_inputs():
    if not registration_entry.get():
        return "Registration No. is required."
    if not date_entry.get():
        return "Date is required."
    if not name_entry.get():
        return "Patient Name is required."
    if not birth_year_entry.get():
        return "Birth Year is required."
    
    for entry in entries:
        if not entry.get():
            return f"{entry.get()} is required."
    
    return None

# Function to save patient data to CSV
def save_patient_data(user_inputs, prediction):
    filename = "patient_data.csv"
    file_exists = os.path.isfile(filename)

    # Prepare data dictionary
    data = {
        "Registration No": registration_entry.get(),
        "Date": date_entry.get(),
        "Patient Name": name_entry.get(),
        "Birth Year": birth_year_entry.get(),
        "Age": user_inputs[0],
        "Sex": user_inputs[1],
        "Chest Pain Type": user_inputs[2],
        "Resting Blood Pressure": user_inputs[3],
        "Serum Cholesterol": user_inputs[4],
        "Fasting Blood Sugar": user_inputs[5],
        "Resting ECG": user_inputs[6],
        "Max Heart Rate": user_inputs[7],
        "Exercise Induced Angina": user_inputs[8],
        "Oldpeak": user_inputs[9],
        "Slope": user_inputs[10],
        "Number of Major Vessels": user_inputs[11],
        "Thalassemia": user_inputs[12],
        "Prediction": "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
    }

    # Write data to CSV
    df = pd.DataFrame([data])
    df.to_csv(filename, mode='a', index=False, header=not file_exists)

# Function to perform prediction based on user input and update graphs
def perform_analysis():
    error_message = validate_inputs()
    if error_message:
        messagebox.showerror("Input Error", error_message)
        return

    try:
        user_inputs = []
        user_inputs.append(float(entries[0].get()))  # Age

        # Mapping categorical fields to numerical values
        sex_mapping = {"Male": 1, "Female": 0}
        cp_mapping = {"Type 0": 0, "Type 1": 1, "Type 2": 2, "Type 3": 3}
        fbs_mapping = {"No": 0, "Yes": 1}
        restecg_mapping = {"Type 0": 0, "Type 1": 1, "Type 2": 2}
        exang_mapping = {"No": 0, "Yes": 1}
        slope_mapping = {"Type 0": 0, "Type 1": 1, "Type 2": 2}
        vessels_mapping = {"0": 0, "1": 1, "2": 2, "3": 3}
        thal_mapping = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

        # Append mapped values to user_inputs list
        user_inputs.append(sex_mapping[entries[1].get()])  # Sex
        user_inputs.append(cp_mapping[entries[2].get()])  # Chest Pain Type
        user_inputs.append(float(entries[3].get()))  # Resting Blood Pressure
        user_inputs.append(float(entries[4].get()))  # Serum Cholesterol
        user_inputs.append(fbs_mapping[entries[5].get()])  # Fasting Blood Sugar
        user_inputs.append(restecg_mapping[entries[6].get()])  # Resting ECG
        user_inputs.append(float(entries[7].get()))  # Max Heart Rate
        user_inputs.append(exang_mapping[entries[8].get()])  # Exercise Induced Angina
        user_inputs.append(float(entries[9].get()))  # Oldpeak
        user_inputs.append(slope_mapping[entries[10].get()])  # Slope
        user_inputs.append(vessels_mapping[entries[11].get()])  # Number of Major Vessels
        user_inputs.append(thal_mapping[entries[12].get()])  # Thalassemia

        # Scale user input in the same way as training data
        user_inputs_scaled = scaler.transform([user_inputs])
        prediction = model.predict(user_inputs_scaled)
        result = "No Heart Disease Detected" if prediction[0] == 0 else "Heart Disease Detected"
        result_label.config(text=f"{name_entry.get()}, {result}")

        # Save patient data
        save_patient_data(user_inputs, prediction)

        # Update graphs based on the new data
        axs[0, 0].clear()
        axs[0, 0].plot([0, 1, 2, 3], [0, 1, 4, 9])
        axs[0, 0].set_title("Sample Graph 1")

        axs[0, 1].clear()
        axs[0, 1].bar(["A", "B", "C"], [10, 20, 15])
        axs[0, 1].set_title("Sample Graph 2")

        axs[1, 0].clear()
        axs[1, 0].scatter([1, 2, 3, 4], [1, 4, 9, 16])
        axs[1, 0].set_title("Sample Graph 3")

        axs[1, 1].clear()
        axs[1, 1].hist([1, 2, 2, 3, 4], bins=4)
        axs[1, 1].set_title("Sample Graph 4")

        canvas.draw()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to save the report
def save_report():
    error_message = validate_inputs()
    if error_message:
        messagebox.showerror("Input Error", error_message)
        return

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        registration_no = registration_entry.get()

        # Adding data to the PDF
        pdf.cell(200, 10, txt="Heart Attack Prediction Report", ln=True, align="C")
        pdf.ln(10)

        pdf.cell(200, 10, txt=f"Registration No: {registration_no}", ln=True)
        pdf.cell(200, 10, txt=f"Date: {date_entry.get()}", ln=True)
        pdf.cell(200, 10, txt=f"Patient Name: {name_entry.get()}", ln=True)
        pdf.cell(200, 10, txt=f"Prediction Result: {result_label.cget('text')}", ln=True)
        pdf.ln(10)

        # Save the graphs as an image
        if fig is not None:
            fig.savefig("graphs.png")
            pdf.image("graphs.png", x=10, y=pdf.get_y(), w=180)
        else:
            messagebox.showerror("Error", "Graphs are not available.")
            return

        pdf.output("Heart_Attack_Prediction_Report.pdf")
        messagebox.showinfo("Success", "Report saved successfully.")
        
    except IndexError as e:
        messagebox.showerror("Error", f"An IndexError occurred: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while saving the report: {e}")

analysis_button = tk.Button(input_frame, text="Analyze", bg="#2196F3", fg="white", command=perform_analysis)
analysis_button.pack(pady=20)

save_button = tk.Button(input_frame, text="Save Report", bg="#4CAF50", fg="white", command=save_report)
save_button.pack(pady=10)

graph_frame = tk.Frame(main_frame)
graph_frame.grid(row=0, column=1, padx=20, pady=10)

fig, axs = plt.subplots(2, 2, figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack()

result_label = tk.Label(input_frame, text="", font=("Arial", 14), bg="#f5f5f5")
result_label.pack(pady=10)

# Simple Forum Section
forum_frame = tk.Frame(root)
forum_frame.pack(fill="both", expand=True)

forum_label = tk.Label(forum_frame, text="Community Forum", font=("Arial", 20))
forum_label.pack(pady=10)

forum_text = tk.Text(forum_frame, width=80, height=20)
forum_text.pack(pady=10)

root.mainloop()
