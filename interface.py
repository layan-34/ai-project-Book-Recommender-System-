import tkinter as tk
from tkinter import messagebox
from ANN import evaluate_metrics_dynamic_thresholds , test_data, user_mapping, isbn_mapping ,model# Import from your ANN.py
from KNN import calculate_metrics , X_test_filtered, y_test_filtered, model_knn, train_pivot, ratings  # Import from your KNN.py
from mainApp import outputs
# Initialize the main application window
root = tk.Tk()
root.title("Book Recommendation System")
root.geometry("800x600")
root.configure(bg="#e8f1f2")


def get_recommendations_and_metrics(algorithm="KNN"):
    user_id = user_id_entry.get()

    if not user_id:
        messagebox.showwarning("Error", "Please fill in the User ID field.")
        return


    '''if user_id not in X_test_filtered['userID'] :
        messagebox.showwarning("Error", "user ID not found.")
        return'''




    try:
        user_id = int(user_id)

        if algorithm == "KNN":
            metrics = calculate_metrics(user_id, X_test_filtered, y_test_filtered, model_knn, train_pivot, ratings)
        elif algorithm == "ANN":
            metrics = evaluate_metrics_dynamic_thresholds(user_id,model, test_data, user_mapping, isbn_mapping)
        elif algorithm == "Clustering":
            metrics=outputs(user_id)
        else:
            messagebox.showerror("Error", f"Algorithm '{algorithm}' not implemented.")
            return

        # Update recommendations
        recommendations_list.delete(0, tk.END)  # Clear previous recommendations
        for book in metrics["recommended_books"]:
            recommendations_list.insert(tk.END, book)

        # Update metrics
        metrics_list.delete(0, tk.END)
        metrics_display = [
            f"Accuracy: {metrics['Accuracy']:.2f}",
            f"Precision: {metrics['Precision']:.2f}",
            f"recall: {metrics['recall']:.2f}",
            f"F1_Score: {metrics['F1_Score']:.2f}",
        ]
        for metric in metrics_display:
            metrics_list.insert(tk.END, metric)

    except ValueError:
        messagebox.showerror("Error", "Invalid User ID. Please enter a numeric value.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create a frame to hold content
container = tk.Frame(root, bg="#ffffff", padx=20, pady=20, relief="raised", bd=2)
container.pack(pady=20, padx=20, fill="both", expand=True)

# Add a title label
title = tk.Label(container, text="Book Recommendation System", font=("Helvetica", 20, "bold"), bg="#ffffff", fg="#2b6777")
title.pack(pady=10)

# User ID input section
user_id_label = tk.Label(container, text="User ID:", font=("Helvetica", 12), bg="#ffffff", fg="#2b6777")
user_id_label.pack(anchor="w", pady=(15, 5))

user_id_entry = tk.Entry(container, font=("Helvetica", 12), bd=2, relief="solid", width=50)
user_id_entry.pack()

# Algorithm selection buttons stacked vertically
algo_buttons_frame = tk.Frame(container, bg="#ffffff")
algo_buttons_frame.pack(pady=10)

algo1_button = tk.Button(
    algo_buttons_frame,
    text="Get recommendations using kNN",
    bg="#4a7c94",
    fg="#ffffff",
    font=("Helvetica", 12),
    relief="flat",
    width=30,
    command=lambda: get_recommendations_and_metrics("KNN")
)
algo1_button.pack(pady=5)

algo2_button = tk.Button(
    algo_buttons_frame,
    text="Get recommendations using ANN",
    bg="#4a7c94",
    fg="#ffffff",
    font=("Helvetica", 12),
    relief="flat",
    width=30,
    command=lambda: get_recommendations_and_metrics("ANN")
)
algo2_button.pack(pady=5)

algo3_button = tk.Button(
    algo_buttons_frame,
    text="Get recommendations using Clustering",
    bg="#4a7c94",
    fg="#ffffff",
    font=("Helvetica", 12),
    relief="flat",
    width=30,
    command=lambda: get_recommendations_and_metrics("Clustering")
)
algo3_button.pack(pady=5)

# Recommended Books Listbox
recommendations_label = tk.Label(container, text="Recommended Books:", font=("Helvetica", 14, "bold"), bg="#ffffff", fg="#2b6777")
recommendations_label.pack(anchor="w", pady=(15, 5))

recommendations_list = tk.Listbox(container, font=("Helvetica", 12), height=5, width=80, bd=2, relief="solid")
recommendations_list.pack(pady=(0, 10))

# Metrics Display
metrics_label = tk.Label(container, text="Model Metrics:", font=("Helvetica", 14, "bold"), bg="#ffffff", fg="#2b6777")
metrics_label.pack(anchor="w", pady=(15, 5))

metrics_list = tk.Listbox(container, font=("Helvetica", 12), height=5, width=60, bd=2, relief="solid")
metrics_list.pack()

# Run the application
root.mainloop()
