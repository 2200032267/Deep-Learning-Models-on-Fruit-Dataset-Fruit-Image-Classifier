from flask import Flask, render_template, request
from model import (
    train_simple_nn,
    experiment_2_data_loading_and_preprocessing,
    train_sequential_nn,
    train_sequential_nn_with_optimizers,
    experiment_5_random_mini_batch,
)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    experiment_number = int(request.form['experiment'])
    experiment_results = {}

    if experiment_number == 1:
        results = train_simple_nn()
        experiment_results = {
            "accuracy": results.get("accuracy", "N/A"),
            "loss_over_epochs": results.get("loss_over_epochs", []),
            "training_time": results.get("training_time", "N/A"),
            "sample_predictions": results.get("sample_predictions", []),
            "confusion_matrix": results.get("confusion_matrix", []),
            "classification_report": results.get("classification_report", {})
        }
        experiment_name = "Simple Neural Network"

    elif experiment_number == 2:
        results = experiment_2_data_loading_and_preprocessing()
        experiment_results = {
            "dataset_shape": results.get("dataset_shape", "N/A"),
            "sample_rows": results.get("sample_rows", {}),
            "normalized_values": results.get("normalized_values", []),
            "encoded_labels": results.get("encoded_labels", []),
            "training_data_shape": results.get("training_data_shape", "N/A"),
            "testing_data_shape": results.get("testing_data_shape", "N/A"),
        }
        experiment_name = "Data Loading & Preprocessing"

    elif experiment_number == 3:
        results = train_sequential_nn()
        experiment_results = {
            "accuracy": results.get("accuracy"),
            "confusion_matrix": results.get("confusion_matrix"),
            "classification_report": results.get("classification_report"),
            "train_losses": results.get("train_losses"),
            "loss_plot_url": results.get("loss_plot_url"),
            "cm_plot_url": results.get("cm_plot_url"),
        }
        experiment_name = "Sequential Model Classification"

    elif experiment_number == 4:
        results = train_sequential_nn_with_optimizers()
        experiment_results = {
            "optimizer_results": results,
        }
        experiment_name = "Sequential Model with Multiple Optimizers"

    elif experiment_number == 5:
        results = experiment_5_random_mini_batch()
        experiment_results = {
            "epoch_losses": results.get("epoch_losses"),
            "epoch_accuracies": results.get("epoch_accuracies"),
            "mini_batch_accuracies": results.get("mini_batch_accuracies"),
            "loss_plot_url": results.get("loss_plot_url"),
            "accuracy_plot_url": results.get("accuracy_plot_url"),
            "mini_batch_accuracy_plot_url": results.get("mini_batch_accuracy_plot_url"),
            "decision_boundary_plot_url": results.get("decision_boundary_plot_url"),
        }
        experiment_name = "Random Mini-Batch Evaluations"

    else:
        return "Invalid Experiment Number", 400

    # Pass the enumerate function to the template context
    return render_template('results.html', experiment=experiment_name, experiment_results=experiment_results, enumerate=enumerate)

if __name__ == '__main__':
    app.run(debug=True)