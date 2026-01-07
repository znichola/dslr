from logistic_regression import logistic_regression
import matplotlib.pyplot as plt
from describe import house_color_map
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(predictions, true_labels, labels):
    cm = confusion_matrix(true_labels, predictions, labels=labels)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.show()


def plot_loss_history(loss_history):
    fig, ax = plt.subplots(figsize=(10, 6))

    for house, losses in loss_history.items():
        ax.plot(losses, label=house, color=house_color_map.get(house))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log Loss")
    ax.set_title("Logistic Regression Training Loss per House")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # try:
        parser = argparse.ArgumentParser(description=("House prediction\n"))

        parser.add_argument(
            "dataset_path",
            type=str
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=3000
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.42
        )
        parser.add_argument(
            "--optimization",
            type=str,
            default="batch_GD",
            choices=["batch_GD", "mini_batch_GD", "stochastic_GD", "adam"]
        )
        args = parser.parse_args()

        lr = logistic_regression(args.dataset_path)
        lr.learning_rate = 0.1
        lr.batch = {
            "batch_GD": 0,
            "mini_batch_GD": 420,
            "stochastic_GD": 120,
            "adam": 420,
        }[args.optimization]
        lr.stochastic = args.optimization == "stochastic_GD"
        lr.max_epoch = args.epochs
        lr.learning_rate = args.learning_rate

        lr.train()
        lr.save()

        plot_loss_history(lr._loss_history)
        plot_confusion_matrix(lr.predict(), lr._houses_data, lr._houses)

    # except Exception as error:
    #     print(f"Error: {error}")
    #     exit(1)
