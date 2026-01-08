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
    disp.plot(ax=ax, cmap="Reds", colorbar=True)

    ax.set_title("Confusion Matrix")
    fig.tight_layout()


def plot_loss_history(loss_history, log_interval):
    fig, ax = plt.subplots(figsize=(10, 6))

    for house, losses in loss_history.items():
        epochs = [log_interval * i for i in range(len(losses))]
        ax.plot(epochs, losses, label=house, color=house_color_map.get(house))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log Loss")
    ax.set_title("Logistic Regression Training Loss per House")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()


def setupArgs():
    parser = argparse.ArgumentParser(description=("House prediction\n"))

    parser.add_argument(
        "dataset_path",
        type=str
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1420
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
        choices=["batch_GD", "mini_batch_GD", "stochastic_GD", "momentum_GD"]
    )
    parser.add_argument(
        "--save_fig",
        type=bool,
        default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = setupArgs()
        lr = logistic_regression(args.dataset_path)

        lr.batch = {
            "batch_GD": 0,
            "mini_batch_GD": 142,
            "stochastic_GD": 1,
            "momentum_GD": 0,
        }[args.optimization]
        lr.useStochastic = args.optimization == "stochastic_GD"
        lr.useMomentum = args.optimization == "momentum_GD"
        lr.max_epoch = args.epochs
        lr.learning_rate = args.learning_rate
        lr.logging_interval = int(max(lr.batch / 300, 1))

        lr.train()

        print({
            "batch_GD": "Batch Gradient Decent",
            "mini_batch_GD": "Mini-batch Gradient Decent",
            "stochastic_GD": "Stochastic Gradient Decent",
            "momentum_GD": "Momentum Gradient Decent",
        }[args.optimization], "- model finished training")

        lr.save()

        plot_loss_history(lr._loss_history, lr.logging_interval)
        if args.save_fig:
            plt.savefig("log_loss")

        plot_confusion_matrix(lr.predict(), lr._houses_data, lr._houses)
        if args.save_fig:
            plt.savefig("confusion_matrix")

        if not args.save_fig:
            plt.show()

    except Exception as error:
        print(f"Error: {error}")
        exit(1)
