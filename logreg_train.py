from logistic_regression import logistic_regression
import matplotlib.pyplot as plt
from describe import house_color_map
import argparse

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=("House prediction\n"))

        parser.add_argument(
            "dataset_path",
            type=str
        )
        parser.add_argument(
            "--optimization",
            type=str,
            default="batch_GD",
            choices=["batch_GD", "mini_batch_GD", "stochastic_GD", "dropout_GD"]
        )
        args = parser.parse_args()

        lr = logistic_regression(args.dataset_path)
        lr.learning_rate = 1
        lr.max_epoch = 100
        lr.batch = 0
        lr.train()
        lr.save()

        fig, ax = plt.subplots(figsize=(10, 6))

        for house, losses in lr._loss_history.items():
            ax.plot(losses, label=house, color=house_color_map[house])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log Loss")
        ax.set_title("Logistic Regression Training Loss per House")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        plt.show()
    except Exception as error:
        print(f"Error: {error}")
        exit(1)
