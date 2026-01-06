from logistic_regression import logistic_regression
import matplotlib.pyplot as plt


if __name__ == "__main__":
    lr = logistic_regression("datasets/2.csv")
    lr.learning_rate = 0.1
    lr.max_epoch = 3000
    lr.batch = 1
    lr.train()
    lr.save()

    fig, ax = plt.subplots(figsize=(10, 6))

    for house, losses in lr._loss_history.items():
        ax.plot(losses, label=house)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log Loss")
    ax.set_title("Logistic Regression Training Loss per House")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    # plt.show()