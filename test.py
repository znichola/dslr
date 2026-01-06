from describe import loadData, describe
import matplotlib.pyplot as plt
import math
import sys

class logistic_regression:

    def __init__(self, path):
        self.path = path
        self.columns_to_drop = ['Arithmancy', 'Defense Against the Dark Arts',
                        'Transfiguration', 'Care of Magical Creatures', 'Flying', 
                        'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Index']
        self.data = self.loadData()
        self.houses_data = self.data.pop("Hogwarts House")
        self.houses = list(set(self.houses_data))
        self.subjects = list(self.data.keys())
        self.dataToArray()
        self.loss_history = {h: [] for h in self.houses}
        self.learning_rate = 0.1
        self.max_epoch = 100000
        self.batch = 1

        self.weights = [[0 for _ in range(len(self.subjects))] for _ in self.houses]

        self.train()

    def loadData(self):
        df = loadData(self.path)
        df = self.cleanUpData(df)
        df = self.normalizeData(df)
        subject_dict = df.to_dict()
        for subject_key in subject_dict.keys():
            subject_dict[subject_key] = [grade for grade in subject_dict[subject_key].values()]
        return subject_dict

    def dataToArray(self):
        self.data = [d for d in self.data.values()]

    def cleanUpData(self, df):
        return df.drop(columns=self.columns_to_drop, errors="ignore").dropna()

    def normalizeData(self, df):

        subjects = df.select_dtypes(include=["number"]).columns

        df_copy = df.copy()
        xmindict = {}
        xrangedict = {}

        tmp, _ = describe(df)

        for course in subjects:
            
            xmin = tmp[course]["Min"]
            xrange = tmp[course]["Range"]
            xmindict[course] = xmin
            xrangedict[course] = xrange
            if xrange == 0:
                df_copy[course] = [1] * len(df_copy[course])
            else:
                df_copy[course] = [(x - xmin) / xrange for x in df_copy[course]]

        return df_copy


    def train(self):
        for _ in range(self.max_epoch):
            for i, house in enumerate(self.houses):
                self.weights[i] = self.gradient_descent(self.weights[i], house)
                print(self.loss_history[house][-1])
            print("-------------------------")
        self.plot_loss()
        self.save()

    def loss(self, theta, x__, y_):
        m = len(y_)
        total = 0.0
        for x_, y in zip(x__, y_):
            h = self.hypothesis(theta, x_)
            # numerical stability
            h = min(max(h, 1e-15), 1 - 1e-15)
            total += y * math.log(h) + (1 - y) * math.log(1 - h)

        return -total / m


    def gradient_descent(self, weights, house_to_predict):
        theta = weights
        x__ = self.data
        y_ = [1 if house_to_predict == h else 0 for h in self.houses_data]
        current_loss = self.loss(weights, x__, y_)
        self.loss_history[house_to_predict].append(current_loss)
        alpha = self.learning_rate
        print(house_to_predict, " ", end="")
        gradient = self.gradient(theta, x__, y_)

        return [t - alpha * g for t, g in zip(theta, gradient)]
    
    def gradient(self, theta, x__, y_):
        m = len(y_)
        grad = [0.0] * len(theta)

        for x_, y in zip(x__, y_):
            h = self.hypothesis(theta, x_)
            error = h - y
            for j in range(len(theta)):
                grad[j] += error * x_[j]
        return [g / m for g in grad]

    def hypothesis(self, theta, x_):
        z = sum(t * x for t, x in zip(theta, x_))
        return self.sigmoid(z)

    def sigmoid(self, z):
        '''Sigmoid = 1 / (1 + e-z)'''
        return 1 / (1+ math.exp(-z))

    def save(self):
        return
    
    def plot_loss(self):
        plt.figure(figsize=(10, 6))

        for house, losses in self.loss_history.items():
            plt.plot(losses, label=house)

        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.title("Logistic Regression Training Loss per House")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        



if __name__ == "__main__":
    logistic_regression("datasets/dataset_train.csv")
    # try:
    #     logistic_regression("datasets/dataset_train.csv")
    # except Exception as err:
    #     print(err)

