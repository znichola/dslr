from describe import loadData, describe
import pickle
import random
import math
from tqdm import trange

class logistic_regression:
    def __init__(self, path, model=None):
        self.path = path
        self.columns_to_drop = ['Arithmancy', 'Defense Against the Dark Arts',
                        'Transfiguration', 'Care of Magical Creatures', 'History of Magic', 
                        'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Index']
        self._normalize = []
        df = self.loadData(model)
        df = self.dropna(df) if model is None else self.fillna(df)
        self._houses_data = df.pop("Hogwarts House").to_list()
        self._data = self.dfToArray(df.transpose())
        self._subjects = list(df.keys())
        self._houses = list(set(self._houses_data))
        self._weights = [[0.0 for _ in range(len(self._subjects))] for _ in self._houses]
        self._loss_history = {h: [] for h in self._houses}
        self.learning_rate = 0.1
        self.max_epoch = 100
        self.batch = 0
        self.logging_interval = 10
        self.useStochastic = False
        self.useMomentum = False
        self.momentum = 0.9
        self._velocity = {h: [0.0 for _ in range(len(self._subjects))] for h in self._houses}

        if model:
            self._houses = model["houses"]
            self._weights = model["weights"]

# Train model

    def train(self):
        pbar = trange(self.max_epoch, desc="Training", unit="epoch")
        for ep in pbar:
            batches = self.generate_batches()
            for batch in batches:
                for i, house in enumerate(self._houses):
                    self._weights[i] = self.gradient_descent(self._weights[i], house, batch)

            if ep % self.logging_interval == 0:
                self.logProgress(pbar)


    def gradient_descent(self, weights, house_to_predict, batch):
        theta = weights
        start, stop = batch
        x__ = self._data[start : stop]
        y_ = [1 if house_to_predict == h else 0 for h in self._houses_data[start : stop]]
        alpha = self.learning_rate
        gradient = self.gradient(theta, x__, y_)

        if not self.useMomentum:
            return [t - alpha * g for t, g in zip(theta, gradient)]
        else: 
            v_prev = self._velocity[house_to_predict]
            v_new = [self.momentum * v + alpha * g for v, g in zip(v_prev, gradient)]
            self._velocity[house_to_predict] = v_new
            return [t - v for t, v in zip(theta, v_new)]


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
        return 1 / (1+ math.exp(-z))


# Data prep

    def normalizeData(self, df, model):
        subjects = df.select_dtypes(include=["number"]).columns

        df_copy = df.copy()
        xmindict = {}
        xrangedict = {}

        tmp, _ = describe(df)

        for i, course in enumerate(subjects):
            xmin = tmp[course]["Min"] if model is None else model["normalize"][i][0]
            xrange = tmp[course]["Range"] if model is None else model["normalize"][i][1]
            if model is None:
                self._normalize.append((xmin, xrange))
            xmindict[course] = xmin
            xrangedict[course] = xrange
            if xrange == 0:
                df_copy[course] = [1] * len(df_copy[course])
            else:
                df_copy[course] = [(x - xmin) / xrange for x in df_copy[course]]

        return df_copy


    def loadData(self, model):
        df = loadData(self.path)
        if df is None:
            exit(1)
        df["Hogwarts House"] = df["Hogwarts House"].fillna("No house")
        df = self.cleanUpData(df)
        df = self.normalizeData(df, model)
        return df

    def dfToArray(self, df):
        subject_dict = df.to_dict()
        for subject_key in subject_dict.keys():
            subject_dict[subject_key] = [grade for grade in subject_dict[subject_key].values()]
        return [d for d in subject_dict.values()]

    def cleanUpData(self, df):
        return df.drop(columns=self.columns_to_drop, errors="ignore")

    def dropna(self, df):
        return df.dropna()
    
    def fillna(self, df):
        return df.fillna(df.mean(numeric_only=True))


# Utils

    def generate_batches(self):
        m = len(self._data)
        b = self.batch if self.batch > 0 else m
        if self.useStochastic:
            foo = list(zip(self._data, self._houses_data))
            random.shuffle(foo)
            self._data = [ d for d, h in foo]
            self._houses_data = [ h for d, h in foo]
            return [(0, min(b, m))]
        else:
            return [(i, min(i + b, m)) for i in range(0, m, b)]


    def logProgress(self, pbar):
        predictions =  self.predict()
        accuracy = sum(c == p for c, p in zip(self._houses_data, predictions)) / len(predictions)
        pbar.set_postfix(acc=f"{accuracy:.4%}")
        for i, house in enumerate(self._houses):
            y = [1 if house == h else 0 for h in self._houses_data]
            current_loss = self.loss(self._weights[i], self._data, y)
            self._loss_history[house].append(current_loss)

    def loss(self, theta, x__, y_):
        m = len(y_)
        total = 0.0
        for x_, y in zip(x__, y_):
            h = self.hypothesis(theta, x_)
            h = min(max(h, 1e-15), 1 - 1e-15)
            total += y * math.log(h) + (1 - y) * math.log(1 - h)
        return -total / m


    def predict(self):
        predictions = []
        for i, x in enumerate(self._data):
            p = []
            for house_index, _ in enumerate(self._houses):
                p.append(self.hypothesis(self._weights[house_index], x))
            predictions.append(self._houses[p.index(max(p))])
        return predictions


    def save(self, file_path: str = "model.pkl"):
        try:
            data = {
                "weights": self._weights,
                "normalize": self._normalize,
                "houses": self._houses
            }
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Model saved to {file_path}")
        except:
            print(f"Error: Could not save model to '{file_path}'")
