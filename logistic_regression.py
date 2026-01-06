from describe import loadData, describe
import pickle
import math

class logistic_regression:

    def __init__(self, path):
        self.path = path
        self.columns_to_drop = ['Arithmancy', 'Defense Against the Dark Arts',
                        'Transfiguration', 'Care of Magical Creatures', 'Flying', 
                        'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Index']
        self._normalize = []
        df = self.loadData()
        self._houses_data = df.pop("Hogwarts House")
        self._data = self.dfToArray(df.transpose())
        self._subjects = list(df.keys())
        self._houses = list(set(self._houses_data))
        self._weights = [[0 for _ in range(len(self._subjects))] for _ in self._houses]
        self._loss_history = {h: [] for h in self._houses}
        self.learning_rate = 0.1
        self.max_epoch = 100
        self.batch = 1

    def loadData(self):
        df = loadData(self.path)
        df = self.cleanUpData(df)
        df = self.normalizeData(df)
        return df
       
    def dfToArray(self, df):
        subject_dict = df.to_dict()
        for subject_key in subject_dict.keys():
            subject_dict[subject_key] = [grade for grade in subject_dict[subject_key].values()]
        return [d for d in subject_dict.values()]

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
            self._normalize.append((xmin, xrange))
            xmindict[course] = xmin
            xrangedict[course] = xrange
            if xrange == 0:
                df_copy[course] = [1] * len(df_copy[course])
            else:
                df_copy[course] = [(x - xmin) / xrange for x in df_copy[course]]

        return df_copy


    def train(self):
        for ep in range(self.max_epoch):
            for i, house in enumerate(self._houses):
                self._weights[i] = self.gradient_descent(self._weights[i], house)
            
            if ep % 10:
                predictions =  self.predict()
                print(sum([c == p for c, p in  zip(self._houses_data, predictions)])/len(predictions))


    def loss(self, theta, x__, y_):
        m = len(y_)
        total = 0.0
        for x_, y in zip(x__, y_):
            h = self.hypothesis(theta, x_)
            h = min(max(h, 1e-15), 1 - 1e-15)
            total += y * math.log(h) + (1 - y) * math.log(1 - h)

        return -total / m


    def gradient_descent(self, weights, house_to_predict):
        theta = weights
        x__ = self._data
        y_ = [1 if house_to_predict == h else 0 for h in self._houses_data]
        current_loss = self.loss(weights, x__, y_)
        self._loss_history[house_to_predict].append(current_loss)
        alpha = self.learning_rate
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

    def predict(self):
        predictions = []
        for i, x in enumerate(self._data):
            p = []
            for house_index, _ in enumerate(self._houses):
                p.append(self.hypothesis(self._weights[house_index], x))
            predictions.append(self._houses[p.index(max(p))])
        return predictions