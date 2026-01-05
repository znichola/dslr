from describe import loadData, describe
import math
import sys

class House_Prediction:

    def __init__(self, data, house):
        self.weights = [0 for _ in range(len(data.keys()))]
        self.tmp_weights = self.weights.copy()
        self.bias = 0
        self.tmp_bias = self.bias
        self.house_to_predict = house
    
    def train(self, data, house_data, batch):
        errors = []
        for i in range(batch):
            y, x = self.h(data[i])
            z = 1 if self.house_to_predict == house_data[1] else 0
            errors.append(self.calcul(x, z))
        self.tmp_bias -= (- 1 / len(errors)) * sum(errors) # ? maybe
        self.tmp_weights = [w * e for w, e in zip(self.weights, errors)]

    def calcul(self, x, z):
        return max(x, 0) - x * z + math.log(1 + math.exp(-abs(x)))

            
    def h(self, data_i):
        x = self.bias + sum(d * w for d, w in zip(data_i, self.weights))
        return self.g(x), x

    def g(self, z):
        '''Sigmoid = 1 / (1 + e-z)'''
        return 1 / (1+ math.exp(-z))

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
        self.House_Predictors = [House_Prediction(self.data, house) for house in self.houses]
        self.learning_rate = 0.1
        self.max_epoch = 10000
        self.batch = 1
        self.train()

    def isHouse(self, house, index):
        return self.houses_data[index] == house

    def loadData(self):
        df = loadData(self.path)
        df = self.cleanUpData(df)
        df = self.normalizeData(df)
        subject_dict = df.to_dict()
        for subject_key in subject_dict.keys():
            subject_dict[subject_key] = [grade for grade in subject_dict[subject_key].values()]
        return subject_dict

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
        
        epoch = 0
        while epoch < self.max_epoch:
            for i in range(self.batch, len(self.houses_data)):
                for house_predictor in self.House_Predictors:
                    house_predictor.train(self.data[i-self.batch:i], self.houses_data[i-self.batch:i], self.batch)
            epoch += 1
        self.save()


    def save(self):
        return


if __name__ == "__main__":
    try:
        logistic_regression("datasets/dataset_train.csv")
    except Exception as err:
        print(err)

