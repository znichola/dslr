from describe import loadData, describe
import pandas as pd

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
        self.weights = [0 for _ in range(len(self.data.keys()))]
        self.bias = [0 for _ in range(len(self.data.keys()))]
        self.learning_rate = 0
        self.max_epoch = 10000
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
            
            
            epoch += 1



if __name__ == "__main__":
    try:
        logistic_regression("datasets/dataset_train.csv")
    except Exception as err:
        print(err)