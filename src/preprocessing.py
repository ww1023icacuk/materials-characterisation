import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, path, target_col="label", test_size=0.2, random_state=42):
        self.path = path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def load_data(self):
        self.df = pd.read_csv(self.path)
        return self.df

    def clean(self):
        # Simple cleaning strategy: drop missing values
        self.df = self.df.dropna()
        return self.df

    def split(self):
        X = self.df.drop(columns=[self.target_col]).values
        y = self.df[self.target_col].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale(self):
        # Fit scaler on training only, transform both
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        return self.X_train, self.X_test

    def preprocess(self):
        self.load_data()
        self.clean()
        self.split()
        self.scale()
        return self.X_train, self.X_test, self.y_train, self.y_test
