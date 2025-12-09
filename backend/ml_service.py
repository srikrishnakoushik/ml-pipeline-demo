import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class MLEngine:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self, file_path):
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.df = pd.read_excel(file_path)
            
            self.df.dropna(inplace=True)
            
            return {
                "columns": list(self.df.columns),
                "rows": self.df.shape[0],
                "preview": self.df.head().to_dict(orient='records')
            }
        except Exception as e:
            return {"error": str(e)}

    def run_pipeline(self, target_column, split_ratio, preprocessing, model_type):
        if self.df is None:
            return {"error": "No data loaded"}

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        if preprocessing == "StandardScaler":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif preprocessing == "MinMaxScaler":
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        
        test_size = 1 - float(split_ratio)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if model_type == "Logistic Regression":
            self.model = LogisticRegression()
        elif model_type == "Decision Tree":
            self.model = DecisionTreeClassifier()
        
        self.model.fit(self.X_train, self.y_train)

        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        conf_matrix = confusion_matrix(self.y_test, predictions).tolist()

        return {
            "status": "Success",
            "accuracy": round(accuracy * 100, 2),
            "confusion_matrix": conf_matrix
        }
