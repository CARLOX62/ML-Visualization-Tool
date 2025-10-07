import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs, make_gaussian_quantiles
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer, fetch_california_housing, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time
import os

# Import ML models
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier, BaggingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Optional imports for XGBoost, LightGBM, CatBoost
try:
    import xgboost as xgb
    has_xgboost = True
except ImportError:
    has_xgboost = False

try:
    import lightgbm as lgb
    has_lightgbm = True
except ImportError:
    has_lightgbm = False

try:
    import catboost as cb
    has_catboost = True
except ImportError:
    has_catboost = False

def generate_dataset(name, params):
    """
    Generate dataset by name with given parameters.
    Returns X, y after scaling and 2D reduction if needed.

    Parameters:
    - name (str): The name of the dataset to generate. Supported names include:
      - 'make_moons': Generates a two-moon dataset.
      - 'make_circles': Generates a two-circle dataset.
      - 'make_classification': Generates a classification dataset.
      - 'make_blobs': Generates a blob clustering dataset.
      - 'make_gaussian_quantiles': Generates a Gaussian quantiles dataset.
      - 'iris', 'wine', 'digits', 'breast_cancer', 'california_housing', 'diabetes', 'mnist': Loads real datasets.
    - params (dict): A dictionary of parameters specific to the dataset.
      - For 'make_moons': {'noise': float, default 0.3} - Standard deviation of Gaussian noise added to the data.
      - For 'make_circles': {'noise': float, default 0.2} - Standard deviation of Gaussian noise added to the data.
      - For 'make_classification': {'n_features': int, default 2} - Number of features; {'n_informative': int, default 2} - Number of informative features; {'class_sep': float, default 1.0} - The factor multiplying the hypercube size.
      - For 'make_blobs': {'centers': int, default 3} - Number of centers; {'cluster_std': float, default 1.0} - Standard deviation of clusters.
      - For 'make_gaussian_quantiles': {'n_classes': int, default 3} - Number of classes.
      - For real datasets, params is ignored.
    """
    if name == "make_moons":
        X, y = make_moons(n_samples=300, noise=params.get("noise", 0.3), random_state=42)
    elif name == "make_circles":
        X, y = make_circles(n_samples=300, noise=params.get("noise", 0.2), factor=0.5, random_state=42)
    elif name == "make_classification":
        X, y = make_classification(n_samples=300, n_features=params.get("n_features", 2),
                                   n_redundant=0, n_informative=params.get("n_informative", 2),
                                   n_clusters_per_class=1, class_sep=params.get("class_sep", 1.0),
                                   random_state=42)
    elif name == "make_blobs":
        X, y = make_blobs(n_samples=300, centers=params.get("centers", 3), cluster_std=params.get("cluster_std", 1.0), random_state=42)
    elif name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        X = PCA(n_components=2).fit_transform(X)
    elif name == "wine":
        data = load_wine()
        X, y = data.data, data.target
        X = PCA(n_components=2).fit_transform(X)
    elif name == "digits":
        data = load_digits()
        X, y = data.data, data.target
        X = PCA(n_components=2).fit_transform(X)
    elif name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = PCA(n_components=2).fit_transform(X)
    elif name == "make_gaussian_quantiles":
        X, y = make_gaussian_quantiles(n_samples=300, n_features=2, n_classes=params.get("n_classes", 3), random_state=42)
    elif name == "california_housing":
        data = fetch_california_housing()
        X, y = data.data, data.target
        X = PCA(n_components=2).fit_transform(X)
    elif name == "diabetes":
        data = load_diabetes()
        X, y = data.data, data.target
        X = PCA(n_components=2).fit_transform(X)
    elif name == "mnist":
        data = load_digits()
        X, y = data.data, data.target
        X = PCA(n_components=2).fit_transform(X)
    else:
        raise ValueError(f"Dataset {name} not supported or not implemented yet.")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# Add categorization dictionaries for models and datasets

model_categories = {
    "Logistic Regression": {"problem": "Classification", "learning": "Supervised"},
    "Perceptron": {"problem": "Classification", "learning": "Supervised"},
    "RidgeClassifier": {"problem": "Classification", "learning": "Supervised"},
    "SGDClassifier": {"problem": "Classification", "learning": "Supervised"},
    "K-Nearest Neighbors": {"problem": "Classification", "learning": "Supervised"},
    "Linear SVM": {"problem": "Classification", "learning": "Supervised"},
    "SVM": {"problem": "Classification", "learning": "Supervised"},
    "Decision Tree": {"problem": "Classification", "learning": "Supervised"},
    "Random Forest": {"problem": "Classification", "learning": "Supervised"},
    "Extra Trees": {"problem": "Classification", "learning": "Supervised"},
    "AdaBoost": {"problem": "Classification", "learning": "Supervised"},
    "Gradient Boosting": {"problem": "Classification", "learning": "Supervised"},
    "HistGradientBoosting": {"problem": "Classification", "learning": "Supervised"},
    "XGBoost": {"problem": "Classification", "learning": "Supervised"},
    "LightGBM": {"problem": "Classification", "learning": "Supervised"},
    "CatBoost": {"problem": "Classification", "learning": "Supervised"},
    "GaussianNB": {"problem": "Classification", "learning": "Supervised"},
    "BernoulliNB": {"problem": "Classification", "learning": "Supervised"},
    "MultinomialNB": {"problem": "Classification", "learning": "Supervised"},
    "LDA": {"problem": "Classification", "learning": "Supervised"},
    "QDA": {"problem": "Classification", "learning": "Supervised"},
    "MLPClassifier": {"problem": "Classification", "learning": "Supervised"},
    "RBF + Logistic Regression": {"problem": "Classification", "learning": "Supervised"},
    "Bagging": {"problem": "Classification", "learning": "Supervised"},
    "Stacking": {"problem": "Classification", "learning": "Supervised"},
    "Voting": {"problem": "Classification", "learning": "Supervised"},
    "Linear Regression": {"problem": "Regression", "learning": "Supervised"},
    "Ridge Regression": {"problem": "Regression", "learning": "Supervised"},
    "Lasso Regression": {"problem": "Regression", "learning": "Supervised"},
    "ElasticNet Regression": {"problem": "Regression", "learning": "Supervised"},
    "SVR": {"problem": "Regression", "learning": "Supervised"},
    "Decision Tree Regressor": {"problem": "Regression", "learning": "Supervised"},
    "Random Forest Regressor": {"problem": "Regression", "learning": "Supervised"},
    "Gradient Boosting Regressor": {"problem": "Regression", "learning": "Supervised"},
    "MLPRegressor": {"problem": "Regression", "learning": "Supervised"},
    "Bagging Regressor": {"problem": "Regression", "learning": "Supervised"},
    "KMeans": {"problem": "Clustering", "learning": "Unsupervised"},
    "DBSCAN": {"problem": "Clustering", "learning": "Unsupervised"},
    "Agglomerative Clustering": {"problem": "Clustering", "learning": "Unsupervised"},
}

dataset_categories = {
    "make_moons": {"problem": "Classification", "type": "Synthetic"},
    "make_circles": {"problem": "Classification", "type": "Synthetic"},
    "make_classification": {"problem": "Classification", "type": "Synthetic"},
    "make_blobs": {"problem": "Clustering", "type": "Synthetic"},
    "make_gaussian_quantiles": {"problem": "Classification", "type": "Synthetic"},
    "iris": {"problem": "Classification", "type": "Real"},
    "wine": {"problem": "Classification", "type": "Real"},
    "digits": {"problem": "Classification", "type": "Real"},
    "mnist": {"problem": "Classification", "type": "Real"},
    "breast_cancer": {"problem": "Classification", "type": "Real"},
    "california_housing": {"problem": "Regression", "type": "Real"},
    "diabetes": {"problem": "Regression", "type": "Real"},
}

def build_model(name, params):
    """
    Build and return a sklearn model based on name and hyperparameters.

    Parameters:
    - name (str): The name of the model to build. Supported names include:
      - Classification: 'Logistic Regression', 'Perceptron', 'RidgeClassifier', 'SGDClassifier', 'K-Nearest Neighbors', 'Linear SVM', 'SVM', 'Decision Tree', 'Random Forest', 'Extra Trees', 'AdaBoost', 'Gradient Boosting', 'HistGradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost', 'GaussianNB', 'BernoulliNB', 'MultinomialNB', 'LDA', 'QDA', 'MLPClassifier', 'RBF + Logistic Regression', 'Bagging', 'Stacking', 'Voting'
      - Regression: 'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet Regression', 'SVR', 'Decision Tree Regressor', 'Random Forest Regressor', 'Gradient Boosting Regressor', 'MLPRegressor', 'Bagging Regressor'
      - Clustering: 'KMeans', 'DBSCAN', 'Agglomerative Clustering'
    - params (dict): A dictionary of hyperparameters specific to the model.
      - Common parameters: 'C' (regularization strength, float), 'alpha' (regularization parameter, float), 'n_estimators' (number of estimators, int), 'max_depth' (max tree depth, int or None), 'learning_rate' (learning rate, float), 'max_iter' (max iterations, int), 'n_neighbors' (for KNN, int), 'kernel' (for SVM, str), 'gamma' (for SVM/RBF, float or str), 'penalty' (for Logistic, str), 'solver' (for Logistic, str), 'eta0' (for Perceptron, float), 'hidden_layer_sizes' (for MLP, tuple), 'voting' (for Voting, str), etc.
      - Defaults are provided in the code if not specified.
    """
    if name == "Logistic Regression":
        model = LogisticRegression(C=params.get("C", 1.0), penalty=params.get("penalty", "l2"),
                                   solver=params.get("solver", "lbfgs"), max_iter=params.get("max_iter", 100),
                                   multi_class=params.get("multi_class", "auto"), l1_ratio=params.get("l1_ratio", None))
    elif name == "Perceptron":
        model = Perceptron(max_iter=params.get("max_iter", 1000), eta0=params.get("eta0", 1.0), tol=1e-3)
    elif name == "RidgeClassifier":
        model = RidgeClassifier(alpha=params.get("alpha", 1.0))
    elif name == "SGDClassifier":
        model = SGDClassifier(max_iter=params.get("max_iter", 1000), tol=1e-3, alpha=params.get("alpha", 0.0001))
    elif name == "K-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5))
    elif name == "Linear SVM":
        model = LinearSVC(C=params.get("C", 1.0), max_iter=10000)
    elif name == "SVM":
        model = SVC(C=params.get("C", 1.0), kernel=params.get("kernel", "rbf"), gamma=params.get("gamma", "scale"), probability=True)
    elif name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=params.get("max_depth", None))
    elif name == "Random Forest":
        model = RandomForestClassifier(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", None))
    elif name == "Extra Trees":
        model = ExtraTreesClassifier(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", None))
    elif name == "AdaBoost":
        model = AdaBoostClassifier(n_estimators=params.get("n_estimators", 50), learning_rate=params.get("learning_rate", 1.0))
    elif name == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=params.get("n_estimators", 100), learning_rate=params.get("learning_rate", 0.1))
    elif name == "HistGradientBoosting":
        model = HistGradientBoostingClassifier(max_iter=params.get("max_iter", 100))
    elif name == "XGBoost":
        if has_xgboost:
            model = xgb.XGBClassifier(n_estimators=params.get("n_estimators", 100), learning_rate=params.get("learning_rate", 0.1), max_depth=params.get("max_depth", 5))
        else:
            raise ValueError("XGBoost is not installed. Please install it to use this model.")
    elif name == "LightGBM":
        if has_lightgbm:
            model = lgb.LGBMClassifier(n_estimators=params.get("n_estimators", 100), learning_rate=params.get("learning_rate", 0.1), max_depth=params.get("max_depth", 5))
        else:
            raise ValueError("LightGBM is not installed. Please install it to use this model.")
    elif name == "CatBoost":
        if has_catboost:
            model = cb.CatBoostClassifier(iterations=params.get("n_estimators", 100), learning_rate=params.get("learning_rate", 0.1), max_depth=params.get("max_depth", 5), verbose=0)
        else:
            raise ValueError("CatBoost is not installed. Please install it to use this model.")
    elif name == "GaussianNB":
        model = GaussianNB()
    elif name == "BernoulliNB":
        model = BernoulliNB()
    elif name == "MultinomialNB":
        model = MultinomialNB()
    elif name == "LDA":
        model = LinearDiscriminantAnalysis()
    elif name == "QDA":
        model = QuadraticDiscriminantAnalysis()
    elif name == "MLPClassifier":
        model = MLPClassifier(hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)), max_iter=1000)
    elif name == "RBF + Logistic Regression":
        rbf_feature = RBFSampler(gamma=params.get("gamma", 1.0), random_state=42)
        logistic = LogisticRegression(max_iter=1000)
        model = make_pipeline(rbf_feature, logistic)
    elif name == "Bagging":
        base = DecisionTreeClassifier(max_depth=params.get("max_depth", None))
        model = BaggingClassifier(base_estimator=base, n_estimators=params.get("n_estimators", 10))
    elif name == "Stacking":
        estimators = [
            ('lr', LogisticRegression(max_iter=1000)),
            ('dt', DecisionTreeClassifier(max_depth=5)),
            ('svm', SVC(kernel='linear', probability=True))
        ]
        model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    elif name == "Voting":
        estimators = [
            ('lr', LogisticRegression(max_iter=1000)),
            ('dt', DecisionTreeClassifier(max_depth=5)),
            ('svm', SVC(kernel='linear', probability=True))
        ]
        voting_type = params.get("voting", "hard")
        model = VotingClassifier(estimators=estimators, voting=voting_type)
    elif name == "Linear Regression":
        # Remove 'normalize' param as it is deprecated in newer sklearn versions
        fit_intercept = params.get("fit_intercept", True)
        model = LinearRegression(fit_intercept=fit_intercept)
    elif name == "Ridge Regression":
        model = Ridge(alpha=params.get("alpha", 1.0))
    elif name == "Lasso Regression":
        model = Lasso(alpha=params.get("alpha", 1.0))
    elif name == "ElasticNet Regression":
        model = ElasticNet(alpha=params.get("alpha", 1.0), l1_ratio=params.get("l1_ratio", 0.5))
    elif name == "SVR":
        model = SVR(C=params.get("C", 1.0), kernel=params.get("kernel", "rbf"), gamma=params.get("gamma", "scale"))
    elif name == "Decision Tree Regressor":
        model = DecisionTreeRegressor(max_depth=params.get("max_depth", None))
    elif name == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", None))
    elif name == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor(n_estimators=params.get("n_estimators", 100), learning_rate=params.get("learning_rate", 0.1))
    elif name == "MLPRegressor":
        model = MLPRegressor(hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)), max_iter=1000)
    elif name == "Bagging Regressor":
        base = DecisionTreeRegressor(max_depth=params.get("max_depth", None))
        model = BaggingRegressor(base_estimator=base, n_estimators=params.get("n_estimators", 10))
    elif name == "KMeans":
        model = KMeans(n_clusters=params.get("n_clusters", 3))
    elif name == "DBSCAN":
        model = DBSCAN(eps=params.get("eps", 0.5), min_samples=params.get("min_samples", 5))
    elif name == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=params.get("n_clusters", 3))
    else:
        model = None
    return model

def plot_decision_boundary(model, X_train, y_train, X_test, y_test, plot_path, dim='2d', model_name=''):
    """
    Plot decision boundary with train/test points, probability heatmap if available,
    misclassified points highlighted.
    Save plot to plot_path.

    For classification: Uses scatter plot to show data points (colored by class) and contour plot to show the classifier’s decision boundary.
    For regression: Scatter plot for data points and line plot for predictions.
    For clustering: Scatter plot for data points colored by cluster labels.

    Parameters:
    - model: The trained machine learning model to visualize.
    - X_train (array-like): Training feature data (2D).
    - y_train (array-like): Training target labels.
    - X_test (array-like): Test feature data (2D).
    - y_test (array-like): Test target labels.
    - plot_path (str): File path where the plot will be saved (e.g., 'static/plots/plot.png').
    - dim (str): Dimension of the plot, default '2d' (currently only 2D supported).
    - model_name (str): Name of the model for plot title and logic (e.g., 'Cluster' for clustering, 'Regressor' for regression).
    """
    import os

    if plot_path is None:
        plot_path = 'static/plots/plot.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()

    if 'Cluster' in model_name:
        labels = model.labels_
        ax.scatter(X_train[:, 0], X_train[:, 1], c=labels, cmap='rainbow', edgecolors='k', s=20)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title('Clusters')

    elif 'Regressor' in model_name:
        ax.scatter(X_train[:, 0], y_train, c=y_train, cmap='plasma', edgecolors='k', s=20)
        # Sort for line plot
        sort_idx = np.argsort(X_train[:, 0])
        X_sorted = X_train[sort_idx]
        y_pred = model.predict(X_sorted)
        ax.plot(X_sorted[:, 0], y_pred, color='red', linewidth=2)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Target")
        ax.set_title('Regression')

    else:
        # Classification
        h = 0.02  # step size in the mesh
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.5, cmap='rainbow')

        # Scatter plot for training and test points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='rainbow', edgecolors='k', s=20, label='Train')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='rainbow', edgecolors='k', s=20, marker='s', label='Test')

        # Highlight misclassified test points
        y_pred = model.predict(X_test)
        misclassified_idx = np.where(y_pred != y_test)[0]
        ax.scatter(X_test[misclassified_idx, 0], X_test[misclassified_idx, 1], c='red', s=50, marker='x', label='Misclassified')

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title('Decision Boundary')
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

    return f'<img src="/{plot_path}" style="max-width:100%; height:auto;">'

def evaluate_model(model, X_test, y_test, model_name=''):
    """
    Evaluate model and return metrics dictionary.

    Parameters:
    - model: The trained machine learning model to evaluate.
    - X_test (array-like): Test feature data.
    - y_test (array-like): Test target labels.
    - model_name (str): Name of the model to determine evaluation type ('Cluster' for clustering, 'Regressor' for regression, else classification).
    """
    if 'Cluster' in model_name:
        labels = model.labels_
        silhouette = silhouette_score(X_test, labels)
        ch_score = calinski_harabasz_score(X_test, labels)
        db_score = davies_bouldin_score(X_test, labels)
        metrics = {
            "silhouette_score": silhouette,
            "calinski_harabasz_score": ch_score,
            "davies_bouldin_score": db_score
        }
    elif 'Regressor' in model_name:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {
            "mse": mse,
            "r2": r2
        }
    else:
        y_pred = None
        # Handle models without predict method (e.g., DBSCAN)
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            # For clustering models like DBSCAN, use labels_ as predictions
            if hasattr(model, 'labels_'):
                y_pred = model.labels_
            else:
                raise ValueError(f"Model {model_name} does not support prediction or labels.")
        # Check if y_test is continuous for classification metrics
        is_classification = True
        if y_test is not None:
            # If y_test has float type and many unique values, treat as regression
            if np.issubdtype(y_test.dtype, np.floating) and len(np.unique(y_test)) > 20:
                is_classification = False
        if is_classification:
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            # If continuous target, fallback to regression metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = {
                "mse": mse,
                "r2": r2
            }
    return metrics
