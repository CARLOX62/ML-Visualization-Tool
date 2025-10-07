from flask import Flask, render_template, request, jsonify
import os
import time
from ml_tool import generate_dataset, build_model as original_build_model, plot_decision_boundary, evaluate_model
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update():
    try:
        data = request.json
        dataset_name = data.get('dataset')
        model_name = data.get('model')
        params = data.get('params', {})
        dataset_params = data.get('dataset_params', {})
        dim = data.get('dim', '2d')

        # Generate dataset
        X, y = generate_dataset(dataset_name, dataset_params)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Build model
        model = build_model(model_name, params)

        # Train and measure time
        start_train = time.time()
        if 'Cluster' in model_name:
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - start_train

        # Plot decision boundary
        # Use new plot_decision_boundary that returns Plotly HTML string
        plot_html = plot_decision_boundary(model, X_train, y_train, X_test, y_test, None, dim, model_name)

        # Evaluate
        start_infer = time.time()
        if 'Cluster' in model_name:
            metrics = evaluate_model(model, X_train, None, model_name)
        else:
            metrics = evaluate_model(model, X_test, y_test, model_name)
        infer_time = time.time() - start_infer

        metrics['train_time'] = train_time
        metrics['infer_time'] = infer_time

        return jsonify({
            'plot_html': plot_html,
            'metrics': metrics
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Error in /update: {e}\n{tb}")
        return jsonify({'error': str(e), 'traceback': tb}), 500

def build_model(name, params):
    if name == "Linear Regression":
        # Remove 'normalize' param if present
        params = params.copy()
        if 'normalize' in params:
            params.pop('normalize')
    return original_build_model(name, params)

if __name__ == '__main__':
    app.run(debug=True)
