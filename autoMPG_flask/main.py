import pickle
from flask import Flask, request, jsonify
from model_files.ml_model import predict_mpg

app = Flask("mpg_predictions")

@app.route('/', methods=['POST'])
def predict():
    vehicle_config = request.get_json()
    with open('./model_files/model_mpg.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = predict_mpg(vehicle_config, model)

    response = {
        'mpg_predictions': list(predictions)
    }
    return jsonify(response)

# @app.route('/')
# def hello_world():
#     return 'hello test'
if __name__ == '__main__':
    app.run()