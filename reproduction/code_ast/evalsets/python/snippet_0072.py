from flask import Flask, Response, jsonify
from model import Model
from flask import request
import pandas as pd

app = Flask(__name__)
model_grade = Model('Modeles/model_grade.joblib')
model_stade = Model('Modeles/model_stade.joblib')


@app.route('/predict_grade_n_stade', methods=["GET"])
def predict_grade():
    input_json = {k: [request.args.get(k)] for k in model_stade.meta_data["required_input"]}
    input = pd.DataFrame(input_json)
    try:
        prediction_grade = model_grade.predict(input)
        prediction_stade = model_stade.predict(input)

        result = {
            'grade': prediction_grade[0],
            'stade': prediction_stade[0]
        }
        # print(result)
        # return Response(result, status=200, mimetype='application/json')
        return jsonify(result)
    except:
        return Response({"message":"Something went wrong, please try again."}, status=500, mimetype='application/json')

@app.route('/health')
def health_check():
    return Response("", status=200)


# if __name__ == '__main__':
#     app.run(debug=True)
    # app.run(debug=True, host='0.0.0.0', port=4000)