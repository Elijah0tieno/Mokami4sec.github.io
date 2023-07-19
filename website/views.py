from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Note
from . import db
import json



import pickle
import numpy as np
import pandas as pd
import os

def make_prediction(inputs_ll):
    # Load the pre-trained model
    model = pd.read_pickle("website/models/xg.pkl")
    scaler = pd.read_pickle("website/models/scaler.pkl")

    data = pd.DataFrame(inputs_ll, index=[0])[['PhysHlth', 'BMI', 'MentHlth', 'Age', 'GenHlth', 'HighBP', 'DiffWalk']]
    preds = model.predict_proba(scaler.transform(data))
    if preds.argmax() == 1:
        return f"Predicted Dibetes  with Probs score of  {preds.max():.4f}", 1
    else:
        return f"Predicted No Dibetes  with Probs score of  {preds.max():.4f}", 0


views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        input_dict = {
            'PhysHlth': request.form.get('health'),
            'BMI': request.form.get('bmi'),
            'MentHlth': request.form.get('men_hlth'),
            'Age': request.form.get('age'),
            'GenHlth': request.form.get('gen_hlth'),
            'HighBP': request.form.get('blood_pressure'),
            'DiffWalk': request.form.get('diff_walk'),

            }


        print(f"Input data passed are   {input_dict}")
        prediction, class_lbl = make_prediction(input_dict)
        print(f"Results from the input are  {prediction}")
        results = f"Results from the input are  {prediction}"
        if class_lbl == 0:
            flash(results, category='success')
        else:
            flash(results, category='warning')

        return render_template("home.html", user=current_user, results=prediction)


    return render_template("home.html", user=current_user, results=None)

#
# @views.route('/delete-note', methods=['POST'])
# def delete_note():
#     note = json.loads(request.data) # this function expects a JSON from the INDEX.js file
#     noteId = note['noteId']
#     note = Note.query.get(noteId)
#     if note:
#         if note.user_id == current_user.id:
#             db.session.delete(note)
#             db.session.commit()
#
#     return jsonify({})
