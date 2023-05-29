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
    model = pd.read_pickle("website/models/model.pkl")
    s1 = pd.read_pickle("website/models/scaler1.pkl")
    s2 = pd.read_pickle("website/models/scaler2.pkl")
    s3 = pd.read_pickle("website/models/scaler3.pkl")


    # {'age': '23', 'cholesterol': '1', 'bmi': '12', 'activity': '0', 'health': '0', 'blood_pressure': '1'}

    data = pd.DataFrame(inputs_ll, index=[0])
    data[['BMI']] == s1.fit_transform(data[['BMI']])
    data[['Age']] = s2.fit_transform(data[['Age']])
    data[['PhysHlth']] = s3.transform(data[['PhysHlth']])
    #print(data)

    # do prediction using the pre-trained model
    preds = model.predict_proba(data)

    if preds.argmax() == 1:
        return f"Predicted Dibetes  with Probs score of  {preds.max():.4f}", 1
    else:
        return f"Predicted No Dibetes  with Probs score of  {preds.max():.4f}", 0


views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        # note = request.form.get('note')#Gets the note from the HTML
        #
        # if len(note) < 1:
        #     flash('Note is too short!', category='error')
        # else:
        #     new_note = Note(data=note, user_id=current_user.id)  #providing the schema for the note
        #     db.session.add(new_note) #adding the note to the database
        #     db.session.commit()

        input_dict = {
            'Age': request.form.get('age'),
            'HighChol': request.form.get('cholesterol'),
            'BMI': request.form.get('bmi'),
            'PhysActivity': request.form.get('activity'),
            'PhysHlth': request.form.get('health'),
            'HighBP': request.form.get('blood_pressure'),

            'CholCheck': request.form.get('chol_check'),
            'HeartDiseaseorAttack': request.form.get('heart_disease_or_attack'),
            'GenHlth': request.form.get('gen_hlth'),
            'MentHlth': request.form.get('men_hlth'),
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
