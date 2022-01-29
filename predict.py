import numpy as np
import sys
from joblib import load

clf2 = load('heart_disease_svm_classifier.joblib')  # import classifier

# sample user data -->  58;1;2;132;224;0;0;173;0;3.2;2;2;3

user_data = list(map(float, sys.argv[1].split(';')))
user_data = np.reshape(user_data, (1, -1))
prediction = clf2.predict(user_data)

print(prediction[0])

