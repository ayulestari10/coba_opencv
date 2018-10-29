from flask import Flask

# Load libraries
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np


# Load dataset iris
# link : https://archive.ics.uci.edu/ml/datasets/iris
from sklearn.datasets import load_iris

# import OpenCV

import cv2


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
 

@app.route('/deteksi_wajah')
def deteksi_wajah():
	
	face_cascade = cv2.CascadeClassifier('C:\\xampp\\htdocs\\gmisvm\\static\\haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('C:\\xampp\\htdocs\\gmisvm\\static\\haarcascade_eye.xml')

	img = cv2.imread('C:\\xampp\\htdocs\\coba_opencv\\lena.png')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]
	    eyes = eye_cascade.detectMultiScale(roi_gray)
	    for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	cv2.imshow('img',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	for f in faces:
		x, y, w, h = [v for v in f]
		cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255))
		sub_face = img[y:y+h, x:x+w]

		face_file_name = "C:\\xampp\\htdocs\\coba_opencv\\hasil\\face_" + str(y) + ".png"
		cv2.imwrite(face_file_name, sub_face)

	return 'Coba deteksi wajah'



@app.route('/library_python')
def library_python():
	
	data = load_iris()

	features = data.data[:, :4]  # 4 fitur
	labels = data.target

	k_scores = []
	kf = KFold(n_splits=10, random_state=1, shuffle=True)

	for i, (train_index, test_index) in enumerate(kf.split(features)):

		clf = tree.DecisionTreeClassifier()
		clf.fit(features[train_index], labels[train_index])
		scores = round(clf.score(features[test_index], labels[test_index]) * 100, 2)
		k_scores.append(scores)
		print(f"Score k-{i + 1} : {scores}%")

	rata_rata = np.mean(np.array(k_scores))
	print(f"Mean: {rata_rata}%")

	return 'Coba library python'
