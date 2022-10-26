# from asyncio import FastChildWatcher
import numpy as np
import pandas as pd
import cv2
from sklearn.neighbors import KNeighborsClassifier
import os 
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn import tree
from sklearn import svm
from sklearn import metrics
import pickle


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class ImageClasse:
    def __init__(self):
        # datadir = "./sign_data/train"
        # target_arr = []
        # flat_data_arr = []
        # print("hello")
        # for i in os.listdir(datadir):
        #     path = os.path.join(datadir , i)
        #     for  img in os.listdir(path):
        #         image= cv2.imread(os.path.join(path,img))
        #         img_resized= resize(image,(70,70))
        #         image=  cv2.cvtColor(np.float32(img_resized),cv2.COLOR_BGR2GRAY)
        #         flat_data_arr.append(image)
        #         target_arr.append(i)
        # flat_data= np.array(flat_data_arr) 
        # flat_data= flat_data.reshape(1649,70*70)
        # target = np.array(target_arr)



        # datadir = "./sign_data/test"
        # target_arr = []
        # flat_data_arr = []
        # print("hello")
        # for i in os.listdir(datadir):
        #     path = os.path.join(datadir , i)
        #     for  img in os.listdir(path):
        #         image= cv2.imread(os.path.join(path,img))
        #         img_resized= resize(image,(70,70))
        #         image=  cv2.cvtColor(np.float32(img_resized),cv2.COLOR_BGR2GRAY)
        #         flat_data_arr.append(image)
        #         target_arr.append(i)
        # flat_data= np.array(flat_data_arr) 
        # flat_data= flat_data.reshape(500,70*70)
        # target = np.array(target_arr)
        # len(target)
        # targetbi = []
        # for i in target:
        #     if(i.find("forg")== -1):
        #         targetbi.append("reel")
        #     else:
        #         targetbi.append("forg")
        

        # self.neigh = KNeighborsClassifier(n_neighbors=3)
        # self.neigh.fit(flat_data, target)
        # pickle.dump(self.neigh,open('neigh.sav','wb'))
        # datadir = "./sign_data/train"
        # target_arr = []
        # flat_data_arr = []
        # print("hello")
        # for i in os.listdir(datadir):
        #     path = os.path.join(datadir , i)
        #     for  img in os.listdir(path):
        #         image= cv2.imread(os.path.join(path,img))
        #         img_resized= resize(image,(70,70))
        #         image=  cv2.cvtColor(np.float32(img_resized),cv2.COLOR_BGR2GRAY)
        #         flat_data_arr.append(image)
        #         target_arr.append(i)
        # flat_data= np.array(flat_data_arr) 
        # flat_data= flat_data.reshape(1649,70*70)
        # target = np.array(target_arr)

        # self.neigh = KNeighborsClassifier(n_neighbors=3)
        # self.neigh.fit(flat_data, target)
        # pickle.dump(self.neigh,open('neigh.sav','wb'))

        self.neigh=pickle.load(open('saves/neigh.sav','rb'))
        

        # self.gnb = GaussianNB()
        # self.gnb = self.gnb.fit(flat_data , target)
        # pickle.dump(self.gnb,open('gnb.sav','wb'))

        self.gnb=pickle.load(open('saves/gnb.sav','rb'))

        # targetbi = []
        # for i in target:
        #     if(i.find("forg")== -1):
        #         targetbi.append("reel")
        #     else:
        #         targetbi.append("forg")

        # df= pd.DataFrame(flat_data)
        # df['etiquette']=targetbi
        # self.x=df.iloc[:,:-1]
        # self.y=df.iloc[:,-1]


        # self.model=svm.SVC(kernel='linear')
        # self.model.fit(self.x,self.y)
        # pickle.dump(self.model,open('svm.sav','wb'))
        self.model=pickle.load(open('saves/svm.sav','rb'))


        # X_normalized = pd.DataFrame(flat_data)
        # pca = PCA(n_components = 10)
        # X_principal = pca.fit_transform(X_normalized)
        # X_principal = pd.DataFrame(X_principal)

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( X_principal, target, test_size=0.3, random_state=0)

        # clf = tree.DecisionTreeClassifier()
        # self.clf = clf.fit(flat_data, target)
        # pickle.dump(self.clf,open('dt.sav','wb'))

        self.clf=pickle.load(open('saves/dt.sav','rb'))

        # self.forest = RandomForestClassifier(max_depth=2, random_state=0)
        # self.forest.fit(flat_data , target)
        # pickle.dump(self.forest,open('forest.sav','wb'))

        self.forest=pickle.load(open('saves/forest.sav','rb'))

        # score =[]
        # a= metrics.accuracy_score(self.neigh.predict(flat_data),target)
        # score.append(a)
        # a= metrics.accuracy_score(self.gnb.predict(flat_data),target)
        # score.append(a)
        # a= metrics.accuracy_score(self.model.predict(flat_data),targetbi)
        # score.append(a)
        # a= metrics.accuracy_score(self.clf.predict(flat_data),target)
        # score.append(a)
        # a= metrics.accuracy_score(self.forest.predict(flat_data),target)
        # score.append(a)
        # print(score)

        self.score = pickle.load(open('saves/score.sav','rb'))
     



        


    def setImage(self,image):
        self.image = image 
    
    def Knn(self):
        print("helmdslllo")
        imageres = self.image
        imageres = resize(imageres, (70,70)) 
        imageres=  cv2.cvtColor(np.float32(imageres),cv2.COLOR_BGR2GRAY)
        imageres = np.array(imageres)
        imageres= imageres.reshape(1,70*70)

        return self.neigh.predict(imageres)
    
    def Nb(self):
        print("hellonb")
        imageres= self.image
        imageres = resize(imageres, (70,70)) 
        imageres=  cv2.cvtColor(np.float32(imageres),cv2.COLOR_BGR2GRAY)
        imageres = np.array(imageres)
        imageres= imageres.reshape(1,70*70)
        print(imageres)

        return self.gnb.predict(imageres)
    
    def Svm(self):
        print("hello dt")
        imageres= self.image
        imageres = resize(imageres, (70,70)) 
        imageres=  cv2.cvtColor(np.float32(imageres),cv2.COLOR_BGR2GRAY)
        imageres = np.array(imageres)
        imageres= imageres.reshape(1,70*70)
        return self.model.predict(imageres)
    
    def Dt(self):
        imageres= self.image
        imageres = resize(imageres, (70,70)) 
        imageres=  cv2.cvtColor(np.float32(imageres),cv2.COLOR_BGR2GRAY)
        imageres = np.array(imageres)
        imageres= imageres.reshape(1,70*70)   
        return self.clf.predict(imageres)


    def Rf(self):

        imageres= self.image
        imageres = resize(imageres, (70,70)) 
        imageres=  cv2.cvtColor(np.float32(imageres),cv2.COLOR_BGR2GRAY)
        imageres = np.array(imageres)
        imageres= imageres.reshape(1,70*70)   

        return self.forest.predict(imageres)

    def evaluate(self):
        print("helllo")
        return self.score



        



        





    
