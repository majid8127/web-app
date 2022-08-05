import streamlit as st
import numpy as np
import matplotlib.pyplot as plt  #"pyplot": unknown word.
from sklearn import datasets     #"sklearn": unknown word.   
from sklearn.model_selection import train_test_split  #"sklearn": unknown word.
from sklearn.decomposition import PCA  # "sklearn": unknown word.
from sklearn.svm import SVC   #"sklearn": unknown word.
from sklearn.neighbors import KNeighborsClassifier  # "sklearn": unknown word.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Headings:
st.write('''
# Explore different ML models and datasets
Daikhtay han kon sa best ha in may say?)         
''')

# data set k name ak box may daal k sidebar pay laga do 
dataset_name=st.sidebar.selectbox(
    'Select Dataset',
    ('Iris','Breast Cancet', 'Wine')
)

# Aur is k nichay classifiers k naam aik dabay main daal do:
classifier_name=st.sidebar.selectbox(
    "select classifier",
    ('KNN' , 'SVM' , 'Random Forest')
)


# Ab hum nay aik fuction define krna hai dataset ko load krnay k liye:
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data=datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y

# Ab is function ko bula lay gayn or x,y variable k wqual rakh layn gay 
X,y = get_dataset(dataset_name)

# Ab hum apnay data set ki shape ko ap pau kr dayn gay:
st.write("Shape of datset:", X.shape)
st.write("number of classes:",len(np.unique(y)))

# Next hum different classifiers k parameters ko user input may add karayn gay 
def add_parameter_ui(classifier_name):
    params=dict()   # create an empty dictionary.
    if classifier_name == "SVM":
        C = st.sidebar.slider("C",0.01, 10.0)
        params['C']=C #its the degree of correct classification:
    elif classifier_name == 'KNN':
        k=st.sidebar.slider('K', 1, 15)
        params['K'] = k    # its the number of nearest neighbour
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params['max_depth']= max_depth  # depth of every tree that grow in random forest
        n_estmators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators']=n_estmators  # number of trees:
    return params

# ab hum is fuction ko call kar lay gaye or params variable equal rakh dye gye.
params=add_parameter_ui(classifier_name)

# ab hum classifier bnaye gay base on classifier_name and params:
def get_classifier(classifier_name, params):
    clf=None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf= KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf= clf= RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth=params['max_depth'], random_state=1234)
    return clf

# ab is function ko bula lay gayn or clf variable k equal rakh lye gay:
clf= get_classifier(classifier_name,params)

# Ab hum dataset ko test and train data main split kr laytay hye by 80/20 ratio:
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1234)


# Ab hum nay apany classifier ki training krni ha.
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Model ka accuracy score check kr layne ha or isay app pay print dyna hy.
acc=accuracy_score(y_test,y_pred)
st.write(f'Classifier={classifier_name}')
st.write(f'Accuracy = ',acc)

#### Plot Dataset ####
# ab hum apnay saray features ko 2 dimensional plot pay draw kr dye gye.
pca= PCA(2)
X_projected = pca.fit_transform(X)
# Ab hum apna data 0 and 1 dimensional may slice kr dye gay.
x1= X_projected[:,0]
x2= X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2, c=y, alpha=0.8,cmap='viridis')

plt.xlabel('Principal Component 1')
plt.xlabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)