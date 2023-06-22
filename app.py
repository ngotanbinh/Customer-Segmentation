import pickle
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

import numpy as np
with open('mallCustomer_classifier.pickle','rb') as f:
    clf = pickle.load(f)

category=['Standard people','Tightwad people','Normal people',
          'Careless people(TARGET)','Rich people(TARGET)']
def main():

    st.title('Mall-Customer-Species-Classifier')
    sl = st.number_input('Annual Income (k$)')
    sw = st.number_input('Spending Score (1-100)')

    if st.button('Predict'):
        inp=np.array([[sl,sw]])
        out=clf.predict(inp)
        category[int(out[0])]

        print(category[int(out[0])])
if __name__=='__main__':
    main()
