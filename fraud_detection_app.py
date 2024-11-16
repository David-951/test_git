
### Pour le deploiment on va sur share streamlit.io

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
#from sklearn.metrics import confusion_matrix, roc_curve #, precision_recall
from sklearn.metrics import precision_score, recall_score

#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

#import streamlit as st
#st.set_option('deprecation.showPyplotGlobalUse', False)

import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)



def main():
    st.title("Application de machine learning pour la detection de fraude pour la carte de credit.")
    st.subheader("Auteur : David")

    # fonction d'importation de donnees
    #@st.cache(persist=True) # pour sauvegarder en memoire nos variable pour ne pas recalculer le dataset car le dataset est le meme
    @st.cache_data
    def load_data():
        data = pd.read_csv("creditcard.csv")
        return data

    # Affichage de la table de donnees
    df = load_data()
    df_sample = df.sample(100)
    
    if st.sidebar.checkbox("Afficher les Donnees brutes", False):
        st.subheader("Jeux de donnees credit card : echantillon de 100 observations")
        st.write(df_sample)

    seed = 123

    # Train / Test / Split
    #@st.cache(persist=True)
    @st.cache_resource
    def split(dataframe):
        y = dataframe["Class"]
        X = dataframe.drop("Class", axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y, # car on est en face de desiquilibre de classe, ex , si ds x_train on a 15% de donnes frauduleuse, il faut que ds les donnees de test on a aussi 15% de donnes frauduleuse
            random_state=seed)
        return X_train, X_test, y_train, y_test
        
    
    X_train, X_test, y_train, y_test = split(dataframe=df)

    class_names = ["T.Authentique", "T.Frauduleuse"]

    classifieur = st.sidebar.selectbox(
        "Classificateur", 
         ("Random Forest", "SVM", "Logistic Regression")
         )
    
    # Analyse de la performance des modeles

    def plot_perf(graphes):
        if "Confusion matrix" in graphes:
            st.subheader("Matrice de confusion")
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, labels= class_names)

            st.pyplot()

        if "ROC curve" in graphes:
            st.subheader("Courbe ROC")
            RocCurveDisplay.from_estimator(model, X_test, y_test)
            st.pyplot()

        if "Precision-Recall curve" in graphes:
            st.subheader("Precision-Recall ")
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
            st.pyplot()


    # Random Forest , il faut ecrire les param comme dans sklearn car on va les reutiliser apres
    if classifieur == "Random Forest":
        st.sidebar.subheader("Hyperparametres du modele")
        n_arbres = st.sidebar.number_input("Choisir le nombre d'arbre dans la foret", 100, 1000, step=10) # 100 est le min par defaut et 1000 est le max 
        profondeur_arbre = st.sidebar.number_input("Choisir la profondeur maximale d'un arbre ",1,20, step = 1)
        bootstrap = st.sidebar.radio(
            "Echantillon bootstrap lors de la creation d'arbres ? ", ("True", "False"))  

        bootstrap = bootstrap == "True"

        graph_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance ML", ("Confusion matrix", "ROC curve", "Precision-Recall curve")
        )

        if st.sidebar.button("Execution", key = "Classify"):
            st.subheader("Random Forest Results")

            # Initialisation d'un objet RandomForestClassifier
            model = RandomForestClassifier(n_estimators= n_arbres, 
                                           max_depth= profondeur_arbre,
                                           bootstrap=bootstrap, random_state = seed)
            # Entrainement de l'algorithme
            model.fit(X_train,y_train)
            
            # Predictions
            y_pred = model.predict(X_test)

            # Metriques de performance
            accuracy = model.score(X_test,y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Afficher les metriques dans l'application 
            st.write("Accuracy :", round(accuracy,3))
            st.write("Precision :", round(precision,3))
            st.write("Recall :", round(recall,3))

            # Afficher les graphiques de performances
            plot_perf(graph_perf)



    # Regression logistique
    if classifieur == "Logistic Regression":
        st.sidebar.subheader("Hyperparametres du modele")
        hyp_C = st.sidebar.number_input("Choisir la valeur du parametre de regularisation", 0.01, 10.0)
        n_max_iter = st.sidebar.number_input("Choisir le nombre maximum d'iteration",100,1000, step = 10)


        graph_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance ML", ("Confusion matrix", "ROC curve", "Precision-Recall curve")
        )

        if st.sidebar.button("Execution", key = "Classify"):
            st.subheader("Logistic Regression Results")

            # Initialisation d'un objet LogisticRegression
            model = LogisticRegression(C= hyp_C, max_iter = n_max_iter, random_state = seed)
            # Entrainement de l'algorithme
            model.fit(X_train,y_train)
            
            # Predictions
            y_pred = model.predict(X_test)

            # Metriques de performance
            accuracy = model.score(X_test,y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Afficher les metriques dans l'application 
            st.write("Accuracy :", round(accuracy,3))
            st.write("Precision :", round(precision,3))
            st.write("Recall :", round(recall,3))

            # Afficher les graphiques de performances
            plot_perf(graph_perf)



    # SVM
    if classifieur == "SVM":
        st.sidebar.subheader("Hyperparametres du modele")
        hyp_C = st.sidebar.number_input("Choisir la valeur du parametre de regularisation", 0.01, 10.0)
        kernel = st.sidebar.radio("Choisir le kernel ",("rbf", "linear"))
        gamma = st.sidebar.radio("Gamma", ("scale", "auto"))


        graph_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance ML", ("Confusion matrix", "ROC curve", "Precision-Recall curve")
        )

        if st.sidebar.button("Execution", key = "Classify"):
            st.subheader("Support Vector Machine Results")

            # Initialisation d'un objet SVC
            model = SVC(C= hyp_C, kernel = kernel, gamma=gamma, random_state = seed)
            # Entrainement de l'algorithme
            model.fit(X_train,y_train)
            
            # Predictions
            y_pred = model.predict(X_test)

            # Metriques de performance
            accuracy = model.score(X_test,y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Afficher les metriques dans l'application 
            st.write("Accuracy :", round(accuracy,3))
            st.write("Precision :", round(precision,3))
            st.write("Recall :", round(recall,3))

            # Afficher les graphiques de performances
            plot_perf(graph_perf)





if __name__ == "__main__":
    main()