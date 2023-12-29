import streamlit as st
from preprocess import dataset
from nb import show_nb
from kmeans import show_kmeans
from nn import show_nn
from dt import show_dt
from knn import show_knn
from lr import show_lr

ds = dataset
ds.preprocess_dataset()

selection = st.sidebar.selectbox("Select what you want to find out", ["Predict above avg income", 
                                                                      "See similar jobs", 
                                                                      "Predict salaries", 
                                                                      "Estimate required experience level",
                                                                      "Predict above avg income based on job",
                                                                      "Predict indutsry"])

if selection == "Predict above avg income":
    show_nb()
elif selection == "See similar jobs":
    show_kmeans()
elif selection == "Predict salaries":
    show_nn()
elif selection == "Estimate required experience level":
    show_dt()
elif selection == "Predict above avg income based on job":
    show_lr()
elif selection == "Predict indutsry":
    show_knn()