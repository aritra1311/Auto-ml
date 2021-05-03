import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
st.set_option('deprecation.showPyplotGlobalUse', False)
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
    layout='wide')
#---------------------------------#
def target(df):
    l=list(df.columns)
    t = st.selectbox('Select the target column number',l,index=0);

    return t
# Model building
def build_model(df,c):
    #df = df.loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION




    Y = df[c] # Selecting the last column as Y
    X = df.loc[:, df.columns != c] # Using all column except for the c column as X
    st.markdown('**1.2. Dataset dimension**')
    st.write('X')
    st.info(X.shape)
    st.write('Y')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.write('Y variable')
    st.info(Y.name)
    st.markdown('**1.4. Histograms**:')
    df.hist(alpha=0.5, figsize=(20, 10))
    st.pyplot()
    # Build lazy model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
    if model=='Regression':
        reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None,predictions=True)
    elif model=='Classification':
        reg=LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None,predictions=True)
    models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)
    pr = ProfileReport(df, explorative=True)
    st.header('**2.Pandas Profiling Report(Exploratory data Analysis)**')
    st_profile_report(pr)
    st.subheader('3. Table of Model Performance')

    st.write('Training set')
    st.write(models_train)
    st.markdown(filedownload(models_train,'modeltraining.csv'), unsafe_allow_html=True)

    st.write('Test set')
    st.write(models_test)
    st.markdown(filedownload(models_test,'modeltest.csv'), unsafe_allow_html=True)
    st.subheader('4. Predictions By the models')
    st.write('Training set')
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train,'predicttraining.csv'), unsafe_allow_html=True)

    st.write('Test set')
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test,'predicttest.csv'), unsafe_allow_html=True)



    st.subheader('5. Plot of Model Performance (Test set)')
    if model=='Regression':
        with st.markdown('**R-squared**'):
            # Tall
            models_test["R-Squared"] = [0 if i < 0 else i for i in models_test["R-Squared"] ]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(y=models_test.index, x="R-Squared", data=models_test)
            ax1.set(xlim=(0, 1))
        st.markdown(imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
        # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=models_test.index, y="R-Squared", data=models_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-r2-wide.pdf'), unsafe_allow_html=True)

        with st.markdown('**RMSE (capped at 50)**'):
            # Tall
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax2 = sns.barplot(y=models_test.index, x="RMSE", data=models_test)
        st.markdown(imagedownload(plt,'plot-rmse-tall.pdf'), unsafe_allow_html=True)
            #Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=models_test.index, y="RMSE", data=models_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-rmse-wide.pdf'), unsafe_allow_html=True)

        with st.markdown('**Calculation time**'):
            # Tall
            models_test["Time Taken"] = [0 if i < 0 else i for i in models_test["Time Taken"] ]#        plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax3 = sns.barplot(y=models_test.index, x="Time Taken", data=models_test)
        st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=models_test.index, y="Time Taken", data=models_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)
    elif model=='Classification':
        with st.markdown('**Accuracy**'):
            # Tall
            models_test["Accuracy"] = [0 if i < 0 else i for i in models_test["Accuracy"] ]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(y=models_test.index, x="Accuracy", data=models_test)
            ax1.set(xlim=(0, 1))
        st.markdown(imagedownload(plt,'plot-accuracy-tall.pdf'), unsafe_allow_html=True)
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=models_test.index, y="Accuracy", data=models_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-accuracy-wide.pdf'), unsafe_allow_html=True)
        with st.markdown('**Balanced Accuracy**'):
            # Tall
            models_test["Balanced Accuracy"] = [0 if i < 0 else i for i in models_test["Balanced Accuracy"] ]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(y=models_test.index, x="Balanced Accuracy", data=models_test)
            ax1.set(xlim=(0, 1))
        st.markdown(imagedownload(plt,'plot-balanced-accuracy-tall.pdf'), unsafe_allow_html=True)
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=models_test.index, y="Balanced Accuracy", data=models_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-balanced-accuracy-wide.pdf'), unsafe_allow_html=True)
        with st.markdown('**F1 Score**'):
            # Tall
            models_test["F1 Score"] = [0 if i < 0 else i for i in models_test["F1 Score"] ]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(y=models_test.index, x="F1 Score", data=models_test)
            ax1.set(xlim=(0, 1))
        st.markdown(imagedownload(plt,'plot-F1-Score-tall.pdf'), unsafe_allow_html=True)
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=models_test.index, y="F1 Score", data=models_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-F1-Score-wide.pdf'), unsafe_allow_html=True)
        with st.markdown('**Calculation time**'):
            # Tall
            models_test["Time Taken"] = [0 if i < 0 else i for i in models_test["Time Taken"] ]#        plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax3 = sns.barplot(y=models_test.index, x="Time Taken", data=models_test)
        st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=models_test.index, y="Time Taken", data=models_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)
# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

#---------------------------------#
st.write("""
# The Machine Learning Algorithm Comparison App
In this implementation, the **lazypredict** library is used for building several machine learning models at once.
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


model=st.sidebar.selectbox('2.Choose your Model',('Classification','Regression') )

# Sidebar - Specify parameter settings
with st.sidebar.header('3. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)




#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    c=target(df)
    if st.button("Use custom Dataset"):
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        build_model(df,c)
else:
    st.info('Awaiting for CSV file to be uploaded.')

        # Diabetes dataset
    diabetes = load_diabetes()
        #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        #Y = pd.Series(diabetes.target, name='response')
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names).loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
    Y = pd.Series(diabetes.target, name='response').loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
    df = pd.concat( [X,Y], axis=1 )
    c=target(df)
    if st.button("Use Example Dataset"):
        st.markdown('The Diabetes dataset is used as the example.')
        st.write(df.head(5))

        # Boston housing dataset
        boston = load_boston()
        #X = pd.DataFrame(boston.data, columns=boston.feature_names)
        #Y = pd.Series(boston.target, name='response')
        #X = pd.DataFrame(boston.data, columns=boston.feature_names).loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        #Y = pd.Series(boston.target, name='response').loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        #df = pd.concat( [X,Y], axis=1 )

        #st.markdown('The Boston housing dataset is used as the example.')
        #st.write(df.head(5))

        build_model(df,c)
