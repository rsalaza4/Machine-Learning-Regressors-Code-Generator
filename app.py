# Import libraries and dependencies
import streamlit as st
import numpy as np
import pandas as pd
import base64
from PIL import Image

# Set a title
st.title("Machine Learning Regressors Code Generator")

# Python iogo image
python_logo = Image.open("python.png")
st.write("")
st.image(python_logo, width=600)

# Data Source
st.sidebar.subheader("Data Source")
data_source = st.sidebar.selectbox("Select the data source file extension:", [".csv file", ".xlsx file"])

if data_source == ".csv file":
	data_source = "csv"
else:
	data_source = "excel"

# Data File Path
st.sidebar.subheader("Input Data File Path")
path = st.sidebar.text_input("Enter the input data file path here:", "Desktop/")

# Machine Learning Algorithm

st.sidebar.subheader("Regressor Algorithm")
algorithm = st.sidebar.selectbox("Select a machine learning algorithm:", ["Extra Trees", "Lasso", "Linear Regression", "Random Forest", "Ridge", "Stochastic Gradient Descent"])

if algorithm == "Extra Trees":
	algorithm_import = "from sklearn.ensemble import ExtraTreesRegressor"
	algorithm_instance = "etr"
	algorithm_class = "ExtraTreesRegressor()"

elif algorithm == "Lasso":
	algorithm_import = "from sklearn.linear_model import Lasso"
	algorithm_instance = "lasso"
	algorithm_class = "Lasso()"

elif algorithm == "Linear Regression":
	algorithm_import = "from sklearn.linear_model import LinearRegression"
	algorithm_instance = "lr"
	algorithm_class = "LinearRegression()"

elif algorithm == "Random Forest":	
	algorithm_import = "from sklearn.ensemble import RandomForestRegressor"
	algorithm_instance = "rf"
	algorithm_class = "RandomForestRegressor()"

elif algorithm == "Ridge":	
	algorithm_import = "from sklearn.linear_model import Ridge"
	algorithm_instance = "ridge"
	algorithm_class = "Ridge()"

elif algorithm == "Stochastic Gradient Descent":	
	algorithm_import = "from sklearn.linear_model import SGDRegressor"
	algorithm_instance = "sgd"
	algorithm_class = "SGDRegressor()"
	
# Train/Test Split Ratio

st.sidebar.subheader("Train/Test Split Ratio")
train_test_ratio = st.sidebar.number_input("Enter the percentage of the training set:", 0, max_value = 99, value = 70)

# Set instuctions
st.subheader("Instructions:")

st.write("1. Specify the variables on the side bar (*click on > if closed*)")
st.write("2. Copy the generated Python script to your clipboard")
st.write("3. Paste the generated Python script on your IDE of preference")
st.write("4. Run the Python script")

# Display generated Python code
st.subheader("Python Code:")

st.code(

	"# Import libraries and dependencies" +"\n"+ 
	"import numpy as np" +"\n"+ 
	"import pandas as pd" +"\n\n"+

	"# ------------------------------ Data Set Loading ------------------------------" +"\n\n"+

	"# Read data set" +"\n"+
	"df = pd.read_" + data_source + "('" + path + "')" +"\n\n"+

	"# Visualize data set" +"\n"+
	"display(df.head())" +"\n\n"+ 

	"# ------------------------------- Data Cleaning --------------------------------" +"\n\n"+

	"# Remove null values" +"\n"+
	"df.dropna(inplace = True)" +"\n\n"+

	"# Specify the features columns" +"\n"+
	"X = df.drop(columns = [df.columns[-1]])" +"\n\n"+

	"# Specify the target column" +"\n"+
	"y = df.iloc[:,-1]" +"\n\n"+

	"# Transform non-numerical columns into binary-type columns" +"\n"+
	"X = pd.get_dummies(X)" +"\n\n"+

	"# ----------------------------- Data Preprocessing -----------------------------" +"\n\n"+

	"# Import train_test_split class" +"\n"+ 
	"from sklearn.model_selection import train_test_split" +"\n\n"+ 

	"# Divide data set into traning and testing subsets" +"\n"+ 
	"X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = " + str(round(train_test_ratio/100,2)) + ")" +"\n\n"+

	"# ------------------------------- Model Building -------------------------------" +"\n\n"+ 

	"# Import machine learning model class" +"\n"+ 
	algorithm_import +"\n\n"+ 

	"# Instatiate machine learning model" +"\n"+ 
	algorithm_instance + " = " + algorithm_class +"\n\n"+

	"# Fit the machine learning model with the training data" +"\n"+
	algorithm_instance + '.fit(X_train, y_train)' +"\n\n"+

	"# Make predictions using the testing data" +"\n"+ 
	"y_pred = " + algorithm_instance + '.predict(X_test)' +"\n\n"+ 

	"# ------------------------------ Model Evaluation ------------------------------" +"\n\n"+

	"# Get the coefficient of determination R2" +"\n"+ 
	"from sklearn.metrics import r2_score" +"\n"+ 
	"print(r2_score(y_test, y_pred))" +"\n\n"+

	"# Get the Mean Squared Error" +"\n"+ 
	"from sklearn.metrics import mean_squared_error" +"\n"+ 
	"print(mean_squared_error(y_test, y_pred))" +"\n\n"+

	"# Get the Root Mean Squared Error" +"\n"+ 
	"print(np.sqrt(mean_squared_error(y_test, y_pred)))"

	)

st.markdown("---")

st.subheader("About the Author")

profile_picture = Image.open("Roberto Salazar - Photo.PNG")
st.write("")
st.image(profile_picture, width=250)

st.markdown("### Roberto Salazar")
st.markdown("Roberto Salazar is an Industrial and Systems engineer with a passion for coding. He obtained his bachelor's degree from University of Monterrey and his master's degree from Binghamton University, State University of New York. His research interests include data analytics, machine learning, lean six sigma, continuous improvement and simulation.")

st.markdown(":envelope: [Email](mailto:rsalaza4@binghamton.edu) | :bust_in_silhouette: [LinkedIn](https://www.linkedin.com/in/roberto-salazar-reyna/) | :computer: [GitHub](https://github.com/rsalaza4) | :page_facing_up: [Programming Articles](https://robertosalazarr.medium.com/) | :coffee: [Buy Me a Coffe](https://www.buymeacoffee.com/robertosalazarr) ")
