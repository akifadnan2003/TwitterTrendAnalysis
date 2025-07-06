# **AI-Powered Tweet Engagement Analyzer**

This is a full-stack data science project that predicts the potential engagement (likes) of a tweet and provides AI-driven suggestions to improve its content. The system uses a machine learning model trained on both the tweet's text and the user's follower count to make nuanced predictions.

The application is served through a modern web interface where a user can input their tweet text and follower count to receive instant feedback.


## **Technology Stack**

* **Backend:** Python, FastAPI, Uvicorn  
* **Frontend:** JavaScript, React.js  
* **Machine Learning:** Scikit-learn, Pandas, NumPy  
* **Natural Language Processing (NLP):** Sentence-Transformers, KeyBERT  
* **Model Serialization:** Joblib

## **How to Run This Project**

Follow these steps to set up and run the application on your local machine.

### **1\. Prerequisites**

* Python 3.8+  
* Node.js and npm  
* Git

### **2\. Clone the Repository**

git clone https://github.com/akifadnan2003/TwitterTrendAnalysis.git  
cd TwitterTrendAnalysis

### **3\. Set Up the Environment**

Install the necessary dependencies for both the backend and frontend.

\# Install Python dependencies for the backend  
pip install \-r requirements.txt

\# Navigate to the frontend folder and install npm packages  
cd frontend  
npm install  
cd ..

### **4\. IMPORTANT: Place the Dataset**

Dataset Name: US Congressional Tweets Dataset

Download Link: https://www.kaggle.com/datasets/oscaryezfeijo/us-congressional-tweets-dataset

Instructions: After downloading, place the tweets.json and users.json files into the data/raw/ directory inside the project

**Note:** The src/modeling/train.py script is configured to read this specific file from the root directory. It will then use the original text and Reach columns, but it will generate **synthetic 'Likes' data** to solve the "zero-like" problem in the original dataset, ensuring the model provides dynamic predictions for the demo.

### **5\. Train the Machine Learning Model**

Before you can run the API, you must train the model. Run the training script from the **root directory** of the project.

python \-m src.modeling.train

This script will process the data, train the advanced multi-input model, and save the final model file (.pkl) to the models/ directory.

### **6\. Run the Full-Stack Application**

You need to run the backend and frontend in **two separate terminals**.

#### **Terminal 1: Start the Backend Server**

From the project's root directory:

uvicorn api.main:app \--reload

The API will be running at http://127.0.0.1:8000.

#### **Terminal 2: Start the Frontend Application**

From the project's root directory:

cd frontend  
npm start

This will automatically open a new tab in your web browser at http://localhost:3000.

You can now use the application\!

---

## **License & Usage**

This project is released under the GPL-3.0 License. See the `LICENSE` file for more details.

This is an academic project developed for a final year submission at Bursa Technical University. It is intended for educational and demonstrational purposes. While the code is public for portfolio and review purposes, direct commercial use or redistribution without acknowledgment is discouraged.
