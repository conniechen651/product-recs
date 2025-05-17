import streamlit as st
import pandas as pd
from collections import defaultdict
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Load the dataset
filename = "All_Beauty.csv"

df = pd.read_csv(filename)

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[["user_id", "parent_asin", "rating"]], reader)

def get_top_n(predictions, n=5):

    # Map the predictions to each user
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

@st.cache_data
def model():
    param_grid = {
        'n_epochs': [15, 20, 25],
        'n_factors': [120, 140, 155],
        'lr_all': [0.01, 0.015, 0.017, 0.02],
        'reg_all': [0.12, 0.15, 0.2], 
        'init_std_dev': [0.01, 0.05, 0.1],
    }

    # Initialize GridSearchCV with the SVD algorithm
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)

    # Perform grid search on the entire dataset
    gs.fit(data)

    # Get the best score and corresponding parameters
    print("Best RMSE score:", gs.best_score['rmse'])
    print("Best parameters:", gs.best_params['rmse'])

    # Use the best parameters to train the final model
    best_params = gs.best_params['rmse']
    svd_algo = SVD(n_factors=best_params['n_factors'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'], 
                n_epochs=best_params['n_epochs'], init_std_dev=best_params['init_std_dev'])

    # Split the data into train and test sets
    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()

    # Train the model on the trainset
    svd_algo.fit(trainset)

    # Test the model on the testset
    svd_predictions = svd_algo.test(testset)

    svd_top_n = get_top_n(svd_predictions, n=5)
    return svd_top_n
svd_top_n = model()

# Streamlit app
st.title("Production Recommendation System")
user_list = df['user_id'].unique()

user_name = st.selectbox('Select a user',user_list)

with st.spinner("Wait for it...", show_time=True):

    item_list = list(svd_top_n.items())
    for key, value in item_list:
        if key == user_name:
            st.write("Top 5 recommendations for user", key)
            for (iid, _) in value:
                url = "https://www.amazon.com/dp/"+iid
                st.write(url)

                options = Options()
                options.add_argument("--headless")  # Run Chrome in headless mode
                options.add_argument("--disable-gpu")  # Disable GPU acceleration (optional)
                options.add_argument("--no-sandbox")  # Bypass OS security model (useful in some environments)
                options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
                driver = webdriver.Chrome(options=options)  # Ensure you have the ChromeDriver installed
                driver.get(url)

                # Get the rendered HTML
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")

                # Find the image
                image = soup.find('img', id='landingImage')
                if image:
                    st.image(image['src'])
                else:
                    st.write("Image not found")

                driver.quit()
            break


