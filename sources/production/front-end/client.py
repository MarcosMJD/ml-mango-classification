# Streamlit app that shows a set o sample images
# Chooses a random image from another set of images
# Let the user to predict the variety
# Predicts the variety
# Shows the winner

import streamlit as st
import requests
import os
import random

DATA_PATH = "./data/Classification_dataset/"
SAMPLE_IMAGE = DATA_PATH + "Anwar Ratool/" + "IMG_20210630_102920.jpg"

def create_samples (data_path: str) -> list:

    classes = [subdir.name for subdir in os.scandir(data_path) if subdir.is_dir()]
    class_dirs = [subdir for subdir in os.scandir(data_path) if subdir.is_dir()]

    all_images = []
    all_classes = []

    for class_dir in class_dirs:
        for file in os.scandir(class_dir.path):
            if file.is_file():
                all_classes.append(class_dir.name)
                all_images.append(file.path)
    return list(zip(all_images, all_classes)), classes

def predict(api_uri, sample_image, sample_class):

    #files = {'mango': open(sample_image, 'rb')}
    #response = requests.post(url=api_uri, files=files)
    file = open(sample_image, 'rb')
    response = requests.post(url=api_uri, data={'filename': file , "msg":"hello" ,"type" : "multipart/form-data"}, files = { "file" : file  } )
    predictions = response.json()
    prediction = max(predictions, key=predictions.get)
    st.markdown(f"Prediction = {prediction}")
    st.markdown(f"Actual = {sample_class}")
    if prediction != sample_class: 
        st.subheader("Fail!")
    else:
        st.subheader('IA was right!')
    st.markdown(f"Detailed predictions: {predictions} ")


def render(api_gateway_base_url, data_path):

    st.title('Mango game')

    api_gateway_base_uri = f"{api_gateway_base_url}/predict"
   
    api_uri = st.text_input(
        "API Gateway Endpoint URI",
        value=api_gateway_base_uri,
        type="default",
        help="The URI of the ML server running on, <url>:<port>/predict. May be localhost or any other external url",
    )

    dataset, classes = create_samples(data_path)
    index = random.randint(0,len(dataset))
    sample_image = dataset[index][0]
    sample_class = dataset[index][1]

    st.markdown("Random sample image")
    st.image(sample_image, width=300)

    user_class = st.selectbox(
        'Which is the variety of this mango?',
        index=0,
        options = ['make your choice'] + classes,
        on_change = predict,
        args=(api_uri, sample_image, sample_class))

    st.markdown("For your reference, these are samples from each class")    
    class_samples = [
        './data/Classification_dataset/Anwar Ratool/IMG_20210630_102834.jpg',
        './data/Classification_dataset/Chaunsa (Black)/IMG_20210705_091828.jpg',
        './data/Classification_dataset/Chaunsa (Summer Bahisht)/IMG_20210704_185922.jpg',
        './data/Classification_dataset/Chaunsa (White)/IMG_20210705_100452.jpg',
        './data/Classification_dataset/Dosehri/IMG_20210629_182957.jpg',
        './data/Classification_dataset/Fajri/IMG_20210705_104139.jpg',
        './data/Classification_dataset/Langra/IMG_20210702_074756.jpg',
        './data/Classification_dataset/Sindhri/IMG_20210702_182507 - Copy.jpg'
    ]

    st.image(class_samples, width=150, caption=classes)






if __name__ == "__main__":

    api_gateway_base_url = os.getenv("API_GATEWAY_BASE_URL", "http://localhost:9000")

    render(api_gateway_base_url, DATA_PATH)
