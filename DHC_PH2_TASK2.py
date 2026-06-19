#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import zipfile
import numpy as np
import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Concatenate,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2


# In[2]:


zip_path = "E:\\housing_dataset_with_images.zip"
extract_path = "housing_dataset"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset Extracted Successfully!")


csv_path = os.path.join(extract_path, "housing.csv")
df = pd.read_csv(csv_path)

print(df.head())


# In[3]:


IMG_SIZE = (128, 128)

image_data = []

for image_name in df["image_name"]:

    image_path = os.path.join(
        extract_path,
        "images",
        image_name
    )

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    img = np.array(img) / 255.0

    image_data.append(img)

image_data = np.array(image_data)

print("Images Shape:", image_data.shape)


X_tabular = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft",
        "garage_age"
    ]
]

y = df["price"]


# In[4]:


scaler = StandardScaler()
X_tabular = scaler.fit_transform(X_tabular)

(
    X_img_train,
    X_img_test,
    X_tab_train,
    X_tab_test,
    y_train,
    y_test
) = train_test_split(
    image_data,
    X_tabular,
    y,
    test_size=0.20,
    random_state=42
)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = False

image_input = Input(shape=(128, 128, 3))

x = base_model(image_input)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)


# In[5]:


tabular_input = Input(shape=(4,))

t = Dense(32, activation="relu")(tabular_input)
t = Dense(16, activation="relu")(t)


combined = Concatenate()([x, t])

z = Dense(64, activation="relu")(combined)
z = Dense(32, activation="relu")(z)

output = Dense(1)(z)

model = Model(
    inputs=[image_input, tabular_input],
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()


history = model.fit(
    [X_img_train, X_tab_train],
    y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=8
)


predictions = model.predict(
    [X_img_test, X_tab_test]
)


mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)


# In[8]:


print("\n==========================")
print("MODEL PERFORMANCE")
print("==========================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))


results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions.flatten()
})

print("\nSample Predictions:")
print(results.head(10))


# In[ ]:




