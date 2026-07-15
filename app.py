import os
import pickle

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Smart Product Recommendation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Smart Product Recommendation System")
st.write("Find products similar to your selected product using NLP.")


# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("final_data.csv")


# ----------------------------
# Load/Create Similarity Matrix
# ----------------------------

if os.path.exists("similarities.pkl"):

    with open("similarities.pkl", "rb") as f:
        similarity = pickle.load(f)

else:

    with st.spinner("Generating recommendation model... Please wait."):

        cv = CountVectorizer(
            max_features=10000,
            stop_words="english"
        )

        vectors = cv.fit_transform(df["tags"])

        similarity = cosine_similarity(vectors)

        with open("similarities.pkl", "wb") as f:
            pickle.dump(similarity, f)


# ----------------------------
# Recommendation Function
# ----------------------------

def recommend(product_name):

    index = df[df["name"] == product_name].index[0]

    distances = list(enumerate(similarity[index]))

    distances = sorted(
        distances,
        reverse=True,
        key=lambda x: x[1]
    )

    recommendations = []

    for item in distances[1:6]:

        product = df.iloc[item[0]]

        recommendations.append({

            "name": product["name"],

            "brand": product["brand"],

            "category": product["category"],

            "price": float(product["price"]),

            "discount_price": float(product["discount_price"]),

            "image": product["image"],

            "score": round(item[1] * 100, 2)

        })

    return recommendations


# ----------------------------
# Product Selection
# ----------------------------

selected_product = st.selectbox(
    "🔍 Search Product",
    sorted(df["name"].unique()),
    placeholder="Type product name..."
)

# ----------------------------
# Selected Product Details
# ----------------------------

selected = df[df["name"] == selected_product].iloc[0]

st.subheader("Selected Product")

col1, col2 = st.columns([1, 2])

with col1:

    try:
        st.image(selected["image"], use_container_width=True)
    except:
        st.write("Image not available")

with col2:

    st.markdown(f"### {selected['name']}")

    st.write(f"**Brand :** {selected['brand']}")

    st.write(f"**Category :** {selected['category']}")

    st.write(f"**Original Price : ₹{selected['price']:.2f}**")

    st.write(f"**Discount Price : ₹{selected['discount_price']:.2f}**")


if selected["price"] > selected["discount_price"]:
    discount = ((selected["price"] - selected["discount_price"]) / selected["price"]) * 100
    st.success(f"🔥 {discount:.0f}% OFF")


st.divider()


# ----------------------------
# Recommendation Button
# ----------------------------

if st.button("Recommend Products"):

    recommendations = recommend(selected_product)

    st.subheader("Recommended Products")

    cols = st.columns(5)

    for i, (col, product) in enumerate(zip(cols, recommendations)):

        with col:
            

            try:
                st.image(product["image"], use_container_width=True)
            except:
                st.write("No Image")

            st.markdown(
                f"**{product['name']}**"
            )

            st.caption(product["brand"])

            st.write(f"₹ {product['discount_price']:.2f}")

            st.caption(
                f"Similarity : {product['score']} %"
            )
            st.markdown("---")

st.markdown(
    
    """
    <div style='text-align:center; color:gray;'>
        <h5>Smart Product Recommendation System</h5>
        <p>Built using Python • NLP • CountVectorizer • Cosine Similarity • Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)