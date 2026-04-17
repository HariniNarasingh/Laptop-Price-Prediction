import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Laptop Price Prediction System", layout="wide")

# Remove anchor links beside headings
st.markdown("""
<style>
h1 a, h2 a, h3 a {display:none !important;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR (Theme + Accuracy)
# ----------------------------
mode = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])

if mode == "Dark":
    st.markdown("""
    <style>
    .stApp {background-color:#0e1117; color:white;}
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# TITLE
# ----------------------------
st.title("Laptop Price Prediction System")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("laptop_data.csv")

required_cols = ["Brand","RAM","SSD","ScreenSize","Weight","Price"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

# ----------------------------
# PREPROCESSING
# ----------------------------
le = LabelEncoder()
df["Brand_encoded"] = le.fit_transform(df["Brand"])

X = df[["Brand_encoded","RAM","SSD","ScreenSize","Weight"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

accuracy = r2_score(y_test, model.predict(X_test))
st.sidebar.info(f"Model Accuracy: {accuracy:.2f}")

# ----------------------------
# DATASET PREVIEW BUTTON
# ----------------------------
if st.button("Dataset Preview"):
    st.dataframe(df.head(20))

# ----------------------------
# TABS (LIKE YOUR SCREENSHOT)
# ----------------------------
tab1, tab2 = st.tabs(["Price Prediction", "Compare Laptops"])

# ==================================================
# TAB 1 → PRICE PREDICTION
# ==================================================
with tab1:
    st.subheader("Enter Laptop Details")

    brand = st.selectbox("Brand", sorted(df["Brand"].unique()))
    ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64])
    ssd = st.selectbox("SSD (GB)", [128, 256, 512, 1024, 2048])
    screen = st.selectbox("Screen Size", [13.3, 14, 15.6, 16, 17])
    weight = st.slider("Weight (kg)", 1.0, 3.5, 2.0)

    if st.button("Predict Price"):
        brand_encoded = le.transform([brand])[0]

        input_df = pd.DataFrame({
            "Brand_encoded": [brand_encoded],
            "RAM": [ram],
            "SSD": [ssd],
            "ScreenSize": [screen],
            "Weight": [weight]
        })

        predicted_price = model.predict(input_df)[0]

        st.success(f"Predicted Laptop Price: ₹ {int(predicted_price):,}")

        # -------- Graph --------
        avg_price = df["Price"].mean()

        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            ["Average Price", "Predicted Price"],
            [avg_price, predicted_price]
        )

        # Auto color logic
        bars[0].set_color("#4A90E2")  # blue
        bars[1].set_color("green" if predicted_price >= avg_price else "red")

        ax.set_title("Price Comparison")
        ax.set_ylabel("Price (₹)")
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"₹{int(height):,}",
                ha='center',
                va='bottom'
            )

        st.pyplot(fig)

# ==================================================
# TAB 2 → COMPARE LAPTOPS
# ==================================================
with tab2:
    st.subheader("Compare Laptops")

    n = st.selectbox("Number of laptops to compare", [2, 3])

    results = []

    for i in range(n):
        st.markdown(f"### Laptop {i+1}")

        b = st.selectbox(f"Brand {i+1}", df["Brand"].unique(), key=f"b{i}")
        r = st.selectbox(f"RAM {i+1}", [4, 8, 16, 32], key=f"r{i}")
        s = st.selectbox(f"SSD {i+1}", [128, 256, 512], key=f"s{i}")
        sc = st.selectbox(f"Screen {i+1}", [13.3, 14, 15.6], key=f"sc{i}")
        w = st.slider(f"Weight {i+1}", 1.0, 3.5, 2.0, key=f"w{i}")

        enc = le.transform([b])[0]

        data = pd.DataFrame(
            [[enc, r, s, sc, w]],
            columns=["Brand_encoded","RAM","SSD","ScreenSize","Weight"]
        )

        price = model.predict(data)[0]
        results.append(price)

    if st.button("Compare Laptops"):
        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            [f"Laptop {i+1}" for i in range(n)],
            results
        )

        ax.set_title("Laptop Comparison")
        ax.set_ylabel("Price (₹)")
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"₹{int(height):,}",
                ha='center',
                va='bottom'
            )

        st.pyplot(fig)
