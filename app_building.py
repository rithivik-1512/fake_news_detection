import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# ---------------------------------------------
# page configuration
# ---------------------------------------------
st.set_page_config(
    page_title='TruthLens - Fake News Detector',
    page_icon='🔍',
    layout='wide'
)

# ---------------------------------------------
# CSS styling
# ---------------------------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
          color: white;
        }
        .main-header{
          text-align : center;
          color : white;
          background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
          padding : 20px;
          border-radius : 15px;
          margin-bottom:20px;
        }
        .side-bar{
          text-align : center;
          color : white;
          padding : 20px;
        }
        .stRadio > label{
          color: white !important;
          font-size : 18px;
          font-weight : 600;
        }
        .stRadio > div {
          background-color: rgba(255, 255, 255, 0.1);
          padding: 15px;
          border-radius: 10px;
        }
        .metric-card {
          text-align:center;
          background-color:White;
          border-radius:10px;
          padding:20px;
          border-left:5px solid #3b82f6;
          margin:10px 0;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          transition: all 0.3s;
        }
        .metric-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        }
        .metric-card h3 {
          margin: 10px 0;
        }
        .metric-value {
          font-size: 48px !important;
          font-weight: 700 !important;
          color: #1e293b !important;
          margin: 15px 0 !important;
          line-height: 1.2 !important;
        }

        /* CSV Upload / Rules Header */
        .rules-header {
            background: #0f172a;
            padding: 32px;
            border-radius: 12px;
            margin-top: 25px;
            border-left: 6px solid #3b82f6;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.25);
        }
        .rules-header h2 {
            color: #e2e8f0;
            margin-bottom: 10px;
            font-weight: 700;
        }
        .rules-header h4 {
            color: #cbd5e1;
            margin-bottom: 22px;
            font-weight: 400;
        }
        .rules-header ul {
            list-style-type: none;
            padding-left: 0;
        }
        .rules-header li {
            margin: 14px 0;
            font-size: 17px;
            color: #f1f5f9;
            line-height: 1.6;
        }
        .rules-header code {
            background-color: #1e293b;
            padding: 5px 10px;
            border-radius: 6px;
            color: #93c5fd;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 14px;
            border: 1px solid #334155;
            box-shadow: inset 0 0 4px rgba(0, 0, 0, 0.4);
        }
        .rules-header b {
            color: #fbbf24;
            font-weight: 600;
        }

        /* Post page input styling */
        .post-input label {
            font-weight: 600;
            color: #e2e8f0;
            font-size: 16px;
        }
        .post-input input, .post-input textarea {
            background-color: #1e293b !important;
            color: #f1f5f9 !important;
            border-radius: 6px;
            border: 1px solid #334155;
            padding: 10px;
            width: 100%;
            font-family: 'Consolas', 'Courier New', monospace;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# main app
# ---------------------------------------------
def main():
    # Load model and vectorizer
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfifd_vectorizer.pkl")

    # Load sample data for Home page metrics
    df = pd.read_csv("test_df.csv")
    x = df['content']
    y = df['label']
    x_num = vectorizer.transform(x)
    ypred = model.predict(x_num)
    accuracy_val = round(accuracy_score(y, ypred), 3)
    precision_val = round(precision_score(y, ypred), 3)
    recall_val = round(recall_score(y, ypred), 3)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""<h2 style="text-align:center;font-size:28px">🔍 TruthLens</h2>""", unsafe_allow_html=True)
        st.markdown("""<p style="text-align:center;font-size:18px;margin-top:-10px;color:#e0e7ff">A Fake News Detector</p>""", unsafe_allow_html=True)
        st.write("-----")
        page = st.radio(
            "📒 Navigation",
            ["🏠 Home", "🎯 Predict", "📝 Post"]
        )
        st.write("----")
        st.markdown("""
            <h3>ℹ️ About</h3>
            <p style="color: #e0e7ff; font-size: 13px;">
            TruthLens uses advanced Natural Language Processing to detect fake news and misinformation. Stay informed, stay safe! ✨
            </p>
        """, unsafe_allow_html=True)

    # Home Page
    if page == "🏠 Home":
        st.markdown("<div class='main-header'><h1>🔍 Welcome to TruthLens</h1><p>Your AI-Powered Fake News Detection System</p></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                <h3 style='color:red; margin-bottom: 5px;'>🎯 Accuracy</h3>
                <div class="metric-value">{accuracy_val}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                <h3 style='color:green; margin-bottom: 5px;'>📌 Precision</h3>
                <div class="metric-value">{precision_val}</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                <h3 style='color:blue; margin-bottom: 5px;'>📈 Recall</h3>
                <div class="metric-value">{recall_val}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y, ypred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='PuBu',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            linewidths=1,
            linecolor='white',
            annot_kws={"size": 16, "weight": 'bold', "color": 'black'},
            cbar=False
        )
        ax.set_xlabel("Predicted", fontsize=14, fontweight='bold', color='#1e293b')
        ax.set_ylabel("Actual", fontsize=14, fontweight='bold', color='#1e293b')
        ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold', color='#1e293b', pad=20)
        ax.tick_params(axis='x', labelsize=12, colors='#1e293b')
        ax.tick_params(axis='y', labelsize=12, colors='#1e293b')
        st.pyplot(fig)

    # Predict Page
    if page == "🎯 Predict":
        st.markdown("""
            <div class="main-header">
                <h1>🚀 Upload and Predict</h1>
                <p>📂 Drop your CSV file here to classify news articles instantly</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="rules-header">
            <h2>📌 CSV Upload Rules</h2>
            <h4>⚠️ Please follow the below rules before uploading your CSV file</h4>
            <ul class="rules-list">
                <li>📄 Upload only <code>.csv</code> files</li>
                <li>📌 Your CSV must contain one column: <code>content</code></li>
                <li>🔤 Column name must be exactly: <code>content</code> (lowercase)</li>
                <li>🚫 No <b>missing / empty / null values</b> in any row</li>
                <li>📰 Each row must contain one <b>complete news article</b> text</li>
                <li>🎯 Do not include <b>labels</b> (0 or 1) — the model will predict them</li>
            </ul>
        </div>""", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "📂 Drag & Drop or Browse CSV File",
            type=['csv'],
            help="Upload a CSV file with news articles to predict"
        )

        if uploaded_file is not None:
            st.success("✅ File uploaded successfully!")
            user_df = pd.read_csv(uploaded_file)
            if 'content' not in user_df.columns:
                st.error("❌ CSV must contain a column named 'content'")
            else:
                X_user = vectorizer.transform(user_df['content'])
                user_predictions = model.predict(X_user)
                user_df['prediction'] = user_predictions
                st.markdown("### 📰 Prediction Results")
                st.dataframe(user_df)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=user_df.to_csv(index=False).encode('utf-8'),
                    file_name='predictions.csv',
                    mime='text/csv'
                )

    # Post Page
    if page == "📝 Post":
        st.markdown("""
            <div class="main-header">
                <h1>🖊️ Write and Predict</h1>
                <p>📝 Enter a Title and Subject, then predict if it is fake or real</p>
            </div>
        """, unsafe_allow_html=True)

        title = st.text_input("Title", placeholder="Enter news title here")
        subject = st.text_area("Subject", placeholder="Enter news content here", height=200)

        if st.button("Predict Post"):
            if title.strip() == "" or subject.strip() == "":
                st.error("❌ Please enter both Title and Subject")
            else:
                combined_content = title + " " + subject
                X_post = vectorizer.transform([combined_content])
                post_prediction = model.predict(X_post)[0]
                result_color = "#22c55e" if post_prediction == 1 else "#ef4444"  # green for real, red for fake
                st.markdown(f"<h3 style='color:{result_color};'>Prediction: {'Real' if post_prediction == 1 else 'Fake'}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
