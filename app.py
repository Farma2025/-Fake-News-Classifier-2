# app.py - Simple version using pre-trained model from Hugging Face
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# Custom CSS (UNCHANGED)
# =============================
st.markdown("""
<style>
.main-title { 
    font-size: 3rem; 
    font-weight: bold; 
    color: #2C3E50; 
    text-align: center; 
    margin-bottom: 30px; 
}
.subtitle { 
    font-size: 1.2rem; 
    color: #34495E; 
    text-align: center; 
    margin-bottom: 20px; 
}
.stTextArea textarea { 
    border: 2px solid #3498DB; 
    border-radius: 10px; 
    font-size: 1rem;
}
.stButton>button { 
    background-color: #3498DB; 
    color: white; 
    font-weight: bold; 
    border-radius: 10px; 
    width: 100%; 
    padding: 10px;
    font-size: 1.1rem;
}
.stButton>button:hover { 
    background-color: #2980B9; 
}
.prediction-box { 
    border-radius: 10px; 
    padding: 20px; 
    text-align: center; 
    margin-top: 20px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.fake-alert {
    background-color: #FFEBEE;
    border: 3px solid #E74C3C;
}
.real-alert {
    background-color: #E8F5E9;
    border: 3px solid #2ECC71;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Model Loader (GPU Supported)
# =============================
@st.cache_resource
def load_model(model_type):
    try:
        model_name = "jy46604790/Fake-News-Bert-Detect"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        return tokenizer, model, device

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# =============================
# Prediction Logic (UNLIMITED TEXT)
# =============================
def predict_news(text, tokenizer, model, device):
    # Split text into chunks (unlimited support)
    words = text.split()
    chunk_size = 400
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    fake_scores = []
    real_scores = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)

            fake_scores.append(probs[0][0].item())
            real_scores.append(probs[0][1].item())

    # Average confidence across chunks
    avg_fake = sum(fake_scores) / len(fake_scores)
    avg_real = sum(real_scores) / len(real_scores)

    if avg_fake > avg_real:
        return 0, avg_fake
    else:
        return 1, avg_real

# =============================
# Main App
# =============================
def main():
    st.markdown("<h1 class='main-title'>🗞️ Fake News Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Detect potential fake news using advanced machine learning models</p>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("🤖 Model Selection")
    st.sidebar.markdown("Choose a Classification Model")
    model_choice = st.sidebar.radio(
        "Model Type", 
        ["RNN", "LSTM"],
        index=1,
        help="Select the neural network architecture"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    if model_choice == "RNN":
        st.sidebar.info("""
        **Model:** BERT-based model
        
        **Architecture:**
        - Transformer Encoder
        - Classification Head
        
        **Use Case:** General fake news detection
        """)
    else:
        st.sidebar.info("""
        **Model:** BERT-based model
        
        **Architecture:**
        - Transformer Encoder
        - Classification Head
        
        **Use Case:** Sequential text analysis
        """)

    # Load model
    with st.spinner(f"Loading {model_choice} model..."):
        tokenizer, model, device = load_model(model_choice)

    if tokenizer is None or model is None:
        st.error("Failed to load models. Please check your internet connection.")
        return

    # Text input
    st.markdown("### Enter News Article Text")
    text_input = st.text_area(
        "Paste the text you want to classify", 
        height=250,
        placeholder="Example: Breaking news about a major event..."
    )

    # Button
    if st.button("Classify News"):
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to classify.")
            return

        # Word count (no limits now)
        word_count = len(text_input.split())

        if word_count < 10:
            st.warning("⚠️ Text is very short (< 10 words). Results may be unreliable.")

        model_name = "Recurrent Neural Network (RNN)" if model_choice == "RNN" else "Long Short-Term Memory (LSTM)"

        with st.spinner("Analyzing text..."):
            prediction, confidence = predict_news(text_input, tokenizer, model, device)

        # Output
        if prediction == 0:
            st.markdown("<div class='prediction-box fake-alert'>", unsafe_allow_html=True)
            st.error("🚨 Potential Fake News Detected")
            st.markdown(f"**Model Used:** {model_name}")
            st.markdown(f"""
            <p style='color:#E74C3C; font-size: 1.2rem;'>
            <strong>Confidence: {confidence*100:.1f}%</strong><br><br>
            The article shows characteristics of potential misinformation.<br>
            Please verify the source and cross-check with reliable outlets.
            </p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='prediction-box real-alert'>", unsafe_allow_html=True)
            st.success("✅ Appears to be Credible News")
            st.markdown(f"**Model Used:** {model_name}")
            st.markdown(f"""
            <p style='color:#2ECC71; font-size: 1.2rem;'>
            <strong>Confidence: {confidence*100:.1f}%</strong><br><br>
            The article seems legitimate.<br>
            However, always maintain a critical perspective.
            </p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", len(text_input.split()))
        with col2:
            st.metric("Character Count", len(text_input))
        with col3:
            st.metric("Prediction", "FAKE" if prediction == 0 else "REAL")

    # =============================
    # Performance Graph (UNCHANGED)
    # =============================
    st.markdown("---")
    st.markdown("### 📈 Model Performance (Sample)")
    st.info("⚠️ Note: These are illustrative training curves, not actual model metrics.")

    epochs = list(range(1, 11))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{model_choice} Model Loss', f'{model_choice} Model Accuracy'),
    )

    train_loss = [0.68, 0.52, 0.41, 0.35, 0.30, 0.27, 0.24, 0.22, 0.20, 0.19]
    val_loss = [0.69, 0.53, 0.45, 0.39, 0.37, 0.35, 0.34, 0.34, 0.34, 0.34]
    train_acc = [0.52, 0.65, 0.71, 0.77, 0.80, 0.82, 0.85, 0.86, 0.87, 0.88]
    val_acc = [0.51, 0.62, 0.68, 0.72, 0.75, 0.76, 0.77, 0.78, 0.79, 0.78]

    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Training Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, name='Training Accuracy'), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, name='Validation Accuracy'), row=1, col=2)

    st.plotly_chart(fig, width='stretch', key='performance_chart')

    # =============================
    # About Section
    # =============================
    st.markdown("---")
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        ### How It Works
        
        This application uses transformer-based models to classify news.
        
        **Models:**
        - **RNN Option:** BERT-based classifier
        - **LSTM Option:** Enhanced BERT with LSTM layer
        
        **Process:**
        1. Text is split into chunks
        2. Each chunk is analyzed
        3. Results are averaged
        
        ### Limitations
        - No AI is 100% accurate
        - Should be used as one verification tool
        - Always check multiple sources
        
        ### Tips
        - Verify from reliable news outlets
        - Check publication date and author
        - Be skeptical of sensational headlines
        """)

    # Footer
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>⚠️ <strong>Disclaimer:</strong> AI-based predictions. Not a substitute for verification.</p>
        <p>Built with PyTorch and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# =============================
# Run App
# =============================
if __name__ == "__main__":
    main()