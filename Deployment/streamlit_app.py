import streamlit as st
import requests

st.set_page_config(page_title="AI Text Detector", page_icon="🤖")

st.title("🤖 AI Text Detection")
st.write("Detect if text is written by AI or Human")

text_input = st.text_area("Enter text to analyze:", height=200, placeholder="Paste your text here...")

if st.button("Analyze", type="primary"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    "https://hamza-003-ai-text-detection.hf.space/predict",
                    json={"text": text_input}
                )
                response.raise_for_status()
                result = response.json()
                
                st.success("Analysis Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", result["prediction"])
                with col2:
                    st.metric("Confidence", f"{result['confidence_score']*100:.2f}%")
                
                st.subheader("Probabilities")
                st.progress(result["probabilities"]["ai"], text=f"AI: {result['probabilities']['ai']*100:.1f}%")
                st.progress(result["probabilities"]["human"], text=f"Human: {result['probabilities']['human']*100:.1f}%")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter some text to analyze")
