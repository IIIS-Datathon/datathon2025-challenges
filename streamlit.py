import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import pickle

from core.config import GROUNDING_PROMPT
from core.utils import (
    configure_gemini, load_model_and_explainer, get_gemini_local_explanation,
    load_counterfactual_assets, generate_counterfactual
)
from core.ui import render_feature_inputs

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SAILES",
    page_icon="📈",
    initial_sidebar_state="expanded"
)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Initialize resources
    model, explainer = load_model_and_explainer()
    gemini_model = configure_gemini()
    vae, scaler = load_counterfactual_assets()
    
    # Header
    st.title("📈 SAILES")
    st.subheader("Supervised Artificial Intelligence Learning Explainable System")
    st.markdown("**Understand** why our model predicts deals as Won or Lost, and **explore** live predictions.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Ask About the Model", "🔮 Live Prediction"])
    
    # ========================================================================
    # TAB 1: GLOBAL CHATBOT
    # ========================================================================
    with tab1:
        st.header("Ask About Our Model's General Strategy")
        st.markdown(
            "Ask questions like:\n"
            "- *Which competitors matter most?*\n"
            "- *What's the most important factor for winning?*\n"
            "- *How does customer history affect predictions?*"
        )
        
        # Initialize chat history
        if "chat" not in st.session_state:
            st.session_state.chat = gemini_model.start_chat(history=[
                {'role': 'user', 'parts': [GROUNDING_PROMPT]},
                {'role': 'model', 'parts': ["Understood. I am a sales analyst for Schneider Electric. I am ready to answer questions about our model's behavior based on the facts provided."]}
            ])
        
        # Display chat history (skip the grounding prompt)
        for message in st.session_state.chat.history[2:]:
            with st.chat_message(message.role):
                st.markdown(message.parts[0].text)
        
        # Chat input
        if prompt := st.chat_input("What would you like to know about the model?"):
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat.send_message(prompt)
                    if response.parts:
                        with st.chat_message("assistant"):
                            st.markdown(response.text)
                    else:
                        # This means the response was empty (likely blocked)
                        st.error("The model's response was blocked. This can happen due to safety filters. Try rephrasing your question.")
                        # Optional: Log the full response to see the safety ratings
                        print("BLOCKED RESPONSE:", response)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # ========================================================================
    # TAB 2: LIVE PREDICTION
    # ========================================================================
    with tab2:
        st.header("Get a Live Prediction for a New Opportunity")
        st.markdown("Use the sidebar to configure opportunity details, then click **Predict** to see the outcome and explanation.")
        
        # Render inputs
        input_values = render_feature_inputs()
        
        # Add a toggle to switch between audience types
        is_technical_view = st.toggle("Show Technical View", value=False, help="Switch between a business-friendly summary and a technical deep-dive for data scientists.")
        audience_type = "technical" if is_technical_view else "business"
                        
        
        # Prediction button
        if st.sidebar.button("🎯 Predict and Explain", type="primary", use_container_width=True):
            
            # Create DataFrame with correct feature order
            new_data_df = pd.DataFrame([input_values])
            
            # Ensure column order matches model training
            if hasattr(model, 'feature_names_in_'):
                new_data_df = new_data_df[model.feature_names_in_]
            
            # Display input summary
            with st.expander("📋 Input Data Summary", expanded=False):
                st.dataframe(new_data_df, use_container_width=True)
            
            # Make prediction
            with st.spinner("Making prediction..."):
                try:
                    prediction = model.predict(new_data_df)[0]
                    prediction_proba = model.predict_proba(new_data_df)[0]
                    
                    prediction_text = "WON" if prediction == 1 else "LOST"
                    probability_percent = prediction_proba[prediction] * 100
                    
                    # Display prediction
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric(
                            label="Prediction",
                            value=prediction_text,
                            delta=f"{probability_percent:.1f}% confident"
                        )
                    
                    # Calculate SHAP values
                    with st.spinner("Generating explanation..."):
                        shap_values_array = explainer.shap_values(new_data_df)
                        
                        # Extract SHAP values for positive class
                        # For binary classification: shap_values_array is list of 2 arrays
                        if isinstance(shap_values_array, list):
                            shap_values_won = shap_values_array[1][0]  # Class 1, first sample
                        else:
                            shap_values_won = shap_values_array[0,:,1]  # Single array case
                        
                        # Generate waterfall plot
                        st.subheader("📊 Why did the model decide this?")
                        st.markdown("Factors pushing toward **Won** (red/pink) vs **Lost** (blue):")
                        
                        # Get base value (expected value for positive class)
                        if isinstance(explainer.expected_value, (list, np.ndarray)):
                            if len(explainer.expected_value) > 1:
                                base_value = float(explainer.expected_value[1])  # Class 1
                            else:
                                base_value = float(explainer.expected_value[0])
                        else:
                            base_value = float(explainer.expected_value)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_values_won,
                                base_values=base_value,
                                data=new_data_df.iloc[0].values,
                                feature_names=new_data_df.columns.tolist()
                            ),
                            show=False
                        )
                        st.pyplot(fig, bbox_inches='tight', use_container_width=True)
                        plt.close()
                        
                        # Generate LLM explanation
                        st.subheader("🤖 Explanation")
                        
                        # Get top 5 factors
                        feature_names = new_data_df.columns.tolist()
                        top_indices = np.argsort(np.abs(shap_values_won))[-5:]
                        top_factors = {
                            feature_names[i]: float(shap_values_won[i])
                            for i in reversed(top_indices)
                        }
                        
                        explanation = get_gemini_local_explanation(
                            gemini_model,
                            prediction_text,
                            top_factors,
                            new_data_df.to_numpy()[0],
                            audience_type
                        )
                        st.info(explanation)

                        # Generate Counterfactual if prediction is LOST
                        if prediction_text == "LOST":
                            st.subheader("💡 How to Win This Deal (Counterfactual)")
                            with open("./model/model_7features.pkl", 'rb') as f:
                                model_7_features = pickle.load(f)
                            
                            cf_result = generate_counterfactual(model_7_features, vae, scaler, new_data_df)
                            if cf_result:
                                if isinstance(cf_result, tuple):
                                    st.markdown(cf_result[0])
                                    st.dataframe(cf_result[1], use_container_width=True, hide_index=True)
                                else:
                                    st.warning(cf_result)

                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    st.exception(e)

if __name__ == "__main__":
    main()