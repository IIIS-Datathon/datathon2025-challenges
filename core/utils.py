import streamlit as st
import pickle
import shap
import google.generativeai as genai
import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from .addAnalysis import top7_model
from .models import TransformerVAE as VAE


# ============================================================================
# API CONFIGURATION
# ============================================================================

@st.cache_resource
def configure_gemini():
    """Configure Gemini API with proper error handling."""
    try:
        # Try Streamlit secrets first, then environment variable
        api_key =os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            st.error("⚠️ **API Key Missing**: Please configure GOOGLE_API_KEY in Streamlit secrets or environment variables.")
            st.stop()
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    
    except Exception as e:
        st.error(f"❌ Error configuring Gemini API: {e}")
        st.stop()

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model_and_explainer():
    """Load the ML model and SHAP explainer with error handling."""
    model_path = Path('./model/random_forest_model.pkl')
    
    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path.absolute()}")
        st.info("Please ensure `random_forest_model.pkl` is in the `./model/` directory.")
        st.stop()
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        explainer = shap.TreeExplainer(model)
        
        # Validate model has required methods
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            st.error("❌ Loaded model doesn't have required prediction methods.")
            st.stop()
        
        return model, explainer
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

@st.cache_resource
def load_counterfactual_assets():
    """Load VAE and scaler for counterfactuals."""
    vae_path = Path('./model/vae_model.pth')
    scaler_path = Path('./model/scaler.pkl')

    if not vae_path.exists() or not scaler_path.exists():
        st.warning(f"⚠️ Counterfactual assets (`{vae_path.name}`, `{scaler_path.name}`) not found. 'How to Win' feature will be disabled.")
        return None, None

    # try:
        # Load the entire model saved with torch.save()
    vae = torch.load(vae_path)
    # 1. Instantiate the model architecture.
    # The input dimension must match the number of features in your data (15).
    vae = VAE(
            input_dim=7,
            categorical_indices=[6],
            numerical_indices=[0, 1, 2, 3, 4, 5],
            categorical_dims=[2],
            latent_dim=64,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            dropout=0.1,
            tau=0.5
        ).to('cuda')
    # 2. Load the saved state dictionary into the model instance.
    state_dict = torch.load(vae_path, weights_only=True)
    vae.load_state_dict(state_dict)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    vae.eval() # Set VAE to evaluation mode
    return vae, scaler
    # except Exception as e:
    #     st.error(f"❌ Error loading counterfactual assets: {e}")
    #     return None, None


# ============================================================================
# LLM EXPLANATION HELPERS
# ============================================================================

def get_gemini_local_explanation(gemini_model, prediction_text, local_shap_factors, instance, audience_type):
    """
    Generate plain-English explanation for a specific prediction.
    
    Args:
        gemini_model: Configured Gemini model instance
        prediction_text: "WON" or "LOST"
        local_shap_factors: Dict of {feature_name: shap_value}
    
    Returns:
        str: Natural language explanation
    """
    model_rules = top7_model(instance)
    factors_list = []
    for feature, value in local_shap_factors.items():
        impact_strength = "strong" if abs(value) > 0.1 else "slight"
        impact_direction = "positive" if value > 0 else "negative"
        factors_list.append(f"- '{feature}': had a {impact_strength} {impact_direction} impact (value: {value:.3f})")
    
    shap_factors_string = "\n".join(factors_list)

    if audience_type == 'technical':
        prompt = f"""
You are a Principal ML Engineer providing a technical review of a local prediction.
The model (Random Forest) predicted: {prediction_text}

Below is the evidence for this prediction.

EVIDENCE 1: LOCAL SHAP CONTRIBUTIONS
The top features contributing to this prediction (from SHAP) are:
{shap_factors_string}

EVIDENCE 2: SURROGATE MODEL RULES
The most frequent decision rules (from a surrogate model) triggered for this instance are:
{model_rules}

---
YOUR TASK:
Provide a concise, technical summary for a data scientist.
- Synthesize the evidence from SHAP and the rules.
- Explain *how* these factors (quantitatively) led to the model's output.
- It is expected that you use technical terms (SHAP, feature, contribution, etc.).
"""
    else: # Default to 'business' prompt
        prompt = f"""
You are a world-class sales analyst explaining *one specific* prediction to a sales manager.
The model predicted this opportunity will be: {prediction_text}

To help you, here is a two-part analysis:

PART 1: KEY DRIVERS FOR THIS SPECIFIC DEAL (from SHAP)
These are the top factors that *pushed* the prediction one way or the other *for this deal*:
{shap_factors_string} 

PART 2: COMMON RULES THAT APPLY (from top7_model)
For this type of opportunity, these are the common decision rules the model has learned from past data. This is *how* the model is making its decision:
{model_rules}

---
YOUR TASK:
Please write a 2-3 sentence, user-friendly summary explaining *why* the model made this prediction.
- Synthesize *both* Part 1 and Part 2 into a simple, single explanation.
- **Do not** use technical terms like 'SHAP', 'model', or 'feature'.
- **Do** use business language (e.g., 'customer's past success rate' instead of 'cust_hitrate').
- Focus on actionable insights.
"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating explanation: {str(e)}"

# ============================================================================
# COUNTERFACTUAL GENERATION
# ============================================================================

class StandaloneCounterfactualGenerator:
    """
    Generates counterfactual explanations.
    """
    def __init__(self, vae, classifier, feature_cols, categorical_features,
                 numerical_features, scaler, device, immutable_features=None):
        self.vae = vae
        self.classifier = classifier
        self.feature_cols = feature_cols
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.scaler = scaler
        self.device = device
        self.immutable_features = immutable_features if immutable_features else []
        self.immutable_indices = [i for i, feat in enumerate(feature_cols)
                                  if feat in self.immutable_features]

    def generate_counterfactual_progressive(self, x_original, target_class,
                                           num_iterations=1000, learning_rate=0.01):
        x_tensor = torch.FloatTensor(x_original).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, _ = self.vae.encode(x_tensor)
            z_original = mu

        z = z_original.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=learning_rate)

        for _ in range(num_iterations):
            optimizer.zero_grad()
            x_cf_tensor = self.vae.decode(z)
            
            # We need to inverse transform for the classifier
            x_cf_unscaled = self.scaler.inverse_transform(x_cf_tensor.detach().cpu().numpy())
            pred_proba = self.classifier.predict_proba(x_cf_unscaled)[0]
            confidence = pred_proba[target_class]

            # Loss components
            validity_loss = torch.max(torch.tensor(0.0), torch.tensor(0.6) - torch.tensor(confidence))
            proximity_loss = torch.norm(z - z_original, p=2)
            sparsity_loss = torch.norm(x_cf_tensor - x_tensor, p=1)
            
            immutability_loss = torch.tensor(0.0)
            for idx in self.immutable_indices:
                immutability_loss += 100.0 * torch.abs(x_cf_tensor[0, idx] - x_tensor[0, idx])

            total_loss = 10 * validity_loss + 0.1 * proximity_loss + 1.0 * sparsity_loss + immutability_loss
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            x_cf_final_scaled = self.vae.decode(z).cpu().numpy()
        
        return x_cf_final_scaled

def postprocess_counterfactual(x_cf, x_original, feature_cols, categorical_features):
    x_cf_processed = x_cf.copy()
    print(x_cf_processed.shape)
    for feat in categorical_features:
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            x_cf_processed[0, idx] = np.round(x_cf_processed[0, idx])
    return x_cf_processed

def evaluate_counterfactual(classifier, x_cf_unscaled, target_class):
    pred_proba = classifier.predict_proba(x_cf_unscaled)[0]
    predicted_class = np.argmax(pred_proba)
    is_valid = (predicted_class == target_class)
    return is_valid, pred_proba[target_class]

def generate_counterfactual(model, vae, scaler, input_df):
    """
    Main function to generate and format a counterfactual explanation.
    """
    if vae is None or scaler is None:
        return None

    categorical_features = ['opp_old']
    numerical_features = ['product_A_sold_in_the_past', 'product_B_sold_in_the_past', 'cust_hitrate', 'cust_interactions', 'cust_contracts', 'opp_month']
    
    immutable_features = ['age', 'race', 'sex', 'native-country']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    input_df = input_df[categorical_features+numerical_features]
    feature_cols = list(input_df.columns)
    # Scale the input data
    x_original_unscaled = input_df.values
    x_original_scaled = scaler.transform(x_original_unscaled)

    # Initialize generator
    cf_generator = StandaloneCounterfactualGenerator(
        vae=vae,
        classifier=model,
        feature_cols=feature_cols,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        scaler=scaler,
        device=device,
        immutable_features=immutable_features
    )

    # Generate
    with st.spinner("Calculating how to win this deal..."):
        x_cf_scaled = cf_generator.generate_counterfactual_progressive(
            x_original_scaled, target_class=1
        )

    # Post-process
    x_cf_scaled = postprocess_counterfactual(
        x_cf_scaled, x_original_scaled, feature_cols, categorical_features
    )

    # Inverse transform to original scale
    x_cf_unscaled = scaler.inverse_transform(x_cf_scaled)

    # Evaluate
    is_valid, confidence = evaluate_counterfactual(model, x_cf_unscaled, target_class=1)

    if not is_valid:
        return "Could not find a clear path to 'Won' for this opportunity with minor changes."

    # Format for display
    df_orig = pd.DataFrame(x_original_unscaled, columns=feature_cols)
    df_cf = pd.DataFrame(x_cf_unscaled, columns=feature_cols)

    changes = []
    for col in feature_cols:
        if col == 'opp_old' or col == 'opp_month':
            continue
        orig_val = df_orig[col].iloc[0]
        cf_val = df_cf[col].iloc[0]

        # Check for significant change
        if abs(orig_val - cf_val) > 1e-4:
            # Handle categorical features for display
            if col in categorical_features:
                change_desc = f"Change from **{'No' if orig_val == 0 else 'Yes'}** to **{'No' if cf_val == 0 else 'Yes'}**"
            else: # Numerical features
                if orig_val == 0:
                    perc_change = float('inf') # Represent as a large increase
                else:
                    perc_change = ((cf_val - orig_val) / abs(orig_val)) * 100
                
                if perc_change > 0:
                    change_desc = f"Increase by **{perc_change:.1f}%** (from {orig_val:.4f} to {cf_val:.4f})"
                else:
                    change_desc = f"Decrease by **{-perc_change:.1f}%** (from {orig_val:.4f} to {cf_val:.4f})"
            
            changes.append({
                "Feature": col,
                "Suggested Change": change_desc
            })
    
    if not changes:
        return "The model's prediction is sensitive, but no single actionable change was identified to flip the outcome."

    result_df = pd.DataFrame(changes)
    
    header = f"To flip this prediction to **WON** (with {confidence:.1%} confidence), the model suggests these changes:"
    
    return header, result_df