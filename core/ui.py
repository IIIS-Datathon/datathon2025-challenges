import streamlit as st
from .config import FEATURE_CONFIG

def render_feature_inputs():
    """Render all feature inputs in sidebar and return collected values."""
    st.sidebar.header("📝 Enter Opportunity Data")
    
    values = {}
    
    # Group features by category
    st.sidebar.subheader("Customer Metrics")
    for feature in ['cust_hitrate', 'cust_interactions', 'cust_contracts', 'cust_in_iberia']:
        values[feature] = render_single_input(feature, FEATURE_CONFIG[feature])
    
    st.sidebar.subheader("Opportunity Details")
    for feature in ['opp_old', 'opp_month']:
        values[feature] = render_single_input(feature, FEATURE_CONFIG[feature])
    
    st.sidebar.subheader("Product History")
    for feature in ['product_A', 'product_A_sold_in_the_past', 'product_A_recommended',
                    'product_B_sold_in_the_past', 'product_C', 'product_D']:
        values[feature] = render_single_input(feature, FEATURE_CONFIG[feature])
    
    st.sidebar.subheader("Competitive Landscape")
    for feature in ['competitor_X', 'competitor_Y', 'competitor_Z']:
        values[feature] = render_single_input(feature, FEATURE_CONFIG[feature])
    
    return values

def render_single_input(feature_name, config):
    """Render a single input widget based on configuration."""
    widget = config['widget']
    label = config['label']
    
    if widget == 'slider':
        value = st.sidebar.slider(
            label,
            min_value=float(config['min']),
            max_value=float(config['max']),
            value=float(config['default']),
            help=config.get('help'),
            step=0.01
        )
        return value
    
    elif widget == 'number':
        return st.sidebar.number_input(
            label,
            min_value=config['min'],
            value=config['default'],
            help=config.get('help')
        )
    
    elif widget == 'selectbox':
        return st.sidebar.selectbox(
            label,
            options=config['options'],
            index=config['options'].index(config['default']),
            format_func=config.get('format_func'),
            help=config.get('help')
        )