import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Machine Risk Predictor",
    page_icon="⚙️",
    layout="wide"
)

# Title and description
st.title("⚙️ Machine Risk Prediction System")
st.markdown("""
This app predicts machine failure risk based on usage patterns, age, maintenance history, and sensitivity.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Input Data", "Examples", "About"])

if page == "Examples":
    st.header("📚 Machine Sensitivity Examples")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🟢 LOW Sensitivity")
        st.info("Very robust machines")
        st.write("- Basic sterilizers")
        st.write("- Autoclaves")
        st.write("- Non-invasive thermometers")
        st.write("- Blood pressure cuffs")
    
    with col2:
        st.markdown("### 🟡 MEDIUM Sensitivity")
        st.warning("Moderately vulnerable machines")
        st.write("- Ultrasound machines")
        st.write("- Infusion pumps")
    
    with col3:
        st.markdown("### 🔴 HIGH Sensitivity")
        st.error("Fragile/Critical machines")
        st.write("- Ventilators")
        st.write("- Life-support systems")
        st.write("- MRI scanners")
        st.write("- CT scanners")
        st.write("- Defibrillators")
        st.write("- Dialysis machines")

elif page == "About":
    st.header("ℹ️ About This App")
    st.markdown("""
    ### How it works:
    1. **Input machine data** - Enter details for each machine
    2. **Risk calculation** - The app normalizes the data and calculates risk scores
    3. **Time-to-failure prediction** - Predicts when machines might need attention
    4. **Visualization** - Shows risk vs time-to-failure relationships
    
    ### Risk Factors:
    - **Usage hours** (50% weight)
    - **Machine age** (30% weight)
    - **Maintenance overdue** (20% weight)
    
    ### Sensitivity multipliers:
    - Low sensitivity: 0.5x
    - Medium sensitivity: 1x
    - High sensitivity: 2x
    """)

else:  # Input Data page
    st.header("📊 Machine Data Input")
    
    # Initialize session state for storing machine data
    if 'machines_data' not in st.session_state:
        st.session_state.machines_data = pd.DataFrame(columns=[
            'usage_hours', 'age_years', 'maintenance_overdue_days',
            'daily_usage', 'alpha', 'sensitivity'
        ])
    
    # Input form for new machine
    with st.form("machine_input_form"):
        st.subheader("Add New Machine")
        
        col1, col2 = st.columns(2)
        
        with col1:
            usage_hours = st.number_input("Total usage hours", min_value=0.0, value=1000.0, step=100.0)
            age_years = st.number_input("Age (years)", min_value=0.0, value=5.0, step=0.5)
            maintenance_overdue = st.number_input("Maintenance overdue (days)", min_value=0.0, value=0.0, step=1.0)
        
        with col2:
            daily_usage = st.number_input("Average daily usage (hours)", min_value=0.1, value=8.0, step=0.5)
            lifespan_alpha = st.number_input("Expected lifespan (hours)", min_value=1.0, value=50000.0, step=1000.0)
            sensitivity = st.selectbox("Sensitivity", ["low", "medium", "high"])
        
        submitted = st.form_submit_button("Add Machine")
        
        if submitted:
            new_machine = pd.DataFrame({
                'usage_hours': [usage_hours],
                'age_years': [age_years],
                'maintenance_overdue_days': [maintenance_overdue],
                'daily_usage': [daily_usage],
                'alpha': [lifespan_alpha],
                'sensitivity': [sensitivity]
            })
            
            st.session_state.machines_data = pd.concat([st.session_state.machines_data, new_machine], ignore_index=True)
            st.success("Machine added successfully!")
    
    # Display current machines
    if not st.session_state.machines_data.empty:
        st.subheader("Current Machines")
        st.dataframe(st.session_state.machines_data, use_container_width=True)
        
        # Button to clear all data
        if st.button("Clear All Machines"):
            st.session_state.machines_data = pd.DataFrame(columns=[
                'usage_hours', 'age_years', 'maintenance_overdue_days',
                'daily_usage', 'alpha', 'sensitivity'
            ])
            st.rerun()
        
        # Analysis section
        st.header("📈 Analysis Results")
        
        data = st.session_state.machines_data.copy()
        
        # Normalization function
        def normalize(column):
            if column.max() == column.min():
                return column / column.max() if column.max() != 0 else column
            return (column - column.min()) / (column.max() - column.min())
        
        # Calculate normalized values
        data['usage_norm'] = normalize(data['usage_hours'])
        data['age_norm'] = normalize(data['age_years'])
        data['maint_norm'] = normalize(data['maintenance_overdue_days'])
        
        # Define weights
        weights = {
            'usage': 0.5,
            'age': 0.3,
            'maintenance': 0.2
        }
        
        # Calculate risk score
        data['risk_score'] = (
            data['usage_norm'] * weights['usage'] +
            data['age_norm'] * weights['age'] +
            data['maint_norm'] * weights['maintenance']
        ) * 100
        
        # Map sensitivity to beta
        def get_beta(choice):
            if choice == "low":
                return 0.5
            elif choice == "medium":
                return 1
            elif choice == "high":
                return 2
            else:
                return 1
        
        data['beta'] = data['sensitivity'].apply(get_beta)
        
        # Predict time to failure
        data['predicted_TTF_hours'] = np.maximum(
            data['alpha'] - (data['beta'] * data['risk_score'] * 10),
            0
        )
        
        data['predicted_TTF_days'] = data['predicted_TTF_hours'] / data['daily_usage']
        
        # Risk classification
        def risk_flag(days):
            if pd.isna(days):
                return "Invalid"
            elif days < 30:
                return "🔴 High Risk"
            elif days < 180:
                return "🟡 Moderate Risk"
            else:
                return "🟢 Safe"
        
        data['risk_flag'] = data['predicted_TTF_days'].apply(risk_flag)
        
        # Display results
        st.subheader("Prediction Results")
        
        results_df = data[['sensitivity', 'risk_score', 'predicted_TTF_days', 'risk_flag']].round(2)
        
        # Color-code the risk flag column
        def color_risk_flag(val):
            if "High" in val:
                return 'background-color: #ffcccc'
            elif "Moderate" in val:
                return 'background-color: #ffffcc'
            elif "Safe" in val:
                return 'background-color: #ccffcc'
            return ''
        
        styled_results = results_df.style.applymap(color_risk_flag, subset=['risk_flag'])
        st.dataframe(styled_results, use_container_width=True)
        
        # Visualization
        if len(data) > 1:
            st.subheader("📊 Risk Score vs Time to Failure")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = {"low": "green", "medium": "orange", "high": "red"}
            
            for s in data['sensitivity'].unique():
                subset = data[data['sensitivity'] == s]
                ax.scatter(subset['risk_score'], subset['predicted_TTF_days'],
                          label=s.capitalize(), color=colors.get(s, 'blue'), s=80, alpha=0.7)
            
            # Add trendline
            if len(data) > 1:
                z = np.polyfit(data['risk_score'], data['predicted_TTF_days'], 1)
                p = np.poly1d(z)
                ax.plot(data['risk_score'], p(data['risk_score']), 'b--', alpha=0.5, label='Trend')
            
            ax.set_xlabel("Risk Score", fontsize=12)
            ax.set_ylabel("Predicted TTF (Days)", fontsize=12)
            ax.set_title("Risk Score vs Time to Failure", fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Export options
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export to CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="machine_risk_predictions.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export plot
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    label="📥 Download Plot (PNG)",
                    data=buf,
                    file_name="risk_analysis_plot.png",
                    mime="image/png"
                )
        
        elif len(data) == 1:
            st.info("Add at least one more machine to see the comparison plot and trendline.")
            
            # Show single machine metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score", f"{data['risk_score'].values[0]:.1f}")
            
            with col2:
                ttf = data['predicted_TTF_days'].values[0]
                st.metric("Time to Failure", f"{ttf:.1f} days")
            
            with col3:
                st.metric("Risk Level", data['risk_flag'].values[0])
    
    else:
        st.info("👆 Add your first machine using the form above to get started!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Machine Risk Predictor v1.0")
st.sidebar.markdown("Predict and visualize machine failure risks")