# Import Required Libraries
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime

# Set page config. This is the only place we set the theme.
st.set_page_config(
    layout="wide", 
    page_title="Screen Time Tracker", 
    page_icon="📱", 
    initial_sidebar_state="expanded"
)

# Set the plot theme and colors permanently
plot_template = "plotly_dark"
title_color = "cyan"

# Data Cleaning Function
def clean_data(data):
    """
    Cleans the input data by handling duplicates and missing values.
    """
    data = data.drop_duplicates()
    data = data.dropna()  # Drop rows with missing values
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Ensure Date is in datetime format
    data = data.dropna(subset=['Date'])  # Drop rows where Date couldn't be parsed
    
    # Ensure numeric columns are numeric
    numeric_cols = ['Usage (minutes)', 'Notifications', 'Times Opened']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna(subset=numeric_cols) # Drop rows where conversion failed
    
    # Convert types for memory efficiency
    data['Usage (minutes)'] = data['Usage (minutes)'].astype(int)
    data['Notifications'] = data['Notifications'].astype(int)
    data['Times Opened'] = data['Times Opened'].astype(int)
    
    return data

# Data Manipulation Function
def process_data(data):
    """
    Processes the data for insights and aggregation.
    """
    # Calculate total usage metrics by app
    app_summary = data.groupby('App').agg({
        'Usage (minutes)': 'sum',
        'Notifications': 'sum',
        'Times Opened': 'sum'
    }).reset_index().sort_values(by='Usage (minutes)', ascending=False)

    # Daily usage summary
    daily_summary = data.groupby('Date').agg({
        'Usage (minutes)': 'sum',
        'Notifications': 'sum',
        'Times Opened': 'sum'
    }).reset_index()

    return app_summary, daily_summary

# Data Visualization Function (using Plotly)
def visualize_data(app_summary, daily_summary, data_for_heatmap):
    """
    Generates and displays visualizations for the given data using Plotly.
    """
    
    # --- Row 1: App Usage & Daily Trends ---
    col1, col2 = st.columns(2)
    
    with col1:
        # App Usage by App (Barplot) - PLOTLY
        st.write("### Total Usage by App (Top 10)")
        fig = px.bar(app_summary.head(10), 
                     x='App', 
                     y='Usage (minutes)', 
                     title="Total Usage by App (Top 10)",
                     template=plot_template)
        fig.update_layout(title_font_color=title_color)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Daily Usage Trends (Lineplot) - PLOTLY
        st.write("### Daily Usage Trends")
        fig = px.line(daily_summary, 
                      x='Date', 
                      y='Usage (minutes)', 
                      title="Daily Usage Trends",
                      markers=True,
                      template=plot_template)
        fig.update_traces(line_color='lime')
        fig.update_layout(title_font_color=title_color)
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 2: Rolling Avg & Pie Chart ---
    col3, col4 = st.columns(2)
    
    with col3:
        # Rolling Average Visualization - PLOTLY
        st.write("### Daily Usage with Rolling Average")
        daily_summary['Rolling Avg'] = daily_summary['Usage (minutes)'].rolling(window=7, min_periods=1).mean()
        
        plot_data_melted = daily_summary.melt(id_vars=['Date'], 
                                              value_vars=['Usage (minutes)', 'Rolling Avg'], 
                                              var_name='Metric', 
                                              value_name='Minutes')
                                              
        fig = px.line(plot_data_melted, 
                      x='Date', 
                      y='Minutes', 
                      color='Metric', 
                      title="Daily Usage with Rolling Average",
                      template=plot_template,
                      color_discrete_map={'Usage (minutes)': 'lime', 'Rolling Avg': 'red'})
        fig.update_layout(title_font_color=title_color)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # App Usage Distribution (Pie Chart) - PLOTLY
        st.write("### App Usage Distribution (Top 10)")
        top_10 = app_summary.head(10)
        other_usage = app_summary.iloc[10:]['Usage (minutes)'].sum()
        if other_usage > 0:
            other_df = pd.DataFrame([{'App': 'Other', 'Usage (minutes)': other_usage}])
            plot_data = pd.concat([top_10, other_df], ignore_index=True)
        else:
            plot_data = top_10

        fig = px.pie(plot_data, 
                     names='App', 
                     values='Usage (minutes)', 
                     title="App Usage Distribution",
                     hole=0.3, # Donut chart
                     template=plot_template)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title_font_color=title_color)
        st.plotly_chart(fig, use_container_width=True)
        
    # --- Row 3: Histogram & Correlation ---
    col5, col6 = st.columns(2)

    with col5:
        # Usage Duration Histogram - PLOTLY
        st.write("### Daily Usage Frequency")
        fig = px.histogram(daily_summary, 
                           x='Usage (minutes)', 
                           title='Histogram of Daily Usage',
                           marginal='box', # Adds a box plot
                           template=plot_template)
        fig.update_traces(marker_color='cyan')
        fig.update_layout(title_font_color=title_color)
        st.plotly_chart(fig, use_container_width=True)
        
    with col6:
        # Correlation Heatmap - PLOTLY
        st.write("### Correlation Matrix")
        corr = data_for_heatmap[['Usage (minutes)', 'Notifications', 'Times Opened']].corr()
        fig = px.imshow(corr, 
                        text_auto=True, 
                        aspect="auto", 
                        title='Usage, Notifications, and Times Opened',
                        template=plot_template,
                        color_continuous_scale='Viridis')
        fig.update_layout(title_font_color=title_color)
        st.plotly_chart(fig, use_container_width=True)

# Improved Alert Function (with Goal Setting)
def display_alerts(daily_summary, app_summary, daily_goal):
    """
    Displays alert messages for high screen time or excessive app usage
    based on a user-defined goal.
    """
    avg_daily_usage = daily_summary['Usage (minutes)'].mean()
    st.markdown("### 🎯 Alerts & Goal Tracking")
    
    # High average daily usage alert
    if avg_daily_usage > daily_goal:
        st.warning(f"⚠️ **Over Target:** Your average daily use is **{avg_daily_usage:.2f} min**, which is **{avg_daily_usage - daily_goal:.2f} min** over your goal of **{daily_goal} min**.")
    elif avg_daily_usage > daily_goal * 0.8:
        st.info(f"ℹ️ **Approaching Target:** Your average is **{avg_daily_usage:.2f} min**, which is close to your goal of **{daily_goal} min**.")
    else:
        st.success(f"✅ **Goal Met!** Your average daily use is **{avg_daily_usage:.2f} min**, well under your goal of **{daily_goal} min**.")

    # Top app excessive usage alert
    if not app_summary.empty:
        top_app = app_summary.iloc[0]
        total_usage = app_summary['Usage (minutes)'].sum()
        top_app_percentage = (top_app['Usage (minutes)'] / total_usage) * 100

        if top_app_percentage > 50:
            st.warning(f"⚠️ **High Concentration:** Your top app, **{top_app['App']}**, accounts for **{top_app_percentage:.1f}%** of your total screen time.")
        else:
            st.success(f"✅ **Usage is well-distributed.** Your top app, **{top_app['App']}**, only accounts for **{top_app_percentage:.1f}%** of your total usage.")
    else:
        st.info("ℹ️ No app summary data to analyze for alerts.")

# --- NEW MACHINE LEARNING FUNCTION ---
def train_and_forecast(daily_summary):
    """
    Trains a simple ML model to forecast the next 7 days of usage.
    """
    st.write("### 🚀 Your 7-Day Usage Forecast")
    
    # 1. Feature Engineering
    df = daily_summary.copy()
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month
    df['lag_1'] = df['Usage (minutes)'].shift(1) # Usage from yesterday
    df = df.dropna()

    if df.empty or len(df) < 10: # Need enough data to train
        st.info("Not enough data to create a forecast (need at least 10 days).")
        return

    # 2. Define Features (X) and Target (y)
    features = ['day_of_week', 'day_of_year', 'month', 'lag_1']
    X = df[features]
    y = df['Usage (minutes)']

    # 3. Train Model (Train on all available data for best forecast)
    model = LinearRegression()
    model.fit(X, y)

    # 4. Evaluate Model (Show in-sample accuracy)
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    st.write(f"**Model Accuracy (In-Sample RMSE):** {rmse:.2f} minutes. (Lower is better)")
    st.caption("This simple model predicts future usage based on date and previous day's usage.")

    # 5. Forecast Future
    forecast_dates = pd.to_datetime([df['Date'].max() + datetime.timedelta(days=i) for i in range(1, 8)])
    last_known_usage = df['Usage (minutes)'].iloc[-1]
    
    forecast_features_list = []
    for date in forecast_dates:
        features_dict = {
            'day_of_week': date.dayofweek,
            'day_of_year': date.dayofyear,
            'month': date.month,
            'lag_1': last_known_usage # Use last known usage to predict next day
        }
        # Predict
        prediction = model.predict(pd.DataFrame([features_dict]))[0]
        # Ensure prediction is non-negative
        prediction = max(0, prediction) 
        
        forecast_features_list.append({'Date': date, 'Forecasted Usage': prediction})
        
        # Update lag for next prediction
        last_known_usage = prediction 

    forecast_df = pd.DataFrame(forecast_features_list)

    # 6. Plot Forecast
    fig = px.line(daily_summary, x='Date', y='Usage (minutes)', title="Usage Forecast vs. History", template=plot_template, labels={'Usage (minutes)': 'Actual Usage'})
    fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Forecasted Usage'], mode='lines', name='Forecast', line=dict(color='orange', dash='dash'))
    fig.update_layout(title_font_color=title_color)
    st.plotly_chart(fig, use_container_width=True)

# Streamlit Dashboard
def main():
    st.title("📊 Predictive Screen Time Manager")
    st.markdown("""
        **Features**:
        - 🤖 **ML Forecasting**: Predicts your next 7 days of usage.
        - 📈 **Interactive Visualizations**: Hover and zoom on charts.
        - 🎯 **Goal Setting**: Set and track your personal screen time goals.
    """)
    st.sidebar.title("Settings")
    st.sidebar.info("Upload your screen time data in CSV format for analysis.")
    
    # --- GOAL SETTING WIDGET ---
    if 'daily_goal' not in st.session_state:
        st.session_state.daily_goal = 240 # Default goal of 4 hours

    st.sidebar.number_input(
        "Set Your Daily Usage Goal (minutes):",
        min_value=30, 
        max_value=1440, 
        value=st.session_state.daily_goal,
        step=15,
        key='daily_goal' # This links the input to the session state
    )
    daily_goal = st.session_state.daily_goal # Get the goal
    st.sidebar.write(f"Your current goal is **{daily_goal // 60} hours, {daily_goal % 60} minutes**.")
    # --- END GOAL SETTING ---

    # File Upload
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            # Load and display data
            data = pd.read_csv(uploaded_file)
            st.write("### Raw Data (First 5 Rows):")
            st.dataframe(data.head())

            # Clean data
            cleaned_data = data.copy() # Use .copy() to avoid changing original
            cleaned_data = clean_data(cleaned_data)
            if cleaned_data.empty:
                st.error("🚨 **Error:** The uploaded file has no valid data after cleaning. Please check the file format and content.")
                return # Stop execution if data is empty

            st.write("### Cleaned Data (First 5 Rows):")
            st.dataframe(cleaned_data.head())

            # Process data
            app_summary, daily_summary = process_data(cleaned_data)

            # Key Metrics at a Glance
            st.write("### 🔑 Key Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Screen Time (min)", f"{daily_summary['Usage (minutes)'].sum():,}")
            col2.metric("Average Daily Usage (min)", f"{daily_summary['Usage (minutes)'].mean():.2f}")
            if not app_summary.empty:
                top_app = app_summary.iloc[0]
                col3.metric("Most Used App", top_app['App'])
            else:
                col3.metric("Most Used App", "N/A")
            
            # Display alerts
            display_alerts(daily_summary, app_summary, daily_goal)
            
            # --- ADD ML FORECASTING ---
            train_and_forecast(daily_summary)

            # Interactive App Summary Table with Highlight
            st.write("### Interactive App Summary Table")
            st.dataframe(app_summary.style.highlight_max(axis=0, subset=['Usage (minutes)', 'Notifications', 'Times Opened']), use_container_width=True) 
            
            # Visualize data
            visualize_data(app_summary, daily_summary, cleaned_data)
        
        except Exception as e:
            st.error(f"🚨 **An error occurred while processing the file:** {e}")
            st.error("Please ensure your file has the correct columns: 'Date', 'App', 'Usage (minutes)', 'Notifications', 'Times Opened'.")

    else:
        st.info("Upload a CSV file to begin analysis.")

# Run the Streamlit app
if __name__ == "__main__": 
    main()