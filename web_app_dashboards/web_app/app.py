import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import subprocess


st.set_page_config(
    page_title="Hotel Revenue Dashboard",
    page_icon="🏨",
    layout="wide",
)


def _load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                return
        except Exception:
            pass

    # fallback inline CSS
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f7fb 0%, #eef3f8 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.3rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #102a43 0%, #1d4e89 100%);
            color: white;
            box-shadow: 0 18px 40px rgba(16, 42, 67, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1, .hero p {
            margin: 0;
        }
        .metric-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: white;
            border: 1px solid rgba(16, 42, 67, 0.08);
            box-shadow: 0 8px 24px rgba(16, 42, 67, 0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_load_css()


@st.cache_data(show_spinner=False)
def load_data(uploaded_file: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def create_demo_data(n=240) -> pd.DataFrame:
    rng = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")
    np.random.seed(42)
    countries = ["United States", "United Kingdom", "Germany", "France", "India", "Australia", "Canada", "Brazil", "Spain", "Italy"]
    channels = ["Direct", "OTA", "Corporate", "Travel Agent"]
    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    df = pd.DataFrame({
        "Date": rng,
        "Total_Revenue": (np.random.gamma(5, 200, size=n) * (1 + np.sin(np.linspace(0, 6.28, n)) * 0.25)).round(2),
        "Total_Costs": (np.random.gamma(3, 120, size=n)).round(2),
        "Season": np.random.choice(seasons, size=n),
        "Occupancy_Rate": (np.random.beta(2, 5, size=n) * 100).round(1),
        "Booking_Channel": np.random.choice(channels, size=n),
        "Guest_Country": np.random.choice(countries, size=n),
    })
    df["Month_Name"] = df["Date"].dt.month_name()
    df.to_csv("assets/demo_data.csv", index=False)
    return df


def fill_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_object_dtype, is_string_dtype

    for column in cleaned.columns:
        col = cleaned[column]
        # Numeric columns: fill with median
        if is_numeric_dtype(col):
            try:
                replacement = col.median()
            except Exception:
                replacement = col.mean() if col.dropna().size > 0 else 0
            cleaned[column] = col.fillna(replacement)
        # Categorical or string-like columns: fill with mode or 'Unknown'
        elif is_categorical_dtype(col) or is_object_dtype(col) or is_string_dtype(col):
            mode_values = col.mode(dropna=True)
            replacement = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            cleaned[column] = col.fillna(replacement)
        else:
            # Fallback: attempt median for anything else, otherwise fill with 0
            try:
                cleaned[column] = col.fillna(col.median())
            except Exception:
                cleaned[column] = col.fillna(0)
    return cleaned


def create_profit_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    if {"Total_Revenue", "Total_Costs"}.issubset(enriched.columns):
        enriched["Profit"] = enriched["Total_Revenue"] - enriched["Total_Costs"]
    return enriched


def prepare_model_frame(dataframe: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    model_frame = dataframe.copy()
    for column in model_frame.select_dtypes(include=["object", "category"]).columns:
        encoder = LabelEncoder()
        model_frame[column] = encoder.fit_transform(model_frame[column].astype(str))

    features = model_frame.drop(columns=[target_column])
    target = model_frame[target_column]
    return features, target


def safe_monthly_revenue(dataframe: pd.DataFrame) -> pd.Series | None:
    if "Month_Name" in dataframe.columns and "Total_Revenue" in dataframe.columns:
        month_order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        series = dataframe.groupby("Month_Name")["Total_Revenue"].sum().reindex(month_order)
        return series.dropna()

    if "Month" in dataframe.columns and "Total_Revenue" in dataframe.columns:
        series = dataframe.groupby("Month")["Total_Revenue"].sum()
        try:
            month_order = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
            return series.reindex(month_order).dropna()
        except Exception:
            return series

    return None


def _load_html_snippets():
    header_path = Path("assets/header.html")
    footer_path = Path("assets/footer.html")

    if header_path.exists():
        try:
            with open(header_path, "r", encoding="utf-8") as f:
                st.markdown(f.read(), unsafe_allow_html=True)
        except Exception:
            pass
    else:
        # fallback hero
        st.markdown(
            """
            <div class="hero">
                <h1>Hotel Revenue Prediction & Dashboard</h1>
                <p>Upload your hotel bookings CSV to explore revenue, profit, occupancy, channels, countries, and a predictive model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


_load_html_snippets()


with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload Hotel book.csv", type=["csv"])
    st.caption("If no file is uploaded, the app will look for Hotel book.csv in the same folder.")
    if st.button("Load demo data"):
        create_demo_data()
        st.success("Demo data created — loading now.")


fallback_path = "Hotel book.csv"
dataframe = None

if uploaded_file is not None:
    dataframe = load_data(uploaded_file)
elif pd.io.common.file_exists(fallback_path):
    dataframe = pd.read_csv(fallback_path)
elif pd.io.common.file_exists("assets/demo_data.csv"):
    dataframe = pd.read_csv("assets/demo_data.csv")


if dataframe is None:
    st.info("Upload your CSV to start exploring the dashboards.")
    st.stop()


raw_data = dataframe.copy()
clean_data = fill_missing_values(dataframe)
dashboard_data = create_profit_column(clean_data)

st.sidebar.header("Model Settings")
default_target = "Total_Revenue" if "Total_Revenue" in dashboard_data.columns else dashboard_data.columns[-1]

target_column = st.sidebar.selectbox(
    "Target column",
    dashboard_data.columns.tolist(),
    index=dashboard_data.columns.get_loc(default_target) if default_target in dashboard_data.columns else 0
)

training_data = dashboard_data.copy()
model_ready = target_column in training_data.columns and training_data.shape[1] > 1

model = None
y_test = None
y_pred = None
features = None

if model_ready:
    features, target = prepare_model_frame(training_data, target_column)
    if features.shape[1] > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
        )
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><strong>Rows</strong><br>{dashboard_data.shape[0]:,}</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><strong>Columns</strong><br>{dashboard_data.shape[1]:,}</div>', unsafe_allow_html=True)
with col3:
    if "Total_Revenue" in dashboard_data.columns:
        st.markdown(f'<div class="metric-card"><strong>Total Revenue</strong><br>{dashboard_data["Total_Revenue"].sum():,.0f}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><strong>Total Revenue</strong><br>Not available</div>', unsafe_allow_html=True)
with col4:
    if "Profit" in dashboard_data.columns:
        st.markdown(f'<div class="metric-card"><strong>Total Profit</strong><br>{dashboard_data["Profit"].sum():,.0f}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><strong>Total Profit</strong><br>Not available</div>', unsafe_allow_html=True)


tab_overview, tab_revenue, tab_segments, tab_model, tab_custom, tab_data = st.tabs(
    ["Overview", "Revenue Trends", "Guest Segments", "Model", "Custom Charts", "Data"]
)


with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(raw_data.head(20), use_container_width=True)

    left, right = st.columns(2)
    with left:
        if {"Total_Revenue", "Profit"}.issubset(dashboard_data.columns):
            # interactive Plotly scatter
            fig = px.scatter(
                dashboard_data,
                x="Total_Revenue",
                y="Profit",
                hover_data=["Date", "Guest_Country", "Booking_Channel"],
                title="Revenue vs Profit",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue vs Profit chart requires Total_Revenue and Total_Costs columns.")

    with right:
        if {"Season", "Occupancy_Rate"}.issubset(dashboard_data.columns):
            fig = px.bar(
                dashboard_data.groupby("Season")["Occupancy_Rate"].mean().reset_index(),
                x="Season",
                y="Occupancy_Rate",
                title="Avg Occupancy Rate by Season",
                labels={"Occupancy_Rate": "Occupancy (%)"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Occupancy chart requires Season and Occupancy_Rate columns.")


with tab_revenue:
    monthly_revenue = None
    if "Date" in dashboard_data.columns and "Total_Revenue" in dashboard_data.columns:
        revenue_frame = dashboard_data.copy()
        revenue_frame["Date"] = pd.to_datetime(revenue_frame["Date"], errors="coerce")
        revenue_frame["Month_Name"] = revenue_frame["Date"].dt.month_name()
        monthly_revenue = safe_monthly_revenue(revenue_frame)
    else:
        monthly_revenue = safe_monthly_revenue(dashboard_data)

    if monthly_revenue is not None and not monthly_revenue.empty:
        # Use production-ready matplotlib rendering for monthly revenue
        revenue_frame = dashboard_data.copy()
        revenue_frame["Date"] = pd.to_datetime(revenue_frame.get("Date", revenue_frame.get("date", None)), errors="coerce")
        revenue_frame["Month_Name"] = revenue_frame["Date"].dt.month_name()

        month_order = [
            'January', 'February', 'March', 'April',
            'May', 'June', 'July', 'August',
            'September', 'October', 'November', 'December'
        ]

        monthly_revenue_prod = (
            revenue_frame
            .groupby('Month_Name')['Total_Revenue']
            .sum()
            .reindex(month_order)
        )

        monthly_revenue_prod = monthly_revenue_prod.dropna()

        fig, ax = plt.subplots(figsize=(14, 7))
        # Styling to match dashboard theme
        fig.patch.set_facecolor('none')
        ax.set_facecolor('#ffffff')
        ax.plot(
            monthly_revenue_prod.index,
            monthly_revenue_prod.values,
            marker='o',
            linewidth=3,
            color='#1d4e89',
            markeredgecolor='#0f2540',
            markerfacecolor='#2a6fb7',
        )

        for x, y in zip(monthly_revenue_prod.index, monthly_revenue_prod.values):
            ax.annotate(f'{y:,.0f}', (x, y), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=10, fontweight='600', color='#0f2540')

        ax.set_title('Monthly Revenue Trend', fontsize=18, fontweight='bold')
        ax.set_title('Monthly Revenue Trend', fontsize=20, fontweight='700', color='#071735')
        ax.set_xlabel('Month', fontsize=12, color='#0f2540')
        ax.set_ylabel('Total Revenue', fontsize=12, color='#0f2540')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', rotation=45)
        # Axis label and tick coloring
        ax.tick_params(axis='x', colors='#0f2540')
        ax.tick_params(axis='y', colors='#0f2540')
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Add either a Month column or a Date column to view the monthly revenue trend.")

    if target_column in dashboard_data.columns:
        st.download_button(
            "Download Enriched Data",
            data=dashboard_data.to_csv(index=False).encode("utf-8"),
            file_name="hotel_predictions_ready.csv",
            mime="text/csv",
        )


with tab_segments:
    left, right = st.columns(2)

    with left:
        if "Booking_Channel" in dashboard_data.columns:
            channel_counts = dashboard_data["Booking_Channel"].value_counts()
            fig = px.pie(names=channel_counts.index, values=channel_counts.values, title="Booking Channel Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Booking_Channel column not found.")

    with right:
        if "Guest_Country" in dashboard_data.columns:
            top_countries = dashboard_data["Guest_Country"].value_counts().head(10)
            fig = px.bar(x=top_countries.values, y=top_countries.index, orientation="h", title="Top 10 Guest Countries", labels={"x": "Guests", "y": "Country"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Guest_Country column not found.")


with tab_model:
    if model is None or y_test is None or y_pred is None:
        st.warning("The model could not be trained. Make sure the selected target column exists and there are usable features.")
    else:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.2f}")
        with m2:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.metric("RMSE", f"{rmse:,.2f}")
        with m3:
            st.metric("R²", f"{r2_score(y_test, y_pred):.4f}")

        left, right = st.columns(2)
        with left:
            fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            importances = pd.DataFrame(
                {
                    "Feature": features.columns,
                    "Importance": model.feature_importances_,
                }
            ).sort_values(by="Importance", ascending=False).head(10)
            fig = px.bar(importances, x="Importance", y="Feature", orientation="h", title="Top 10 Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

        sample_index = 0
        sample = features.iloc[[sample_index]]
        prediction = model.predict(sample)[0]
        st.success(f"Sample prediction for row {sample_index}: {prediction:,.2f}")


with tab_custom:
    st.subheader("Custom Chart Builder")
    st.write("🎨 Select X and Y axis columns to generate custom dashboards")
    
    # Get all columns for selection
    all_cols = dashboard_data.columns.tolist()
    numeric_cols = dashboard_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Initialize session state for chart configuration
    if "custom_x_axis" not in st.session_state:
        st.session_state.custom_x_axis = all_cols[0] if all_cols else None
    if "custom_y_axis" not in st.session_state:
        st.session_state.custom_y_axis = numeric_cols[0] if numeric_cols else (all_cols[1] if len(all_cols) > 1 else None)
    if "custom_chart_type" not in st.session_state:
        st.session_state.custom_chart_type = "Scatter"
    
    # Column selectors
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_axis = st.selectbox(
            "📊 X-axis Column", 
            all_cols,
            index=all_cols.index(st.session_state.custom_x_axis) if st.session_state.custom_x_axis in all_cols else 0
        )
        st.session_state.custom_x_axis = x_axis
    
    with col2:
        y_axis = st.selectbox(
            "📈 Y-axis Column",
            all_cols,
            index=all_cols.index(st.session_state.custom_y_axis) if st.session_state.custom_y_axis in all_cols else (1 if len(all_cols) > 1 else 0)
        )
        st.session_state.custom_y_axis = y_axis
    
    with col3:
        chart_type = st.selectbox(
            "📉 Chart Type",
            ["Scatter", "Line", "Bar", "Box", "Histogram", "Area", "Density", "Violin", 
             "Strip", "Funnel", "Pie", "Sunburst", "Treemap"],
            index=["Scatter", "Line", "Bar", "Box", "Histogram", "Area", "Density", "Violin", 
                   "Strip", "Funnel", "Pie", "Sunburst", "Treemap"].index(st.session_state.custom_chart_type) if st.session_state.custom_chart_type in ["Scatter", "Line", "Bar", "Box", "Histogram", "Area", "Density", "Violin", "Strip", "Funnel", "Pie", "Sunburst", "Treemap"] else 0
        )
        st.session_state.custom_chart_type = chart_type
    
    with col4:
        generate_btn = st.button("🔄 Generate Dashboard", use_container_width=True)
    
    # Generate chart
    if x_axis and y_axis:
        try:
            # Prepare data
            plot_data = dashboard_data[[x_axis, y_axis]].dropna()
            
            if len(plot_data) == 0:
                st.warning("⚠️ No valid data available for selected columns.")
            else:
                # Create chart based on type
                if chart_type == "Scatter":
                    fig = px.scatter(
                        plot_data, 
                        x=x_axis, 
                        y=y_axis,
                        title=f"{y_axis} vs {x_axis}",
                        labels={x_axis: x_axis, y_axis: y_axis},
                        hover_data={x_axis: True, y_axis: True}
                    )
                    
                elif chart_type == "Line":
                    # Sort by x-axis if numeric
                    if pd.api.types.is_numeric_dtype(plot_data[x_axis]):
                        plot_data = plot_data.sort_values(x_axis)
                    
                    fig = px.line(
                        plot_data, 
                        x=x_axis, 
                        y=y_axis,
                        title=f"{y_axis} vs {x_axis}",
                        markers=True
                    )
                    
                elif chart_type == "Bar":
                    # Aggregate data for bar chart
                    if plot_data[x_axis].dtype == 'object':
                        agg_data = plot_data.groupby(x_axis)[y_axis].agg(['mean', 'count']).reset_index()
                        fig = px.bar(
                            agg_data,
                            x=x_axis,
                            y='mean',
                            title=f"Average {y_axis} by {x_axis}",
                            labels={'mean': y_axis}
                        )
                    else:
                        fig = px.bar(
                            plot_data,
                            x=x_axis,
                            y=y_axis,
                            title=f"{y_axis} vs {x_axis}"
                        )
                    
                elif chart_type == "Box":
                    fig = px.box(
                        plot_data,
                        x=x_axis if plot_data[x_axis].dtype == 'object' else None,
                        y=y_axis,
                        title=f"Distribution of {y_axis} by {x_axis}"
                    )
                    
                elif chart_type == "Histogram":
                    fig = px.histogram(
                        plot_data,
                        x=y_axis,
                        nbins=30,
                        title=f"Distribution of {y_axis}"
                    )
                
                elif chart_type == "Area":
                    if pd.api.types.is_numeric_dtype(plot_data[x_axis]):
                        plot_data = plot_data.sort_values(x_axis)
                    fig = px.area(
                        plot_data,
                        x=x_axis,
                        y=y_axis,
                        title=f"Area Chart: {y_axis} vs {x_axis}"
                    )
                
                elif chart_type == "Density":
                    fig = px.density_contour(
                        plot_data,
                        x=x_axis if pd.api.types.is_numeric_dtype(plot_data[x_axis]) else y_axis,
                        y=y_axis if pd.api.types.is_numeric_dtype(plot_data[y_axis]) else x_axis,
                        title=f"Density Plot: {y_axis} vs {x_axis}",
                        marginal_x="histogram",
                        marginal_y="histogram"
                    )
                
                elif chart_type == "Violin":
                    fig = px.violin(
                        plot_data,
                        x=x_axis if plot_data[x_axis].dtype == 'object' else y_axis,
                        y=y_axis if plot_data[y_axis].dtype != 'object' else x_axis,
                        title=f"Violin Plot: {y_axis} by {x_axis}",
                        box=True,
                        points="outliers"
                    )
                
                elif chart_type == "Strip":
                    fig = px.strip(
                        plot_data,
                        x=x_axis if plot_data[x_axis].dtype == 'object' else y_axis,
                        y=y_axis if plot_data[y_axis].dtype != 'object' else x_axis,
                        title=f"Strip Plot: {y_axis} by {x_axis}",
                        jitter=True
                    )
                
                elif chart_type == "Funnel":
                    agg_data = plot_data.groupby(x_axis)[y_axis].sum().reset_index().sort_values(y_axis, ascending=False)
                    fig = px.funnel(
                        agg_data,
                        x=y_axis,
                        y=x_axis,
                        title=f"Funnel Chart: {y_axis} by {x_axis}"
                    )
                
                elif chart_type == "Pie":
                    agg_data = plot_data.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.pie(
                        agg_data,
                        values=y_axis,
                        names=x_axis,
                        title=f"Pie Chart: {y_axis} by {x_axis}"
                    )
                
                elif chart_type == "Sunburst":
                    agg_data = plot_data.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.sunburst(
                        agg_data,
                        names=x_axis,
                        values=y_axis,
                        title=f"Sunburst Chart: {y_axis} by {x_axis}"
                    )
                
                elif chart_type == "Treemap":
                    agg_data = plot_data.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.treemap(
                        agg_data,
                        labels=x_axis,
                        values=y_axis,
                        title=f"Treemap: {y_axis} by {x_axis}"
                    )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{x_axis}_{y_axis}_{chart_type}")
                
                # Display statistics
                st.divider()
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                with stat_col1:
                    if pd.api.types.is_numeric_dtype(plot_data[y_axis]):
                        st.metric("Mean", f"{plot_data[y_axis].mean():.2f}")
                
                with stat_col2:
                    if pd.api.types.is_numeric_dtype(plot_data[y_axis]):
                        st.metric("Median", f"{plot_data[y_axis].median():.2f}")
                
                with stat_col3:
                    if pd.api.types.is_numeric_dtype(plot_data[y_axis]):
                        st.metric("Std Dev", f"{plot_data[y_axis].std():.2f}")
                
                # Additional info
                with st.expander("📋 Data Summary"):
                    st.write(f"**Total rows displayed:** {len(plot_data)}")
                    st.write(f"**X-axis type:** {plot_data[x_axis].dtype}")
                    st.write(f"**Y-axis type:** {plot_data[y_axis].dtype}")
                    st.dataframe(plot_data.head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error creating chart: {str(e)}")
            with st.expander("Debug Info"):
                st.write(f"Error details: {str(e)}")


with tab_data:
    st.subheader("Cleaned Data")
    st.dataframe(dashboard_data, use_container_width=True)

    csv_output = dashboard_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Full Cleaned Dataset",
        data=csv_output,
        file_name="hotel_dashboard_dataset.csv",
        mime="text/csv",
    )

    # Export static HTML report
    if st.button("Export static HTML report"):
        try:
            subprocess.run(["python", "export_static.py"], check=True)
            st.success("Static report generated: assets/hotel_report.html")
            with open("assets/hotel_report.html", "r", encoding="utf-8") as f:
                html = f.read()
            st.markdown("[Open static report](assets/hotel_report.html)", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to generate report: {e}")


st.caption("Built from your original analysis flow, adapted into an interactive Streamlit dashboard.")

# Render footer HTML if present
footer_path = Path("assets/footer.html")
if footer_path.exists():
    try:
        with open(footer_path, "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    except Exception:
        pass