import streamlit as st
import pandas as pd
import numpy as np

from streamlit_chartjs.st_chart_component import st_chartjs
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


st.set_page_config(
    page_title="LightweightMMM Dashboard",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Import IBM Plex Sans font */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    
    /* Global styles based on theme */
    .main .block-container {
        font-family: 'IBM Plex Sans', sans-serif;
        color: #363654;
        background-color: #fcfcfe;
    }
    
    /* Main header using theme colors */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #5c69d4;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.025em;
    }
    
    /* Section headers using theme typography */
    .section-header {
        font-size: 1.25rem;
        font-weight: 500;
        color: #363654;
        border-bottom: 2px solid #5c69d4;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* MMM card using theme gradient and colors */
    .mmm-card {
        background: linear-gradient(135deg, #5c69d4 0%, #3f51b5 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0px 4px 12px rgba(160, 163, 180, 0.2);
    }
    
    /* Metric containers using theme background colors */
    .metric-container {
        background-color: #f9fafd;
        padding: 1rem;
        border-radius: 0.375rem;
        border-left: 4px solid #5c69d4;
        margin: 0.5rem 0;
        border: 1px solid #e9e9ec;
    }
    
    /* Tab styling to match theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f9fafd;
        border-radius: 0.5rem;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #7e7e91;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #5c69d4;
        box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f4f5fb;
        border-right: 1px solid #e9e9ec;
    }
    
    /* Button styling to match theme */
    .stButton > button {
        background-color: #5c69d4;
        color: white;
        border: none;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-family: 'IBM Plex Sans', sans-serif;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #3f51b5;
        box-shadow: 0px 2px 8px rgba(92, 105, 212, 0.3);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e9e9ec;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="metric-container"] > div {
        color: #363654;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #e9e9ec;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    /* Success/Warning/Error colors from theme */
    .success-text { color: #24A13F; }
    .warning-text { color: #E3A008; }
    .error-text { color: #FF614C; }
    .info-text { color: #1766f6; }
    
    /* Multiselect tags styling - change from red to primary color */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #5c69d4 !important;
        color: white !important;
        border: 1px solid #5c69d4 !important;
    }
    
    .stMultiSelect [data-baseweb="tag"]:hover {
        background-color: #3f51b5 !important;
        border-color: #3f51b5 !important;
    }
    
    /* Multiselect tag close button */
    .stMultiSelect [data-baseweb="tag"] [aria-label="Delete"] {
        color: white !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] [aria-label="Delete"]:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Chart.js aligned color palette from frontend theme
CHART_COLORS = {
    "primary": [
        "#b8beec",
        "#FF614C",
        "#E3A008",
        "#FF9A61",
        "#3AAA52",
    ],  # purple.300, red.500, yellow.500, orange.500, green.500
    "backgrounds": [
        "rgba(184, 190, 236, 0.6)",
        "rgba(255, 97, 76, 0.6)",
        "rgba(227, 160, 8, 0.6)",
        "rgba(255, 154, 97, 0.6)",
        "rgba(58, 170, 82, 0.6)",
    ],
    "borders": ["#b8beec", "#FF614C", "#E3A008", "#FF9A61", "#3AAA52"],
}


def get_chart_color(index, color_type="primary"):
    """Get color from theme-based palette"""
    colors = CHART_COLORS[color_type]
    return colors[index % len(colors)]


def get_background_color(index):
    """Get background color from theme-based palette"""
    colors = CHART_COLORS["backgrounds"]
    return colors[index % len(colors)]


def create_chartjs_options(
    title="", x_axis_title="", y_axis_title="", legend_position="top"
):
    return {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "title": {
                "display": bool(title),
                "text": title,
                "font": {"family": "IBM Plex Sans", "size": 16, "weight": "600"},
                "color": "#363654",
                "padding": 20,
            },
            "legend": {
                "position": legend_position,
                "labels": {
                    "font": {"family": "IBM Plex Sans", "size": 12},
                    "color": "#363654",
                    "padding": 20,
                },
            },
        },
        "scales": {
            "x": {
                "title": {
                    "display": bool(x_axis_title),
                    "text": x_axis_title,
                    "font": {"family": "IBM Plex Sans", "size": 12},
                    "color": "#363654",
                },
                "grid": {"color": "#e9e9ec", "lineWidth": 1},
                "ticks": {
                    "font": {"family": "IBM Plex Sans", "size": 11},
                    "color": "#363654",
                },
            },
            "y": {
                "title": {
                    "display": bool(y_axis_title),
                    "text": y_axis_title,
                    "font": {"family": "IBM Plex Sans", "size": 12},
                    "color": "#363654",
                },
                "grid": {"color": "#e9e9ec", "lineWidth": 1},
                "ticks": {
                    "font": {"family": "IBM Plex Sans", "size": 5},
                    "color": "#363654",
                },
            },
        },
    }


@st.cache_data
def generate_sample_mmm_data():
    """Generate sample MMM data for demonstration purposes."""
    np.random.seed(42)

    # Date range (52 weeks)
    start_date = datetime.now() - timedelta(weeks=52)
    dates = pd.date_range(start=start_date, periods=52, freq="W")

    # Media channels
    channels = [
        "TV",
        "Digital Video",
        "Search",
    ]

    # Generate base data
    data = []
    base_kpi = 10000  # Base weekly KPI

    for i, date in enumerate(dates):
        week_data = {"date": date, "week": i + 1}

        # Seasonal effects
        seasonal_multiplier = (
            1 + 0.3 * np.sin(2 * np.pi * i / 52) + 0.1 * np.sin(4 * np.pi * i / 52)
        )

        # Generate media spend and effects for each channel
        total_media_effect = 0
        total_spend = 0

        for j, channel in enumerate(channels):
            # Base spend with variation
            base_spend = [50000, 40000, 35000][j]
            spend_variation = (
                1 + 0.4 * np.sin(2 * np.pi * i / 52 + j) + 0.2 * np.random.normal()
            )
            spend = max(0, base_spend * spend_variation * seasonal_multiplier)

            # Simple saturation effect
            saturation_point = [80000, 60000, 50000][j]
            saturation_effect = 1 - np.exp(-spend / saturation_point)

            # ROI varies by channel
            base_roi = [2.1, 3.5, 4.2][j]
            media_effect = (
                spend * base_roi * saturation_effect * (1 + 0.1 * np.random.normal())
            )

            week_data[f"{channel}_spend"] = spend
            week_data[f"{channel}_effect"] = media_effect

            total_media_effect += media_effect
            total_spend += spend

        # Total KPI = base + media effects + noise
        week_data["total_kpi"] = (
            base_kpi * seasonal_multiplier
            + total_media_effect
            + 2000 * np.random.normal()
        )
        week_data["total_spend"] = total_spend
        week_data["baseline_kpi"] = base_kpi * seasonal_multiplier
        week_data["media_kpi"] = total_media_effect

        data.append(week_data)

    df = pd.DataFrame(data)

    # Calculate attribution percentages
    effect_columns = [col for col in df.columns if col.endswith("_effect")]
    total_media_effect = df[effect_columns].sum(axis=1)

    for col in effect_columns:
        channel = col.replace("_effect", "")
        df[f"{channel}_attribution"] = (df[col] / total_media_effect * 100).fillna(0)

    return df, channels


def calculate_channel_metrics(df, channels):
    """Calculate key metrics for each channel."""
    metrics = []

    for channel in channels:
        spend_col = f"{channel}_spend"
        effect_col = f"{channel}_effect"

        total_spend = df[spend_col].sum()
        total_effect = df[effect_col].sum()
        roi = total_effect / total_spend if total_spend > 0 else 0
        avg_attribution = df[f"{channel}_attribution"].mean()

        # Calculate efficiency (effect per dollar)
        efficiency = total_effect / total_spend if total_spend > 0 else 0

        # Calculate contribution to total media effect
        total_media_effect = df[[f"{ch}_effect" for ch in channels]].sum().sum()
        contribution = (
            (total_effect / total_media_effect * 100) if total_media_effect > 0 else 0
        )

        metrics.append(
            {
                "Channel": channel,
                "Total Spend ($)": total_spend,
                "Total Effect": total_effect,
                "ROI": roi,
                "Efficiency (Effect/$)": efficiency,
                "Avg Attribution (%)": avg_attribution,
                "Contribution (%)": contribution,
            }
        )

    return pd.DataFrame(metrics)


def main():
    with st.spinner("Loading MMM model results..."):
        df, channels = generate_sample_mmm_data()
        channel_metrics = calculate_channel_metrics(df, channels)

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    st.markdown(
        '<div class="section-header">📊 MMM Performance Overview</div>',
        unsafe_allow_html=True,
    )

    controls_col, metrics_col = st.columns([1, 2])

    with controls_col:
        # Date range filter
        date_range = st.date_input(
            "Analysis Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # Channel selection
        selected_channels = st.multiselect(
            "Select Channels for Analysis", channels, default=channels
        )

    # Filter data
    filtered_df = df[
        (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
    ]

    with metrics_col:
        st.markdown("### 📈 Key Metrics")

        row1_col1, row1_col2, row1_col3 = st.columns(3)

        with row1_col1:
            total_spend = filtered_df["total_spend"].sum()
            st.metric("Total Media Spend", f"${total_spend:,.0f}")

        with row1_col2:
            media_contribution = (
                filtered_df["media_kpi"].sum() / filtered_df["total_kpi"].sum() * 100
            )
            st.metric("Media Contribution", f"{media_contribution:.1f}%")

        with row1_col3:
            baseline_contribution = (
                filtered_df["baseline_kpi"].sum() / filtered_df["total_kpi"].sum() * 100
            )
            st.metric("Baseline Contribution", f"{baseline_contribution:.1f}%")

    # Tabs for different analyses
    tab1, tab2, tab4, tab5 = st.tabs(
        [
            "Attribution Analysis",
            "ROI & Performance",
            "Time Series",
            "Budget Optimization",
        ]
    )

    with tab1:
        st.markdown(
            '<div class="section-header">📈 Media Attribution Analysis</div>',
            unsafe_allow_html=True,
        )

        # Attribution pie chart
        col1, col2 = st.columns([1, 2])

        with col1:
            # Calculate total attribution
            attribution_data = []
            for channel in selected_channels:
                total_effect = filtered_df[f"{channel}_effect"].sum()
                attribution_data.append({"Channel": channel, "Effect": total_effect})

            attr_df = pd.DataFrame(attribution_data)

            if not attr_df.empty:
                # Create Chart.js pie chart data
                pie_data = {
                    "labels": attr_df["Channel"].tolist(),
                    "datasets": [
                        {
                            "label": "Attribution",
                            "data": attr_df["Effect"].tolist(),
                            "backgroundColor": [
                                get_chart_color(i) for i in range(len(attr_df))
                            ],
                            "borderColor": None,
                            "borderWidth": 0,
                        }
                    ],
                }

                st_chartjs(
                    data=pie_data,
                    chart_type="pie",
                    title="Media Attribution by Channel",
                    legend_position="bottom",
                    # canvas_height=1000,
                    # canvas_width=100,
                )

        with col2:
            st.subheader("Attribution Breakdown")
            if not attr_df.empty:
                attr_df_sorted = attr_df.sort_values("Effect", ascending=False)
                total_effect = attr_df_sorted["Effect"].sum()

                for _, row in attr_df_sorted.iterrows():
                    percentage = (row["Effect"] / total_effect) * 100
                    st.metric(row["Channel"], f"{percentage:.1f}%")

        # Attribution trends over time
        # st.subheader("Attribution Trends Over Time")

        # attribution_columns = [
        #     f"{ch}_attribution"
        #     for ch in selected_channels
        #     if f"{ch}_attribution" in filtered_df.columns
        # ]

        # if attribution_columns:
        #     datasets = []
        #     for i, col in enumerate(attribution_columns):
        #         channel = col.replace("_attribution", "")
        #         if channel in selected_channels:
        #             datasets.append(
        #                 {
        #                     "label": channel,
        #                     "data": filtered_df[col].tolist(),
        #                     "borderColor": get_chart_color(i),
        #                     "backgroundColor": get_background_color(i),
        #                     "borderWidth": 2,
        #                     "fill": False,
        #                     "tension": 0.3,
        #                     "pointRadius": 2,
        #                     "pointHoverRadius": 2,
        #                 }
        #             )

        #     line_data = {
        #         "labels": [date.strftime("%Y-%m-%d") for date in filtered_df["date"]],
        #         "datasets": datasets,
        #     }

        #     st_chartjs(
        #         data=line_data,
        #         chart_type="line",
        #         title="Weekly Attribution Percentage by Channel",
        #         legend_position="bottom",
        #         x_axis_title="Date",
        #         y_axis_title="Attribution (%)",
        #         canvas_height=500,
        #         # canvas_width=100,
        #     )

    with tab2:
        st.markdown(
            '<div class="section-header">💰 ROI & Performance Analysis</div>',
            unsafe_allow_html=True,
        )

        # Channel performance table
        filtered_metrics = channel_metrics[
            channel_metrics["Channel"].isin(selected_channels)
        ]

        # Format the dataframe for better display
        display_metrics = filtered_metrics.copy()
        display_metrics["Total Spend ($)"] = display_metrics["Total Spend ($)"].apply(
            lambda x: f"${x:,.0f}"
        )
        display_metrics["Total Effect"] = display_metrics["Total Effect"].apply(
            lambda x: f"{x:,.0f}"
        )
        display_metrics["ROI"] = display_metrics["ROI"].apply(lambda x: f"{x:.2f}x")
        display_metrics["Efficiency (Effect/$)"] = display_metrics[
            "Efficiency (Effect/$)"
        ].apply(lambda x: f"{x:.3f}")
        display_metrics["Avg Attribution (%)"] = display_metrics[
            "Avg Attribution (%)"
        ].apply(lambda x: f"{x:.1f}%")
        display_metrics["Contribution (%)"] = display_metrics["Contribution (%)"].apply(
            lambda x: f"{x:.1f}%"
        )

        # ROI comparison - table and chart in 2 columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Channel Performance Summary")
            st.dataframe(display_metrics, use_container_width=True)

        with col2:
            # Create custom color scale using theme colors
            roi_colors = []
            for i, roi in enumerate(filtered_metrics["ROI"]):
                if roi >= 3:
                    roi_colors.append("#3AAA52")  # success green
                elif roi >= 2:
                    roi_colors.append("#E3A008")  # warning yellow
                else:
                    roi_colors.append("#FF614C")  # error red

            # Create Chart.js bar chart data
            bar_data = {
                "labels": filtered_metrics["Channel"].tolist(),
                "datasets": [
                    {
                        "label": "ROI",
                        "data": filtered_metrics["ROI"].tolist(),
                        "backgroundColor": roi_colors,
                        "borderColor": ["white"] * len(filtered_metrics),
                        "borderWidth": 1,
                    }
                ],
            }

            st_chartjs(
                data=bar_data,
                chart_type="bar",
                title="Return on Investment by Channel",
                legend_position="top",
                x_axis_title="Channel",
                y_axis_title="ROI",
                canvas_height=350,
                canvas_width=500,
            )

    with tab4:
        st.markdown(
            '<div class="section-header">📅 Time Series Analysis</div>',
            unsafe_allow_html=True,
        )

        # KPI decomposition
        st.subheader("KPI Decomposition Over Time")

        # Create Chart.js line chart with multiple datasets
        datasets = [
            {
                "label": "Baseline KPI",
                "data": filtered_df["baseline_kpi"].tolist(),
                "borderColor": "#5c69d4",
                "backgroundColor": "rgba(92, 105, 212, 0.3)",
                "borderWidth": 2,
                "fill": True,
                "tension": 0.3,
            },
            {
                "label": "Media KPI",
                "data": filtered_df["media_kpi"].tolist(),
                "borderColor": "#FF614C",
                "backgroundColor": "rgba(255, 97, 76, 0.3)",
                "borderWidth": 2,
                "fill": True,
                "tension": 0.3,
            },
            {
                "label": "Total KPI",
                "data": filtered_df["total_kpi"].tolist(),
                "borderColor": "#363654",
                "backgroundColor": "rgba(54, 54, 84, 0.1)",
                "borderWidth": 3,
                "fill": False,
                "tension": 0.3,
                "pointRadius": 4,
                "pointHoverRadius": 6,
            },
        ]

        decomposition_data = {
            "labels": [date.strftime("%Y-%m-%d") for date in filtered_df["date"]],
            "datasets": datasets,
        }

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st_chartjs(
            data=decomposition_data,
            chart_type="line",
            title="KPI Decomposition: Baseline vs Media Contribution",
            legend_position="bottom",
            x_axis_title="Date",
            y_axis_title="KPI Value",
            canvas_height=400,
            canvas_width=900,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab5:
        if selected_channels:
            # Simulate optimization results
            current_allocation = []
            optimized_allocation = []

            for channel in selected_channels:
                current_spend = channel_metrics[channel_metrics["Channel"] == channel][
                    "Total Spend ($)"
                ].iloc[0]
                current_pct = (
                    current_spend / channel_metrics["Total Spend ($)"].sum()
                ) * 100

                # Simulate optimized allocation
                optimization_factor = np.random.uniform(0.8, 1.3)
                optimized_pct = current_pct * optimization_factor

                current_allocation.append(current_pct)
                optimized_allocation.append(optimized_pct)

            # Normalize optimized allocation
            total_optimized = sum(optimized_allocation)
            optimized_allocation = [
                x / total_optimized * 100 for x in optimized_allocation
            ]

            allocation_df = pd.DataFrame(
                {
                    "Channel": selected_channels,
                    "Current (%)": current_allocation,
                    "Optimized (%)": optimized_allocation,
                    "Change (%)": [
                        opt - cur
                        for opt, cur in zip(optimized_allocation, current_allocation)
                    ],
                }
            )

            # Create Chart.js grouped bar chart data
            bar_data = {
                "labels": allocation_df["Channel"].tolist(),
                "datasets": [
                    {
                        "label": "Current",
                        "data": allocation_df["Current (%)"].tolist(),
                        "backgroundColor": "#dee1fb",
                        "borderColor": "white",
                        "borderWidth": 1,
                    },
                    {
                        "label": "Optimized",
                        "data": allocation_df["Optimized (%)"].tolist(),
                        "backgroundColor": "#5c69d4",
                        "borderColor": "white",
                        "borderWidth": 1,
                    },
                ],
            }

            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st_chartjs(
                data=bar_data,
                chart_type="bar",
                title="Current vs Optimized Budget Allocation",
                legend_position="top",
                x_axis_title="Channel",
                y_axis_title="Budget Allocation (%)",
                canvas_height=350,
                canvas_width=550,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("---")


if __name__ == "__main__":
    main()
