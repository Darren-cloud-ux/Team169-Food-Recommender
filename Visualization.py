import os
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Page setup (wider layout)
# -----------------------------
st.set_page_config(page_title="🍽️ Food Recommendation System", layout="wide")
st.title("🍽️ Food Recommendation System by Nutritional Density")

# Backward-compatible cache (works on older Streamlit too)
try:
    cache_data = st.cache_data
except AttributeError:
    cache_data = st.cache

# -----------------------------
# Data paths (absolute)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "food_nutrition_summary_brand_names.csv")
DATA_PARQUET = os.path.join(BASE_DIR, "food_nutrition_summary_brand_names.parquet")

# -----------------------------
# Data loading (LOCAL FILES)
# -----------------------------
@cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """
    Load data, preferring Parquet if available, else CSV with optimizations.
    This is cached so it won't reload on every widget interaction.
    """
    # 1) Prefer Parquet for speed
    if os.path.exists(DATA_PARQUET):
        df = pd.read_parquet(DATA_PARQUET)
    else:
        # 2) Fall back to CSV
        if not os.path.exists(DATA_CSV):
            raise FileNotFoundError(
                f"Dataset not found. Expected either:\n- {DATA_PARQUET}\n- {DATA_CSV}"
            )

        # Only hint text columns; let numeric types infer
        dtype_hints = {
            "brand_name": "string",
            "description": "string",
            "nutriscore": "string",
        }

        # Try faster pyarrow engine first
        try:
            df = pd.read_csv(DATA_CSV, dtype=dtype_hints, engine="pyarrow")
        except Exception:
            df = pd.read_csv(DATA_CSV, dtype=dtype_hints, low_memory=False)

    # ---- Memory optimizations ----
    # Downcast numeric columns to float32 / int32 to reduce RAM usage
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].astype("float32")

    # Convert category column to pandas 'category' dtype (huge memory saver)
    if "branded_food_category" in df.columns:
        df["branded_food_category"] = df["branded_food_category"].astype("category")

    return df


# -----------------------------
# Load dataset with clear error display
# -----------------------------
try:
    df = load_data()
    st.caption(f"✅ Loaded dataset with {len(df):,} rows and {len(df.columns)} columns.")
except Exception as e:
    st.error("❌ Error loading dataset.")
    st.exception(e)  # Show full traceback in the app
    st.stop()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filter Options")

# Standardize other text columns (if present)
for col in ("brand_name", "description"):
    if col in df.columns:
        df[col] = df[col].astype("string")

# Ensure category column exists
if "branded_food_category" not in df.columns:
    st.error("Column 'branded_food_category' is missing from the dataset.")
    st.stop()

# Use the 'category' dtype to get unique categories cheaply
cat_series = df["branded_food_category"]
if not pd.api.types.is_categorical_dtype(cat_series):
    cat_series = cat_series.astype("category")
    df["branded_food_category"] = cat_series

# Get all distinct categories from the categorical metadata
categories = sorted(list(cat_series.cat.categories))

if not categories:
    st.error("No valid categories found in 'branded_food_category'.")
    st.stop()

# Default category = "Alcohol" (fallback to first category if not present)
DEFAULT_CATEGORY = "Alcohol"
if DEFAULT_CATEGORY in categories:
    default_index = categories.index(DEFAULT_CATEGORY)
else:
    default_index = 0  # fallback

selected_category = st.sidebar.selectbox(
    "Select Food Category",
    options=categories,
    index=default_index
)

# Filter dataframe to the selected category only
df_filtered = df[df["branded_food_category"] == selected_category].copy()

# Deduplicate by brand_name within this category (keep first)
if "brand_name" in df_filtered.columns:
    df_filtered = df_filtered.dropna(subset=["brand_name"])
    df_filtered = (
        df_filtered.sort_values("brand_name")
        .drop_duplicates(subset=["brand_name"], keep="first")
    )

# -----------------------------
# Axis Selectors for scatter
# -----------------------------
numeric_cols = df_filtered.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.warning("No numeric columns found in the filtered data.")
    st.stop()

x_axis = st.sidebar.selectbox(
    "Select 1st Nutrient for Comparison",
    numeric_cols,
    index=numeric_cols.index("protein") if "protein" in numeric_cols else 0
)

y_axis_default_index = 0
if "saturated_fat" in numeric_cols:
    y_axis_default_index = numeric_cols.index("saturated_fat")
elif len(numeric_cols) > 1:
    y_axis_default_index = 1

y_axis = st.sidebar.selectbox(
    "Select 2nd Nutrient for Comparison",
    numeric_cols,
    index=y_axis_default_index
)

# -----------------------------
# Prepare Data for Scatter Plot
# -----------------------------
plot_df = df_filtered.dropna(subset=[x_axis, y_axis]).copy()

# Ensure hover fields are strings – include brand_name now
hover_fields = [
    c for c in ["brand_name", "description", "branded_food_category", "nutriscore"]
    if c in plot_df.columns
]
for c in hover_fields:
    plot_df[c] = plot_df[c].astype("string").fillna("N/A")

# Color by brand_name instead of description
color_col = "brand_name" if "brand_name" in plot_df.columns else None

# -----------------------------
# Scatter Plot
# -----------------------------
st.subheader("📊 Comparison of Nutrient Density (per Brand)")

if plot_df.empty:
    st.warning("No data available for the selected filters and nutrients.")
else:
    fig = px.scatter(
        plot_df,
        x=x_axis,
        y=y_axis,
        color=color_col,
        hover_data=hover_fields,
        title=f"{selected_category}: {x_axis} vs {y_axis} (one point per brand)",
    )

    fig.update_layout(
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Brand-level Nutrient Bar Chart
# -----------------------------
st.subheader("🍽️ Nutrient Profile for Selected Brand")

if "brand_name" not in df_filtered.columns or df_filtered.empty:
    st.info("No brand data available for this category.")
else:
    # Brand selector is now in the MAIN AREA, just above the chart
    brand_options = sorted(df_filtered["brand_name"].dropna().unique().tolist())
    selected_brand = st.selectbox(
        "Select Brand for Nutrient Profile",
        options=brand_options,
        key="brand_bar_select",
    )

    brand_row = df_filtered[df_filtered["brand_name"] == selected_brand]
    if brand_row.empty:
        st.warning("No data found for the selected brand.")
    else:
        # take first row (we already deduped, so should be exactly one)
        brand_row = brand_row.iloc[0]

        nutrient_cols = [
            "protein", "fat_total", "saturated_fat", "carbohydrates", "sugars",
            "fiber", "sodium", "potassium", "calcium", "iron",
            "vitamin_c", "vitamin_d", "cholesterol",
            "energy_kcal", "energy_kj"
        ]
        nutrient_cols = [c for c in nutrient_cols if c in df_filtered.columns]

        nutrient_values = []
        for c in nutrient_cols:
            try:
                nutrient_values.append(float(brand_row[c]))
            except Exception:
                nutrient_values.append(None)

        bar_df = pd.DataFrame(
            {"Nutrient": nutrient_cols, "Value": nutrient_values}
        ).dropna(subset=["Value"])

        if bar_df.empty:
            st.info("No numeric nutrient values available for this brand.")
        else:
            fig_bar = px.bar(
                bar_df,
                x="Nutrient",
                y="Value",
                title=f"Nutrient Profile: {selected_brand}",
            )
            fig_bar.update_layout(
                xaxis_tickangle=-45,
                height=500,
                margin=dict(l=40, r=40, t=60, b=80),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Nutri-Score Table Summary (by brand_name)
# -----------------------------
st.subheader("🥦 Brands Grouped by Nutri-Score")

if "nutriscore" not in df_filtered.columns:
    st.warning("Column 'nutriscore' not found in the filtered dataset.")
else:
    df_filtered["nutriscore"] = (
        df_filtered["nutriscore"].astype(str).str.upper().str.strip()
    )
    nutri_order = ["A", "B", "C", "D", "E"]

    # Sort Nutri-Scores with A–E first, others after
    unique_scores = sorted(
        df_filtered["nutriscore"].dropna().unique().tolist(),
        key=lambda x: nutri_order.index(x) if x in nutri_order else 999,
    )

    if not unique_scores:
        st.info("No Nutri-Score values available for this category.")
    else:
        for score in unique_scores:
            st.markdown(f"### Nutri-Score {score}")

            # Show brand names grouped under this score
            subset = (
                df_filtered[df_filtered["nutriscore"] == score][["brand_name"]]
                .dropna()
                .drop_duplicates()
                .reset_index(drop=True)
            )

            if subset.empty:
                st.write("No items found.")
            else:
                # Full width table so brand names do NOT get squeezed
                st.dataframe(
                    subset,
                    height=min(300, 40 * len(subset)),  # auto-adjust height
                    use_container_width=True,
                )

            st.markdown("---")  # separator between groups
