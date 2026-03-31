from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("artifacts/research_run/best_model/best_pipeline.joblib")
PROFILE_PATH = Path("artifacts/research_run/best_model/data_profile.json")
METADATA_PATH = Path("artifacts/research_run/best_model/best_model_metadata.json")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def svg_data_uri(svg: str) -> str:
    return f"data:image/svg+xml;utf8,{quote(svg)}"


def hero_illustration() -> str:
    return svg_data_uri(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 460">
          <defs>
            <linearGradient id="sky" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stop-color="#dff3ff"/>
              <stop offset="100%" stop-color="#f7fbef"/>
            </linearGradient>
            <linearGradient id="hill" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0%" stop-color="#6ba65f"/>
              <stop offset="100%" stop-color="#2d6a3e"/>
            </linearGradient>
            <linearGradient id="field1" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stop-color="#98c76d"/>
              <stop offset="100%" stop-color="#5c913e"/>
            </linearGradient>
            <linearGradient id="field2" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stop-color="#e6c267"/>
              <stop offset="100%" stop-color="#c99a2e"/>
            </linearGradient>
          </defs>
          <rect width="800" height="460" fill="url(#sky)"/>
          <circle cx="644" cy="86" r="42" fill="#ffd65a"/>
          <path d="M0 250 C120 190, 240 190, 360 252 S620 310, 800 220 L800 460 L0 460 Z" fill="url(#hill)"/>
          <path d="M0 300 C140 250, 310 270, 430 320 S670 390, 800 340 L800 460 L0 460 Z" fill="#4f7b36"/>
          <path d="M0 350 L800 260 L800 460 L0 460 Z" fill="url(#field1)"/>
          <path d="M0 405 L800 315 L800 460 L0 460 Z" fill="url(#field2)" opacity="0.95"/>
          <path d="M70 300 L740 228" stroke="#ffffff" stroke-opacity="0.55" stroke-width="4"/>
          <path d="M85 340 L756 270" stroke="#ffffff" stroke-opacity="0.4" stroke-width="4"/>
          <rect x="120" y="180" width="92" height="82" rx="10" fill="#fff6de"/>
          <polygon points="105,190 166,142 228,190" fill="#b45a36"/>
          <rect x="156" y="222" width="20" height="40" rx="4" fill="#78533d"/>
          <rect x="412" y="140" width="12" height="118" fill="#7b593f"/>
          <path d="M418 150 C360 188, 360 238, 418 248 C476 238, 476 188, 418 150 Z" fill="#477b48"/>
          <path d="M545 172 C515 155, 492 170, 494 198 C496 226, 528 237, 550 222 C572 237, 604 226, 606 198 C608 170, 585 155, 555 172 Z" fill="#4b884d"/>
        </svg>
        """
    )


def crop_card_image(label: str, accent: str, icon: str) -> str:
    return svg_data_uri(
        f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 220">
          <defs>
            <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0%" stop-color="{accent}"/>
              <stop offset="100%" stop-color="#f8f0c7"/>
            </linearGradient>
          </defs>
          <rect width="320" height="220" rx="24" fill="url(#bg)"/>
          <circle cx="257" cy="60" r="28" fill="#fff4b2" opacity="0.9"/>
          <path d="M0 170 C60 130, 125 130, 170 158 S260 196, 320 166 L320 220 L0 220 Z" fill="#4e7b36" opacity="0.88"/>
          <path d="M0 184 C72 152, 134 160, 197 184 S275 206, 320 190 L320 220 L0 220 Z" fill="#75a94b"/>
          <text x="28" y="76" font-size="52">{icon}</text>
          <text x="28" y="118" font-size="28" font-family="Verdana" fill="#183123">{label}</text>
        </svg>
        """
    )


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(234, 208, 122, 0.22), transparent 24%),
                radial-gradient(circle at 95% 4%, rgba(113, 173, 120, 0.22), transparent 20%),
                linear-gradient(180deg, #f8f4e5 0%, #eef5e6 48%, #f7fbf7 100%);
        }
        .block-container {
            max-width: 1200px;
            padding-top: 1.1rem;
            padding-bottom: 2.2rem;
        }
        h1, h2, h3 {
            color: #173424;
            letter-spacing: -0.02em;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            color: #264734;
            font-size: 0.95rem;
        }
        .brand {
            font-weight: 700;
            font-size: 1.1rem;
        }
        .hero-wrap {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 1.2rem;
            align-items: stretch;
            margin-bottom: 1.2rem;
        }
        .hero-copy {
            padding: 1.6rem;
            border-radius: 30px;
            background: linear-gradient(135deg, #173f2d 0%, #2e6c4d 55%, #8fbb74 100%);
            color: #f5fff8;
            box-shadow: 0 22px 60px rgba(23, 63, 45, 0.18);
        }
        .hero-copy h1 {
            color: #f6fff7;
            margin-bottom: 0.45rem;
            font-size: 3rem;
            line-height: 1.02;
        }
        .hero-copy p {
            color: rgba(246,255,247,0.92);
            line-height: 1.65;
            font-size: 1rem;
            max-width: 700px;
        }
        .hero-badges {
            display: flex;
            gap: 0.7rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .hero-badge {
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 999px;
            padding: 0.48rem 0.85rem;
            font-size: 0.9rem;
        }
        .hero-art {
            min-height: 100%;
            border-radius: 30px;
            overflow: hidden;
            box-shadow: 0 18px 48px rgba(38, 71, 52, 0.14);
            background: rgba(255,255,255,0.8);
            border: 1px solid rgba(23,63,45,0.08);
        }
        .hero-art img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.9rem;
            margin-bottom: 1rem;
        }
        .stat-card {
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(28, 61, 41, 0.08);
            border-radius: 22px;
            padding: 1rem;
            box-shadow: 0 12px 36px rgba(46, 73, 34, 0.08);
        }
        .stat-label {
            color: #6c7242;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .stat-value {
            color: #193022;
            font-size: 1.6rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .website-section {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(34, 67, 43, 0.08);
            border-radius: 28px;
            padding: 1.2rem;
            box-shadow: 0 18px 46px rgba(44, 66, 32, 0.08);
            margin-bottom: 1rem;
            backdrop-filter: blur(6px);
        }
        .section-note {
            color: #58705f;
            font-size: 0.96rem;
            line-height: 1.55;
            margin-top: -0.2rem;
            margin-bottom: 1rem;
        }
        .crop-gallery {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.9rem;
            margin-top: 0.5rem;
        }
        .crop-gallery img {
            width: 100%;
            border-radius: 24px;
            display: block;
            border: 1px solid rgba(23,63,45,0.08);
            box-shadow: 0 10px 30px rgba(25, 53, 31, 0.08);
        }
        .result-hero {
            background: linear-gradient(135deg, #fff2c1 0%, #ffe38f 100%);
            border: 1px solid #e2bf49;
            border-radius: 26px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 18px 34px rgba(173, 135, 10, 0.12);
        }
        .result-overline {
            color: #8c6b00;
            text-transform: uppercase;
            font-size: 0.82rem;
            letter-spacing: 0.08em;
        }
        .result-title {
            color: #3d3203;
            font-size: 2.2rem;
            font-weight: 800;
            margin-top: 0.22rem;
            margin-bottom: 0.2rem;
        }
        .result-copy {
            color: #5d4b06;
            line-height: 1.55;
            margin: 0;
        }
        div[data-testid="stForm"] {
            background: transparent;
            border: none;
            padding: 0;
        }
        [data-testid="stNumberInput"] label, [data-testid="stTextInput"] label {
            font-weight: 600;
            color: #214131;
        }
        @media (max-width: 900px) {
            .hero-wrap, .stats-row, .crop-gallery {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_input_df(values):
    return pd.DataFrame([values])


def field_value(profile: dict, name: str, key: str, fallback: float) -> float:
    return float(profile.get(name, {}).get(key, fallback))


def number_field(label: str, key_name: str, profile: dict, step: float):
    return st.number_input(
        label,
        min_value=field_value(profile, key_name, "min", 0.0),
        max_value=field_value(profile, key_name, "max", 1000.0),
        value=field_value(profile, key_name, "median", 0.0),
        step=step,
        format="%.2f" if step < 1 else "%.0f",
    )


def render_crop_gallery():
    cards = [
        ("Rice", "#d7ebb0", "🌾"),
        ("Maize", "#f6d272", "🌽"),
        ("Cotton", "#dfe8f7", "☁"),
    ]
    images = "".join(
        f'<img src="{crop_card_image(label, color, icon)}" alt="{label}"/>' for label, color, icon in cards
    )
    st.markdown(f'<div class="crop-gallery">{images}</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="CropWise",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()

    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python run_research_pipeline.py --data <csv>` first.")
        return

    model = load_model()
    profile = load_json(PROFILE_PATH)
    metadata = load_json(METADATA_PATH)
    metrics = metadata.get("metrics", {})

    st.markdown(
        """
        <div class="topbar">
            <div class="brand">CropWise</div>
            <div>Smart crop recommendation for soil and weather conditions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-copy">
                <h1>Choose a better crop<br/>with a cleaner web experience</h1>
                <p>
                    This app uses your trained research model to recommend the most suitable crop from nutrient,
                    pH, humidity, rainfall, and temperature values. Instead of sliders, you can enter values directly
                    like a proper website form and get a more visual result with confidence scores.
                </p>
                <div class="hero-badges">
                    <div class="hero-badge">Best model: {metadata.get("best_model", "unknown")}</div>
                    <div class="hero-badge">Macro-F1: {metrics.get("f1_macro", 0.0):.4f}</div>
                    <div class="hero-badge">Top-3 accuracy: {metrics.get("top3_accuracy", 0.0):.4f}</div>
                </div>
            </div>
            <div class="hero-art">
                <img src="{hero_illustration()}" alt="Farm illustration"/>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="stats-row">
            <div class="stat-card"><div class="stat-label">Accuracy</div><div class="stat-value">{metrics.get("accuracy", 0.0):.4f}</div></div>
            <div class="stat-card"><div class="stat-label">Macro F1</div><div class="stat-value">{metrics.get("f1_macro", 0.0):.4f}</div></div>
            <div class="stat-card"><div class="stat-label">Precision</div><div class="stat-value">{metrics.get("precision_macro", 0.0):.4f}</div></div>
            <div class="stat-card"><div class="stat-label">Recall</div><div class="stat-value">{metrics.get("recall_macro", 0.0):.4f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        st.markdown('<div class="website-section">', unsafe_allow_html=True)
        st.subheader("Enter Field Conditions")
        st.markdown(
            '<div class="section-note">Type the values directly. These are the same agronomic inputs the model was trained on.</div>',
            unsafe_allow_html=True,
        )

        with st.form("crop_form"):
            soil_left, soil_right = st.columns(2, gap="large")
            with soil_left:
                N = number_field("Nitrogen (N)", "N", profile, 1.0)
                P = number_field("Phosphorus (P)", "P", profile, 1.0)
                K = number_field("Potassium (K)", "K", profile, 1.0)
                ph = number_field("Soil pH", "ph", profile, 0.01)
            with soil_right:
                temperature = number_field("Temperature (°C)", "temperature", profile, 0.1)
                humidity = number_field("Humidity (%)", "humidity", profile, 0.1)
                rainfall = number_field("Rainfall (mm)", "rainfall", profile, 0.1)
                st.text_input("Field note", value="Optional context for your own reference", disabled=True)

            submitted = st.form_submit_button("Get Crop Recommendation", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="website-section">', unsafe_allow_html=True)
        st.subheader("Popular Crop Styles")
        st.markdown(
            '<div class="section-note">A small visual section so the page feels more like a website and less like a plain dashboard.</div>',
            unsafe_allow_html=True,
        )
        render_crop_gallery()
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        input_values = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }
        input_df = build_input_df(input_values)
        prediction = model.predict(input_df)[0]

        st.markdown(
            f"""
            <div class="result-hero">
                <div class="result-overline">Recommended crop</div>
                <div class="result-title">{prediction.title()}</div>
                <p class="result-copy">
                    This recommendation is based on the nutrient profile and local climate values you entered.
                    Use the confidence table below to compare the strongest alternatives as well.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_col1, result_col2 = st.columns([0.95, 1.05], gap="large")

        with result_col1:
            st.markdown('<div class="website-section">', unsafe_allow_html=True)
            st.subheader("Submitted Values")
            summary_df = pd.DataFrame(
                {
                    "Feature": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
                    "Value": [N, P, K, temperature, humidity, ph, rainfall],
                }
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with result_col2:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_
                result_df = pd.DataFrame({"Crop": classes, "Confidence": probabilities}).sort_values(
                    "Confidence", ascending=False
                )
                result_df["Confidence %"] = (result_df["Confidence"] * 100).round(2)

                st.markdown('<div class="website-section">', unsafe_allow_html=True)
                st.subheader("Top Crop Matches")
                st.dataframe(result_df.head(5)[["Crop", "Confidence %"]], use_container_width=True, hide_index=True)
                st.bar_chart(result_df.head(5).set_index("Crop")[["Confidence"]], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("View all crop confidence scores"):
                    st.dataframe(result_df[["Crop", "Confidence %"]], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
