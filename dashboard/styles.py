
# HospitalFlow AI
# Dashboard Styling

import streamlit as st


def apply_styles():

    st.markdown(
        """
        <style>

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid #26364D;
        }

        [data-testid="stMetric"] {
            background-color: #152236;
            border: 1px solid #26364D;
            padding: 18px;
            border-radius: 12px;
        }

        [data-testid="stMetricValue"] {
            color: #F1F5F9;
            font-weight: 700;
        }

        [data-testid="stMetricLabel"] {
            color: #94A3B8;
        }

        .hospital-header {
            background: linear-gradient(
                120deg,
                #102B46,
                #152236
            );
            border: 1px solid #26364D;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
        }

        .hospital-header h1 {
            color: #F1F5F9;
            margin: 0;
            font-size: 32px;
        }

        .hospital-header p {
            color: #94A3B8;
            margin-top: 8px;
        }

        .agent-card {
            background-color: #152236;
            border: 1px solid #26364D;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
        }

        .agent-card h4 {
            color: #E2E8F0;
            margin-bottom: 6px;
        }

        .agent-card p {
            color: #94A3B8;
            font-size: 14px;
        }

        div.stButton > button {
            border-radius: 8px;
            font-weight: 600;
            min-height: 42px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def render_header():

    st.markdown(
        """
        <div class="hospital-header">

            <h1>HospitalFlow AI</h1>

            <p>
                Multi-Agent Hospital Patient Flow
                and Bed Management System
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )
