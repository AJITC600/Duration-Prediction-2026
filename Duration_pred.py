# =========================
# 1. IMPORTS
# =========================
import streamlit as st
import joblib
import pandas as pd
import re

# =========================
# 2. LOAD MODEL
# =========================
model = joblib.load(r"C:/Users/USER/Desktop/Kaggle/New Data/datasets/r1.pkl")

# =========================
# 3. PAGE SETUP
# =========================
st.set_page_config(page_title="Delivery Chatbot", page_icon="🚚")
st.title("🚚 Delivery Assistant Chatbot")

# =========================
# 4. CHAT MEMORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# 5. DISPLAY CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================
# 6. FEATURE EXTRACTION
# =========================
def extract_features(text):
    text = text.lower()

    load = re.search(r"load\s*(\d+\.?\d*)", text)
    dashers = re.search(r"(\d+)\s*dasher", text)
    items = re.search(r"(\d+)\s*item", text)
    prep = re.search(r"prep\s*(\d+)", text)

    return (
        float(load.group(1)) if load else None,
        float(dashers.group(1)) if dashers else None,
        float(items.group(1)) if items else None,
        float(prep.group(1)) if prep else None
    )

# =========================
# 7. VALIDATION
# =========================
def validate(order_load, dashers, items, prep):

    if None in [order_load, dashers, items, prep]:
        return False, "❗ Try: load 0.7, 5 dashers, 10 items, prep 15"

    if not (0 <= order_load <= 1):
        return False, "⚠️ Order load must be between 0 and 1"

    if dashers <= 0 or items <= 0:
        return False, "⚠️ Values must be positive"

    if not (1 <= prep <= 60):
        return False, "⚠️ Prep time must be 1–60 minutes"

    return True, ""

# =========================
# 8. CHAT INPUT
# =========================
user_input = st.chat_input("Type your message...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    user_lower = user_input.lower()

    if "hi" in user_lower or "hello" in user_lower:
        bot_reply = "Hello! I can predict delivery time."

    elif "help" in user_lower:
        bot_reply = "Example: load 0.7, 5 dashers, 10 items, prep 15"

    elif any(x in user_lower for x in ["load", "dasher", "item", "prep"]):

        order_load, dashers, items, prep = extract_features(user_input)
        valid, msg = validate(order_load, dashers, items, prep)

        if not valid:
            bot_reply = msg

        else:
            # =========================
            # ✅ FEATURES (MINUTES)
            # =========================
            prep_time = prep
            load_per_dasher = order_load / (dashers + 1)

            features = pd.DataFrame([[
                items,
                dashers,
                order_load,
                prep_time,
                load_per_dasher
            ]], columns=[
                'total_items',
                'total_busy_dashers',
                'order_load',
                'prep_time_estimate',
                'load_per_dasher'
            ])

            # =========================
            # ✅ MODEL PREDICTION
            # =========================
            pred = model.predict(features)[0]

            # =========================
            # 🔥 ADAPTIVE CORRECTION
            # =========================
            ratio = pred / (prep_time + 1)

            if ratio > 2:
                final_pred = 0.5 * pred + 0.5 * prep_time
            elif ratio < 0.8:
                final_pred = 0.8 * pred + 0.2 * prep_time
            else:
                final_pred = pred

            # =========================
            # 🔥 EDGE CASE HANDLING
            # =========================

            # EASY CASE
            if items <= 3 and dashers >= 10 and order_load < 0.4:
                final_pred = min(final_pred, prep_time * 1.3)

            # HEAVY CASE
            if items >= 10 and dashers <= 4:
                final_pred = max(final_pred, prep_time * 1.5)

            # ✅ BALANCED CASE (NEW FIX)
            if 0.3 <= order_load <= 0.7 and 4 <= dashers <= 8 and 4 <= items <= 8:
                final_pred *= 0.85

            # =========================
            # ✅ OUTPUT
            # =========================
            bot_reply = f"🚀 Estimated Delivery Time: {round(final_pred, 1)} minutes"

    elif "late" in user_lower:
        bot_reply = "High load or fewer dashers may cause delays."

    else:
        bot_reply = "I didn’t understand. Type 'help'."

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant"):
        st.write(bot_reply)