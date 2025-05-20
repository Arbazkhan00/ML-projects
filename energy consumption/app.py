# # # # from flask import Flask, render_template, request
# # # # import numpy as np
# # # # import joblib
# # # # import matplotlib.pyplot as plt
# # # #
# # # # app = Flask(__name__)
# # # # model = joblib.load('model.pkl')
# # # # scaler = joblib.load('scaler.pkl')
# # # #
# # # # @app.route('/', methods=['GET', 'POST'])
# # # # def index():
# # # #     prediction = None
# # # #     if request.method == 'POST':
# # # #         try:
# # # #             features = [float(request.form[f]) for f in ['machine', 'hvac', 'lighting', 'other']]
# # # #             features_scaled = scaler.transform([features])
# # # #             prediction = model.predict(features_scaled)[0]
# # # #         except:
# # # #             prediction = 'Error in input!'
# # # #     return render_template('index.html', prediction=prediction)
# # # #
# # # # if __name__ == '__main__':
# # # #     app.run(debug=True)
# # # #
# # #
# # # import streamlit as st
# # # import numpy as np
# # # import joblib
# # #
# # # # Page settings
# # # st.set_page_config(page_title="Energy Inefficiency Detector", layout="centered")
# # #
# # # # Load model and scaler
# # # @st.cache_resource
# # # def load_model():
# # #     model = joblib.load("model.pkl")
# # #     scaler = joblib.load("scaler.pkl")
# # #     return model, scaler
# # #
# # # model, scaler = load_model()
# # #
# # # # App title
# # # st.title("⚡ Energy Inefficiency Detection App")
# # # st.markdown("Enter energy consumption values to detect inefficiency.")
# # #
# # # # Input form
# # # with st.form("input_form"):
# # #     machine = st.number_input("🔧 Machine A (kWh)", min_value=0.0, value=50.0, step=0.1)
# # #     hvac = st.number_input("🌬 HVAC (kWh)", min_value=0.0, value=30.0, step=0.1)
# # #     lighting = st.number_input("💡 Lighting (kWh)", min_value=0.0, value=20.0, step=0.1)
# # #     other = st.number_input("🔌 Other (kWh)", min_value=0.0, value=10.0, step=0.1)
# # #     submitted = st.form_submit_button("🚀 Predict")
# # #
# # # # Prediction
# # # if submitted:
# # #     features = np.array([[machine, hvac, lighting, other]])
# # #     features_scaled = scaler.transform(features)
# # #
# # #     prediction = model.predict(features_scaled)[0]
# # #
# # #     if hasattr(model, "predict_proba"):
# # #         confidence = model.predict_proba(features_scaled)[0]
# # #     else:
# # #         confidence = [1 - prediction, prediction]  # fallback if no probability
# # #
# # #     if prediction == 1:
# # #         st.error("⚠ Inefficient Energy Usage Detected!")
# # #     else:
# # #         st.success("✅ Efficient Energy Usage")
# # #
# # #     st.markdown(f"*Confidence:* {confidence[1]*100:.2f}% Inefficient, {confidence[0]*100:.2f}% Efficient")
# # #     st.progress(float(confidence[1]) if prediction == 1 else float(confidence[0]))
# # #
# # # import streamlit as st
# # # import numpy as np
# # # import joblib
# # #
# # # # Load model and scaler
# # # @st.cache_resource
# # # def load_model():
# # #     model = joblib.load("model.pkl")
# # #     scaler = joblib.load("scaler.pkl")
# # #     return model, scaler
# # #
# # # model, scaler = load_model()
# # #
# # # st.title("⚡ Energy Inefficiency Detection")
# # #
# # # # Input fields
# # # with st.form("input_form"):
# # #     machine = st.number_input("Machine A (kWh)", value=50.0)
# # #     hvac = st.number_input("HVAC (kWh)", value=30.0)
# # #     lighting = st.number_input("Lighting (kWh)", value=20.0)
# # #     other = st.number_input("Other (kWh)", value=10.0)
# # #     submitted = st.form_submit_button("Predict")
# # #
# # # if submitted:
# # #     features = np.array([[machine, hvac, lighting, other]])
# # #     features_scaled = scaler.transform(features)
# # #
# # #     st.write("Feature shape:", features_scaled.shape)  # DEBUG LINE
# # #
# # #     try:
# # #         prediction = model.predict(features_scaled)[0]
# # #
# # #         if hasattr(model, "predict_proba"):
# # #             confidence = model.predict_proba(features_scaled)[0]
# # #         else:
# # #             confidence = [1 - prediction, prediction]
# # #
# # #         if prediction == 1:
# # #             st.error("⚠ Inefficient Energy Usage Detected!")
# # #         else:
# # #             st.success("✅ Efficient Energy Usage")
# # #
# # #         st.markdown(f"*Confidence:* {confidence[1]*100:.2f}% Inefficient, {confidence[0]*100:.2f}% Efficient")
# # #         st.progress(float(confidence[1]) if prediction == 1 else float(confidence[0]))
# # #
# # #     except ValueError as e:
# # #         st.error(f"Feature mismatch: {e}")
# #
# #
# # import streamlit as st
# # import numpy as np
# # import joblib
# #
# # # Load model and scaler
# # @st.cache_resource
# # def load_model():
# #     model = joblib.load("model.pkl")
# #     scaler = joblib.load("scaler.pkl")
# #     return model, scaler
# #
# # model, scaler = load_model()
# #
# # st.title("⚡ Energy Inefficiency Detection")
# #
# # # Input fields
# # with st.form("input_form"):
# #     machine = st.number_input("Machine A (kWh)", value=50.0)
# #     hvac = st.number_input("HVAC (kWh)", value=30.0)
# #     lighting = st.number_input("Lighting (kWh)", value=20.0)
# #     other = st.number_input("Other (kWh)", value=10.0)
# #     submitted = st.form_submit_button("Predict")
# #
# # if submitted:
# #     features = np.array([[machine, hvac, lighting, other]])
# #     features_scaled = scaler.transform(features)
# #
# #     try:
# #         prediction = model.predict(features_scaled)[0]
# #
# #         # Display result
# #         if prediction == 1:
# #             st.error("⚠ Inefficient Energy Usage Detected!")
# #         else:
# #             st.success("✅ Efficient Energy Usage")
# #
# #         # Confidence scores
# #         if hasattr(model, "predict_proba"):
# #             confidence = model.predict_proba(features_scaled)[0]
# #         else:
# #             confidence = [1 - prediction, prediction]
# #
# #         st.markdown(f"Confidence: {confidence[1]*100:.2f}% Inefficient, {confidence[0]*100:.2f}% Efficient")
# #         st.progress(float(confidence[1]) if prediction == 1 else float(confidence[0]))
# #
# #         # 🧾 Prediction Summary Table
# #         st.markdown("### 🧾 Prediction Summary")
# #         st.table({
# #             "Machine A (kWh)": machine,
# #             "HVAC (kWh)": hvac,
# #             "Lighting (kWh)": lighting,
# #             "Other (kWh)": other,
# #             "Detected Inefficiency": "Yes" if prediction == 1 else "No"
# #         })
# #
# #     except ValueError as e:
# #         st.error(f"Feature mismatch: {e}")
#
# import streamlit as st
# import numpy as np
# import joblib
#
# # Load model and scaler
# @st.cache_resource
# def load_model():
#     model = joblib.load("model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     return model, scaler
#
# model, scaler = load_model()
#
# st.title("⚡ Energy Inefficiency Detection")
#
# # Electricity cost per kWh (you can customize this)
# COST_PER_KWH = 0.12
#
# # Input fields
# with st.form("input_form"):
#     machine = st.number_input("Machine A (kWh)", value=50.0)
#     hvac = st.number_input("HVAC (kWh)", value=30.0)
#     lighting = st.number_input("Lighting (kWh)", value=20.0)
#     other = st.number_input("Other (kWh)", value=10.0)
#     submitted = st.form_submit_button("Predict")
#
# if submitted:
#     features = np.array([[machine, hvac, lighting, other]])
#     features_scaled = scaler.transform(features)
#
#     try:
#         prediction = model.predict(features_scaled)[0]
#
#         # Total energy usage
#         total_kwh = machine + hvac + lighting + other
#         total_cost = total_kwh * COST_PER_KWH
#
#         # Display result
#         if prediction == 1:
#             st.error("⚠ Inefficient Energy Usage Detected!")
#         else:
#             st.success("✅ Efficient Energy Usage")
#
#         # Confidence scores
#         if hasattr(model, "predict_proba"):
#             confidence = model.predict_proba(features_scaled)[0]
#         else:
#             confidence = [1 - prediction, prediction]
#
#         st.markdown(f"Confidence: {confidence[1]*100:.2f}% Inefficient, {confidence[0]*100:.2f}% Efficient")
#         st.progress(float(confidence[1]) if prediction == 1 else float(confidence[0]))
#
#         # 🧾 Prediction Summary Table
#         st.markdown("### 🧾 Prediction Summary")
#         summary_data = {
#             "Machine A (kWh)": machine,
#             "HVAC (kWh)": hvac,
#             "Lighting (kWh)": lighting,
#             "Other (kWh)": other,
#             "Total Usage (kWh)": total_kwh,
#             "Estimated Cost ($)": f"${total_cost:.2f}",
#             "Detected Inefficiency": "Yes" if prediction == 1 else "No"
#         }
#         st.table(summary_data)
#
#     except ValueError as e:
#         st.error(f"Feature mismatch: {e}")
#
# import streamlit as st
# import numpy as np
# import joblib
# from io import BytesIO
# from reportlab.pdfgen import canvas
#
# # Load model and scaler
# @st.cache_resource
# def load_model():
#     model = joblib.load("model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     return model, scaler
#
# model, scaler = load_model()
#
# st.title("⚡ Energy Inefficiency Detection")
#
# COST_PER_KWH = 0.12  # cost per unit
#
# # Input fields
# with st.form("input_form"):
#     machine = st.number_input("Machine A (kWh)", value=50.0)
#     hvac = st.number_input("HVAC (kWh)", value=30.0)
#     lighting = st.number_input("Lighting (kWh)", value=20.0)
#     other = st.number_input("Other (kWh)", value=10.0)
#     submitted = st.form_submit_button("Predict")
#
# if submitted:
#     features = np.array([[machine, hvac, lighting, other]])
#     features_scaled = scaler.transform(features)
#
#     try:
#         prediction = model.predict(features_scaled)[0]
#         total_kwh = machine + hvac + lighting + other
#         total_cost = total_kwh * COST_PER_KWH
#
#         if hasattr(model, "predict_proba"):
#             confidence = model.predict_proba(features_scaled)[0]
#         else:
#             confidence = [1 - prediction, prediction]
#
#         efficiency_status = "Inefficient" if prediction == 1 else "Efficient"
#         st.markdown(f"Confidence: {confidence[1]*100:.2f}% Inefficient, {confidence[0]*100:.2f}% Efficient")
#         st.progress(float(confidence[1]) if prediction == 1 else float(confidence[0]))
#
#         st.markdown("### 🧾 Prediction Summary")
#         st.table({
#             "Machine A (kWh)": machine,
#             "HVAC (kWh)": hvac,
#             "Lighting (kWh)": lighting,
#             "Other (kWh)": other,
#             "Total Usage (kWh)": total_kwh,
#             "Estimated Cost ($)": f"${total_cost:.2f}",
#             "Efficiency Status": efficiency_status
#         })
#
#         # ✅ Generate PDF
#         pdf_buffer = BytesIO()
#         c = canvas.Canvas(pdf_buffer)
#         c.setFont("Helvetica-Bold", 14)
#         c.drawString(50, 800, "Energy Inefficiency Detection Report")
#         c.setFont("Helvetica", 12)
#         y = 770
#         data = [
#             ("Machine A (kWh)", machine),
#             ("HVAC (kWh)", hvac),
#             ("Lighting (kWh)", lighting),
#             ("Other (kWh)", other),
#             ("Total Usage (kWh)", total_kwh),
#             ("Estimated Cost ($)", f"${total_cost:.2f}"),
#             ("Efficiency Status", efficiency_status),
#             ("Confidence (Inefficient)", f"{confidence[1]*100:.2f}%"),
#             ("Confidence (Efficient)", f"{confidence[0]*100:.2f}%")
#         ]
#
#         for label, value in data:
#             c.drawString(50, y, f"{label}: {value}")
#             y -= 20
#
#         c.save()
#         pdf_buffer.seek(0)
#
#         # ✅ Download Button
#         st.download_button(
#             label="📄 Download PDF Report",
#             data=pdf_buffer,
#             file_name="energy_report.pdf",
#             mime="application/pdf"
#         )
#
#     except ValueError as e:
#         st.error(f"Feature mismatch: {e}")

import streamlit as st
import numpy as np
import joblib
from io import BytesIO
from reportlab.pdfgen import canvas

# ---------------- Load model and scaler ----------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ---------------- App Title ----------------
st.title("⚡ Energy Inefficiency Detection")

# Cost constant
COST_PER_KWH = 0.12  # cost per unit in USD

# ---------------- Input Form ----------------
with st.form("input_form"):
    machine = st.number_input("Machine A (kWh)", value=50.0)
    hvac = st.number_input("HVAC (kWh)", value=30.0)
    lighting = st.number_input("Lighting (kWh)", value=20.0)
    other = st.number_input("Other (kWh)", value=10.0)
    submitted = st.form_submit_button("Predict")

# ---------------- On Submit ----------------
if submitted:
    # Prepare input data
    features = np.array([[machine, hvac, lighting, other]])
    features_scaled = scaler.transform(features)

    try:
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        total_kwh = machine + hvac + lighting + other
        total_cost = total_kwh * COST_PER_KWH

        # Get confidence if available
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(features_scaled)[0]
        else:
            confidence = [1 - prediction, prediction]

        # Efficiency result
        if prediction == 1:
            efficiency_status = "Inefficient"
            st.error("⚠ Inefficient Energy Usage Detected! Please review your consumption.")
        else:
            efficiency_status = "Efficient"
            st.success("✅ Efficient Energy Usage")

        # # Additional simple result display
        # if prediction == 1:
        #     st.error("⚠ Inefficient Energy Usage Detected!")
        # else:
        #     st.success("✅ Efficient Energy Usage")

        # Confidence and progress bar
        st.markdown(f"Confidence: {confidence[1]*100:.2f}% Inefficient, {confidence[0]*100:.2f}% Efficient")
        st.progress(float(confidence[1]) if prediction == 1 else float(confidence[0]))

        # ---------------- Summary Table ----------------
        st.markdown("### 🧾 Prediction Summary")
        st.table({
            "Machine A (kWh)": machine,
            "HVAC (kWh)": hvac,
            "Lighting (kWh)": lighting,
            "Other (kWh)": other,
            "Total Usage (kWh)": total_kwh,
            "Estimated Cost ($)": f"${total_cost:.2f}",
            "Efficiency Status": efficiency_status
        })

        # ---------------- PDF Report ----------------
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 800, "Energy Inefficiency Detection Report")
        c.setFont("Helvetica", 12)
        y = 770

        report_data = [
            ("Machine A (kWh)", machine),
            ("HVAC (kWh)", hvac),
            ("Lighting (kWh)", lighting),
            ("Other (kWh)", other),
            ("Total Usage (kWh)", total_kwh),
            ("Estimated Cost ($)", f"${total_cost:.2f}"),
            ("Efficiency Status", efficiency_status),
            ("Confidence (Inefficient)", f"{confidence[1]*100:.2f}%"),
            ("Confidence (Efficient)", f"{confidence[0]*100:.2f}%")
        ]

        for label, value in report_data:
            c.drawString(50, y, f"{label}: {value}")
            y -= 20

        c.save()
        pdf_buffer.seek(0)

        # ---------------- PDF Download Button ----------------
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name="energy_report.pdf",
            mime="application/pdf"
        )

    except ValueError as e:
        st.error(f"Feature mismatch: {e}")