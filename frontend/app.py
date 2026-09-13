import streamlit as st
import requests
import os

url = os.getenv("URL")

st.title("Predictive Maintainence System")
st.markdown("Enter the details below :")

Air_temperature = st.number_input("Air temperature",min_value=295.3,max_value=304.5,value=300.0)
Process_temperature = st.number_input("Process temperature",min_value=305.7,max_value=313.8,value=310.0)
Rotational_speed = st.number_input("Rotational Speed",min_value=1168,max_value=2886,value=1503)
Torque = st.number_input("Torque",min_value=3.8,max_value=76.6,value=40.0)
Tool_wear = st.number_input("Tool Wear",min_value=0,max_value=253,value=108)

if st.button("Predict Machine Status") :
    input_data = {
        "Air_temperature" : Air_temperature,
        "Process_temperature" : Process_temperature,
        "Rotational_speed" : Rotational_speed,
        "Torque" : Torque,
        "Tool_wear" : Tool_wear
    }

    try :
        response = requests.post(url,json=input_data,timeout=60)

        if response.status_code == 200 :
            result = response.json()
            prediction = result['Prediction']
            if prediction == 'No Failure' :
                st.success(f"Prediction : 🟢{prediction}")
            else :
                st.error(f"Prediction : 🔴{prediction}")

            st.progress(value=float(result['Machine_Failure_Probability'])/100,text=f"📊Failure Probability : {float(result['Machine_Failure_Probability']):.1f}%")

            if prediction == 'Failure' :
                reasons = result['Top_Reasons']
                formatted_list = "\n".join([f"{val}. {reason}" for val,reason in enumerate(reasons,1)])
                st.warning(f"Top Reasons : \n{formatted_list}")
            
            recommendations = result['Recommendation_Actions']
            format_list = "\n".join([f"{i}. {action}" for i,action in enumerate(recommendations,1)])
            st.info(f"Recommendation Actions : \n{format_list}")

        else :
            st.error(f"Error : {response.status_code}")
            st.write(response.text)
        
    except requests.exceptions.ConnectionError:
        st.error("❌Unable to connect to the prediction FastAPI.")
