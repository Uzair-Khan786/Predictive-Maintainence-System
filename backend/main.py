from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,computed_field
from typing import Literal,Annotated
import shap
import pickle
import pandas as pd
import numpy as np 

with open('my_model.pkl','rb') as f:
    model = pickle.load(f)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

app = FastAPI(title='Predictive Maintainence API')

explainer = shap.TreeExplainer(model)

class UserInput(BaseModel) :
    Air_temperature : Annotated[float, Field(..., ge=295.3, le=304.5, description='Air Temperature(in Kelvin) of the machine')]
    Process_temperature : Annotated[float, Field(..., ge=305.7, le=313.8, description='Process Temperature(in Kelvin) of the machine')]
    Rotational_speed : Annotated[int, Field(..., ge=1168, le=2886, description='Rotational Speed of the machine')]
    Torque : Annotated[float, Field(..., ge=3.8, le=76.6, description='Torque of the machine')]
    Tool_wear : Annotated[int, Field(..., ge=0, le=253, description='Tool Wear of the machine')]

    @computed_field
    @property
    def Temp_change(self) -> float:
        return self.Process_temperature - self.Air_temperature

    @computed_field
    @property
    def Overstrain_index(self) -> float:
        return self.Tool_wear * self.Torque
    
    @computed_field
    @property
    def Thermal_Efficiency(self) -> float:
        return self.Temp_change/self.Rotational_speed

    @computed_field
    @property
    def rotational_speed(self) -> float:
        return np.log1p(self.Rotational_speed)

def get_recommendations(data : UserInput,prediction) :
    recommendations = []

    if prediction == 1 :
        if data.Process_temperature > 310 or (data.Process_temperature - data.Air_temperature) > 10:
            recommendations.append('Check the machine cooling system.')

        if data.Torque > 45 :
            recommendations.append('Inspect machine load and motor conditions.')

        if data.Rotational_speed > 1800 :
            recommendations.append('Check whether rotational speed is within operating limits.')

        if data.Tool_wear > 180 :
            recommendations.append('Inspect or replace the respective tool.')

    if not recommendations :
        recommendations.append('Continue normal monitoring and scheduled maintainance.')

    return recommendations

@app.post('/predict')
def predict(data : UserInput):
    try :
        df = pd.DataFrame([{
            'Rotational_speed' : data.rotational_speed,
            'Torque' : data.Torque,
            'Tool_wear' : data.Tool_wear,
            'Temp_change' : data.Temp_change,
            'Overstrain_index' : data.Overstrain_index,
            'Thermal_Efficiency' : data.Thermal_Efficiency
        }])

        scaled_data = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        prediction = int(model.predict(scaled_df)[0])

        probability = float(model.predict_proba(scaled_df)[0][1])
        failure_probability = round(probability * 100,2)

        output = 'Failure' if prediction == 1 else 'No Failure'

        shap_values = explainer.shap_values(scaled_df)

        if isinstance(shap_values,list) :
            values = shap_values[1][0]
        else :
            values = shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[0]

        feature_impacts = sorted(
            zip(df.columns, np.abs(values)),
            key=lambda x: x[1],
            reverse=True
        )
        top_reasons = [feature[0] for feature in feature_impacts[:3] if prediction == 1]

        recommendations = get_recommendations(data,prediction)

        result = {
            "Prediction" : output,
            "Machine_Failure_Probability" : failure_probability,
            "Top_Reasons" : top_reasons,
            "Recommendation_Actions" : recommendations
        }

        return JSONResponse(status_code=200,content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")