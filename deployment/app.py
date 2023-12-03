import numpy as np
import streamlit as st
import pickle
import joblib
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Initialize transformers
LE = LabelEncoder()
SCL = StandardScaler()
OE = OneHotEncoder()

st.markdown("""
    <style>
    .main {
        background: url("https://github.com/Gbekoilias/Project_STA2017/blob/main/deployment/auto%20cover.jpg?raw=true") no-repeat center center fixed; 
        -webkit-background-size: cover;
        -moz-background-size: cover;
        -o-background-size: cover;
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

#load model using pickle
model_file_path='C:/Users/DONKAMS/Downloads/Project_STA2017/PIPEmodelGBC.sav'
model = pickle.load(open(model_file_path, 'rb'))

#main function
def show_prediction():
    #create the user interface
    # Text with emojis
    st.markdown("<h1 style='text-align: center; color: white;'>🚧 AutoInsurance Prediction 🚧</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>The essential app for your car insurance 🚗</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This app predicts the likelihood of a customer to buy an auto insurance policy. 📈</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please fill in the form below to get your prediction 🔮</p>", unsafe_allow_html=True)

    Gender=('Female', 'Male')
    ProductName=('Car Classic', 'Car Plus', 'CVTP', 'Customized Motor', 'CarFlex',
       'CarSafe', 'Motor Cycle', 'Muuve', 'Car Vintage')
    Car_Category=('JEEP', 'Saloon' ,'Truck', 'Pick Up', 'Mini Bus' ,'Pick Up > 3 Tons', 'Bus',
      'Van', 'Mini Van', 'Wagon' ,'Sedan', 'Shape Of Vehicle Chasis', 'Motorcycle',
        'Station 4 Wheel')
    Subject_Car_Colour=('Black', 'Grey' ,'Silver', 'Red', 'Green' ,'Ash', 'Blue' 'White' , 'Brown', 'Gold',
            'Purple', 'Yellow')
    Subject_Car_Make=('REXTON', 'TOYOTA' ,'Honda' ,'Ford' ,'Iveco', 'Lexus', 'DAF', 'Mercedes',
         'Nissan' ,'Hyundai', 'Kia', 'ACURA', 'Mitsubishi' ,'Mack' ,'Peugeot',
         'Volkswagen', 'Jeep', 'Range Rover', 'Audi', 'Scania', 'Skoda', 'Land Rover',
        'Infiniti', 'BMW', 'Land Rover.', 'Dodge', 'Volvo', 'Suzuki', 'GMC', 'Man',
        'Mazda', 'Chevrolet', 'CHANGAN', 'Porsche' ,'MINI COOPER', 'Fiat', 'GAC',
        'Wrangler Jeep', 'Isuzu', 'Raston', 'Chrysler', 'Jaguar', 'Pontiac' ,'Subaru')
    LGA_Name =('Badagry', 'Ikeja', 'Municipal Area Council', 'Apapa' ,'Oshodi-Isolo',
    'Alimosho','Okpe', 'Iseyin', 'Abeokuta North', 'Lagos Mainland', 'Ojo',
    'Agege', 'Kosofe' ,'Ifo', 'Eti Osa', 'Burutu', 'Surulere Lagos State',
    'Kaduna South', 'Port Harcourt', 'Kano Municipal', 'Ikorodu' ,'Uyo' ,'Mushin',
    'Katagum', 'Amuwo-Odofin', 'Ibadan South-West' ,'Zaria', 'Lagos Island',
    'Shomolu', 'Ido', 'Enugu East', 'Ondo West', 'Awka South', 'Gwagwalada', 'Ila',
    'Ijebu Ode', 'Ifako-Ijaiye', 'Ijero', 'Shagamu', 'Nnewi North', 'Bomadi',
    'Akinyele', 'Oyo', 'Oredo', 'Calabar Municipal', 'Warri North', 'Oguta',
    'Bwari', 'Egbeda', 'Epe', 'Osogbo', 'Idanre', 'Ilorin West', 'Lagelu',
    'Ibadan South-East', 'Katcha', 'Kaduna North', 'Kudan', 'Olorunsogo',
    'Ibadan North-West', 'Ife Central', 'Udi', 'Bonny', 'Essien Udim',
    'Owerri West', 'Warri South', 'Tafa', 'Ogbomosho South', 'Biase', 'Sapele',
    'Orolu', 'Umuahia South', 'Enugu North', 'Bende', 'Ovia South-West',
    'Ajeromi-Ifelodun', 'Argungu', 'Awe', 'Ibadan North-East', 'Ogun Waterside',
    'Jos North', 'Yala', 'Akure South', 'Jos South' 'Owo' 'Ondo East'
    'Nnewi South', 'Oyi', 'Isu', 'Awka North', 'Anambra East', 'Ose', 'Ikot Ekpene',
    'Lavun', 'Ogba/Egbema/Ndoni', 'Ejigbo', 'Yorro', 'Akoko-Edo', 'Goronyo',
    'Ado-Odo/Ota', 'Oyo East', 'Chanchaga', 'Esan West', 'Ikenne', 'Tofa',
    'Ekeremor', 'Owerri North', 'Garko' 'Ibeno' 'Obafemi Owode' 'Esan Central',
    'Ethiope East', 'Mbo', 'Okobo', 'Ado Ekiti', 'Owerri Municipal', 'Bauchi',
    'Ijebu East', 'Ilesa West', 'Orsu', 'Onitsha South', 'Ughelli North',
    'Nwangele', 'Abakaliki', 'Ekiti East', 'Oye', 'Irepodun', 'Etsako West',
    'Ndokwa East', 'Tai', 'Onitsha North', 'Ilorin East', 'Bida', 'Ika',
    'Ife North', 'Ikwerre', 'Akure North', 'Abeokuta South')
    State=('Lagos', 'Federal Capital Territory', 'Delta', 'Oyo', 'Ogun', 'Kaduna',
    'Rivers', 'Kano', 'Akwa Ibom', 'Bauchi', 'Enugu', 'Ondo', 'Anambra', 'Osun',
    'Ekiti', 'Edo', 'Cross River', 'Imo' ,'Kwara', 'Niger' ,'Abia' ,'Kebbi',
    'Nasarawa' ,'Plateau', 'Taraba', 'Sokoto', 'Bayelsa', 'Ebonyi')
    ProductName=('Car Classic', 'Car Plus', 'CVTP', 'Customized Motor', 'CarFlex', 'CarSafe',
    'Motor Cycle', 'Muuve', 'Car Vintage')
    
    Gender=st.selectbox("Gender",Gender)
    Age=st.slider("Age",1,120)
    No_Pol=st.slider("Number of Policy the Customer has",1,7)
    Car_Category=st.selectbox("What kind of Car do you have?",Car_Category)
    Subject_Car_Colour=st.selectbox("What is your Car Colour?",Subject_Car_Colour)
    Subject_Car_Make=st.selectbox("Car Make",Subject_Car_Make)
    LGA_Name=st.selectbox("Where is the policy purchased?",LGA_Name)
    State=st.selectbox("State where policy was purchased?",State)
    ProductName=st.selectbox("Name of the car product?",ProductName)
    ok=st.button("Check if it's a claim or not")
    from sklearn.preprocessing import LabelEncoder

    # Initialize LabelEncoder
    LE = LabelEncoder()

    if ok:
        # Convert categorical variables to numeric
        Gender = LE.fit_transform([Gender])[0]
        Car_Category = LE.fit_transform([Car_Category])[0]
        Subject_Car_Colour = LE.fit_transform([Subject_Car_Colour])[0]
        Subject_Car_Make = LE.fit_transform([Subject_Car_Make])[0]
        LGA_Name = LE.fit_transform([LGA_Name])[0]
        State = LE.fit_transform([State])[0]
        ProductName = LE.fit_transform([ProductName])[0]

        # Scale numeric variables
        Age = SCL.fit_transform([[Age]])[0][0]
        No_Pol = SCL.fit_transform([[No_Pol]])[0][0]

        # Create input_data array
        input_data = np.array([[Gender, Age, No_Pol, Car_Category, Subject_Car_Colour, Subject_Car_Make, LGA_Name, State, ProductName]])

        # Convert input_data to a pandas DataFrame
        input_data = pd.DataFrame(input_data, columns=['Gender', 'Age', 'No_Pol', 'Car_Category', 'Subject_Car_Colour', 'Subject_Car_Make', 'LGA_Name', 'State', 'ProductName'])

        claim = model.predict(input_data)
        prediction = model.predict_proba(input_data)[:, 1]
        if prediction >= 0.16: # 'spam':
            st.subheader("The client will renew its claim")
        else:
                st.subheader("The client will not renew its claim")
show_prediction()
# Make predictions
#prediction = model.predict(input_data)  # Replace with your actual prediction logic

# Display prediction to the user
#st.subheader('Prediction Result:')
#if prediction == 1:  # Assuming binary classification
    #st.write('The prediction is Positive')
#else:
    #st.write('The prediction is Negative')
#st.write('Thank you for using our app')