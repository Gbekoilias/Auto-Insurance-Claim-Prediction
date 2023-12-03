import numpy as np
import streamlit as st
import pickle
import joblib
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib

# CSS for background image
st.markdown("""
    <style>
    .reportview-container {
        background: url("C:/Users/DONKAMS/Downloads/Project_STA2017/deployment/auto cover.jpg") no-repeat center center fixed; 
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
    # CSS for background image
    st.markdown("""
    <style>
    .reportview-container {
        background: url("https://th.bing.com/th/id/R.bab2aff7ce7be842362292e785e69b44?rik=oKyJv25KNNdYSg&pid=ImgRaw&r=0") no-repeat center center fixed; 
        -webkit-background-size: cover;
        -moz-background-size: cover;
        -o-background-size: cover;
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white;'>AutoInsurance Prediction 🙂</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>The essential app for your car insurance 🚗</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This app predicts the likelihood of a customer to buy an auto insurance policy. 📈</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please fill in the form below to get your prediction 🔮</p>", unsafe_allow_html=True)

    gender=('Female', 'Male')
    product=('Car Classic', 'Car Plus', 'CVTP', 'Customized Motor', 'CarFlex',
       'CarSafe', 'Motor Cycle', 'Muuve', 'Car Vintage')
    category=('JEEP', 'Saloon', 'Truck', 'Pick Up', 'Mini Bus',
       'Pick Up > 3 Tons', 'Bus', 'Van', 'Mini Van', 'Wagon', 'Sedan',
       'Shape Of Vehicle Chasis', 'Motorcycle', 'Station 4 Wheel')
    color=('Black', 'Grey', 'Silver', 'Red', 'Green', 'Blue', 'White',
       'Brown', 'Gold', 'As Attached', 'Orange', 'Dark Grey',
       'White & Red', 'Light Green', 'Dark Gray', 'B.Silver', 'Purple',
       'Red & Yellow', 'Yellow', 'Dark Red', 'Black & White',
       'White & Blue', 'Beige', 'Light Blue', 'Gray & Silver',
       'White & Yellow', 'Dark Blue', 'Black & Orange', 'Yellow & White',
       'Beige Mitalic', 'Light Gray')
    make=('REXTON', 'TOYOTA', 'Honda', 'Ford', 'Iveco', 'Lexus', 'DAF',
       'Mercedes', 'Nissan', 'Hyundai', 'Kia', 'ACURA', 'Mitsubishi',
       'Mack', 'Peugeot', 'Volkswagen', 'Jeep', 'Range Rover', 'Audi',
       'Scania', 'Skoda', 'Land Rover', 'Infiniti', 'BMW', 'Land Rover.',
       'Dodge', 'Volvo', 'Suzuki', 'GMC', '.', 'Man', 'Mazda',
       'Chevrolet', 'CHANGAN', 'Porsche', 'MINI COOPER', 'Fiat', 'GAC',
       'Wrangler Jeep', 'Isuzu', 'Raston', 'Chrysler', 'Jaguar',
       'Pontiac', 'Subaru')
    lga=('Badagry', 'Ikeja', 'Abuja Municipal', 'Yaba', 'Oshodi Isolo',
       'Alimosho', 'Okpe Delta State', 'Ibadancentral', 'Abeokuta',
       'Lekki', 'Ibeju Lekki', 'Agege', 'Kosofe', 'Ogun', 'Apapa',
       'Victoria Island', 'Obio Akpor', 'Surulere', 'Kaduna South',
       'Lagos Mainland', 'Gbagada', 'Port Harcourt', 'Central',
       'Ebute Metta', 'Uyo', 'Mushin', 'Katagum', 'Amuwo Odofin',
       'Ibadan South West', 'Isheri', 'Festac', 'Zaria', 'Lagos Island',
       'Shomolu', 'Ido', 'Enugu East', 'Ajah', 'Ketu', 'Ondo West',
       'Eti Osa', 'Ikorodu', 'Awka South', 'Asokoro District',
       'Ile Oluji', 'Ijebu Ode', 'Ifako Ijaiye', 'Oshodi', 'Alagbado',
       'Abuja', 'Shagamu', 'Nnewi North', 'Aboh Mbaise', 'Akinyele',
       'Oyo', 'Ogbadibo', 'Oredo', 'Ilupeju', 'Calabar', 'Akute',
       'Warri Central', 'Ifo', 'Ikoyi', 'Bekwarra', 'Oguta', 'Bwari',
       'Egbeda', 'Epe', 'Osogbo', 'Idanre', 'Kano Municipal',
       'Ilorin West', 'Lagelu Ogbomosho North', 'Ibadan South East',
       'Katcha', 'Isolo', 'Anthony Village', 'Maryland', 'Ipaja',
       'Kaduna North', 'Lagos', 'Magodo', 'Kaduna', 'Olorunsogo',
       'Ibadan North West', 'Ojota', 'Ife Central', 'Nsit Ubium', 'Bonny',
       'Benin', 'Essien Udim', 'Owerri West', 'Warri', 'Minna',
       'Ogbomosho South', 'Biase', 'Asaba', 'Orolu', 'Sapele',
       'Umuahia South', 'Ile Ife', 'Ojodu', 'Abule Egba', 'Enugu North',
       'Ovia South West', 'Okota', 'Argungu', 'Ajegunle Lagos State',
       'Arepo', 'Ibadan North East', 'Ogun Waterside', 'Jos North',
       'Marina', 'Rivers', 'Dopemu', 'Akure South', 'Jos South',
       'Obanikoro', 'Ondo', 'Orile Iganmu', 'Nnewi', 'Oyi',
       'Owerri Municipal', 'Akoka', 'Awka North', 'Ojo', 'Awoyaya',
       'Onitsha', 'Akwa Ibom', 'Anambra East', 'Ajeromi Ifelodun', 'Iba',
       'Ikot Ekpene', 'Ifako', 'Niger State', 'Ogba Egbema Ndoni',
       'Ejigbo', 'Abeokuta North', 'Yorro', 'Ilesha', 'Ajao Estate',
       'Ekiti East', 'Ikeja G R A', 'Ikotun', 'Akoko Edo', 'Obalende',
       'Goronyo', 'Ado Odo Ota', 'Oyo East', 'Udi Agwu', 'Chanchaga',
       'Esan West', 'Ikenne', 'Kano', 'Ilasamaja', 'Ekeremor', 'Oniru',
       'Sango Otta', 'Owerri North', 'Garko', 'Sangotedo', 'Ukpoba',
       'Obafemi Owode', 'Ilorin', 'Esan Central', 'Akure', 'Ethiope East',
       'Quaan Pan', 'Warri North', 'Akuku Toru', 'Ado Ekiti', 'Owerri',
       'Bauchi', 'Ijebu East', 'Umuahia', 'Ilesa West', 'Orsu',
       'Onitsha South', 'Ughelli North', 'Warri South', 'Nwangele',
       'Abakaliki', 'Ekiti', 'Alapere', 'Irepodun', 'Etsako West',
       'Ndokwa East', 'Tai', 'Owode', 'Onitsha North', 'Ilorin East',
       'Bida', 'Nnewi South', 'Olamaboro', 'Ife North', 'Ikwerre',
       'Palm Groove', 'Akure North', 'Abeokuta South', 'Ogbomoso')
    state=('Lagos State',
        'Rivers State', 'Kaduna State',
       'Benue State', 'Akwa Ibom State', 'Bauchi State', 'Oyo State',
        'Enugu State', 'Ondo State', 'Anambra State',
        'Ogun State', 
       'FCT', 'Imo State', 'Benue State',
       'Edo State', 'Cross-River State', 'Osun State',
       'Kano State', 'Kwara State', 'Niger State', 'Edo State', 'Delta State', 'Abia State', 'Kebbi State', 'Plateau State', 'Taraba State', 'Ekiti State',
       'Sokoto State',
       'Bayelsa State',  'Abia State',
       'Ebonyi State', 'Kogi State')
    
    gender=st.selectbox("Gender of the Customer",gender)
    product=st.selectbox("Name of the Insurance policy",product)
    category=st.selectbox("Type of Car",category)
    color=st.selectbox("Car Colour",color)
    lga=st.selectbox("City where the policy was purchased",lga)
    state=st.selectbox("State where policy was purchased",state)
    make=st.selectbox("Car Make",make)

    age=st.slider("Age of the customer (Age range)",15,100,10)
    pol=st.slider("Number of Policy the Customer has",1,5)

    ok=st.button("Check if it's a claim or not")
    if ok:
        X=np.array([[gender,product,category,color,lga,state,make,age,pol]])
        X[:,0]=le_gender.fit_transform(X[:,0])
        X[:,1]=le_product.fit_transform(X[:,1])
        X[:,2]=le_category.fit_transform(X[:,2])
        X[:,3]=le_color.fit_transform(X[:,3])
        X[:,4]=le_lga.fit_transform(X[:,4])
        X[:,5]=le_state.fit_transform(X[:,5])
        X[:,6]=le_make.fit_transform(X[:,6])
        X=X.astype(float)

        claim=classifier.predict(X)
        #st.subheader(f"T {claim}")

        prediction=classifier.predict_proba(X)[:, 1]
        #st.subheader(f"T {prediction}")
        if prediction >= 0.16: # 'spam':
            st.subheader("The client will renew its claim")
        else:
            st.subheader("The client will not renew its claim")
        #st.subheader(f"{prediction}")
show_prediction()    

# Example input fields for user input
age = st.number_input('Enter Age', min_value=0, max_value=120, value=30)
gender = st.selectbox('Select Gender', ['Male', 'Female'])

# Prepare user input as required by your model
# For example, assuming you have 'age' and 'gender' as features
input_data = pd.DataFrame({'Age': [age], 'Gender': [gender]})  # Assuming 'Age' and 'Gender' as features

# Make predictions
prediction = model.predict(input_data)  # Replace with your actual prediction logic

# Display prediction to the user
st.subheader('Prediction Result:')
if prediction == 1:  # Assuming binary classification
    st.write('The prediction is Positive')
else:
    st.write('The prediction is Negative')
st.write('Thank you for using our app')