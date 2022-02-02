import numpy as np
import streamlit as st
import model
import pandas as pd
import model_train
import pickle
st.set_page_config(layout='wide')
import matplotlib.pyplot as plt

pickle_in=open('model.pkl','rb')
modeltrain=pickle.load(pickle_in)
def prediction(Gender,Married,ApplicantIncome,LoanAmount,Credit_History):
    if Gender=='Male':
        Gender=0
    else:
        Gender=1
    if Married=='Unmarried':
        Married=0
    else:
        Married=1
    if Credit_History=="No_Dept":
        Credit_History=1
    else:
        Credit_History=0

    prediction=modeltrain.predict(np.array([Gender,Married,ApplicantIncome,LoanAmount,Credit_History]).reshape(1,-1))
    if prediction==0:
        predict='Rejected'
    else:
        predict='Approved'
    return predict

def main():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.sidebar.image('Compunnel-Digital-Logo.png', width=125)

    st.sidebar.title('Solar Analytics')

    uploaded_files = st.sidebar.file_uploader('Upload Data File', type=['xlsx', 'csv'], accept_multiple_files=False)

    if uploaded_files:
        options = st.sidebar.selectbox('Please Select', ['Solar Generation','Solar Plant Performance',
                                                          'ROI/Payback','Test'])

        if options == "Solar Generation":
            pred, train, df_dup, df = model.solar_generation(uploaded_files)

            train['timestamp'] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day, hour=df.Hour))
            train1 = train.to_numpy().tolist()
            pred1 = pred.tolist()

            train_vehID = [i[10] for i in train1]
            s1 = pd.Series(train_vehID, name='Date')
            s2 = pd.Series(pred1, name='Predicted Gen')

            df_new = pd.concat([s1, s2], axis=1)

            st.info('Dataset')
            df_head = df_dup.head()

            st.dataframe(df_head)

            st.info('Predicted Resp')
            st.write(df_new.T.to_html(escape=True), unsafe_allow_html=True)

        if options == 'Solar Plant Performance':
            df, decomposition, results_ARIMA = model.efficiency_pred(uploaded_files)

            st.info('Dataset')
            df_head = df.head()
            st.dataframe(df_head)

            st.info('Decompositional Plot')
            st.write(decomposition.plot())
            plt.show()

            st.info('Forecasted Result')
            st.write(results_ARIMA.plot_predict(1, 65))
            results_ARIMA.forecast(steps=12)

        if options == 'ROI/Payback':
            st.info('Dataset')
            pred, train, df= model.roi_pred(uploaded_files)
            df_head=df.head()
            st.dataframe(df_head)
            train1 = train.to_numpy().tolist()

            x,y=pred.T

            x = x.tolist()
            y = y.tolist()
            train_ID = [i[0] for i in train1]
            s1 = pd.Series(train_ID, name='Plant ID')
            s2 = pd.Series(x, name='Predicted Payback Period')
            s3 = pd.Series(y, name='Predicted RoI')
            df_new = pd.concat([s1, s2,s3], axis=1)

            df_new['Plant ID']=df_new['Plant ID'].astype(int)


            st.info("Predicted Response")
            st.write(df_new.to_html(escape=False), unsafe_allow_html=True)

            st.write(" ")

            st.info("PowerBI Dashboard")

            st.markdown('<iframe title="RoI - Key Influencer" width="950" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=f0ce2e93-1601-4185-9a53-380f32eaf70f&autoAuth=true&ctid=0f8a5db0-6b60-4ca8-9fc9-8b2ae1cb809e&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLWluZGlhLWNlbnRyYWwtYS1wcmltYXJ5LXJlZGlyZWN0LmFuYWx5c2lzLndpbmRvd3MubmV0LyJ9" frameborder="0" allowFullScreen="false"></iframe>',unsafe_allow_html=True)

        if options =='Test':
            Gender=st.selectbox('Gender',("Male","Female"))
            Married=st.selectbox('Maritial Status',("Unmarried","Married"))
            applicantincome=st.number_input("Applicant Monthly Income")
            loan_amount=st.number_input("Total Loan Amount")
            CreditHistory=st.selectbox('Credit History',("No Dept","In Dept"))
            result=""
            if st.button("Predict"):
                result=prediction(Gender,Married,applicantincome,loan_amount,CreditHistory)
                st.success('Your loan is {}:'.format(result))
                print(loan_amount)


if __name__ == '__main__':
    main()
