import streamlit as st
import model
import pandas as pd
st.set_page_config(layout="wide")
import matplotlib.pyplot as plt

def main():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.sidebar.image('Compunnel-Digital-Logo.png', width=125)
    st.sidebar.title('''**Solar Plant Analytics**''')
    st.markdown("""<style>[data-testid="stSidebar"][aria-expanded="true"] 
    > div:first-child {width: 450px;}[data-testid="stSidebar"][aria-expanded="false"] 
    > div:first-child {width: 450px;margin-left: -400px;}</style>""",
    unsafe_allow_html=True)

    uploaded_files= st.sidebar.file_uploader("Upload Data File",type=['xlsx','csv'],accept_multiple_files=False)


    if uploaded_files:
        options = st.sidebar.selectbox('Please Select',
                                       ['Solar Power Generation', 'Solar Plant Performance', 'RoI/Payback',
                                        'Solar Plant Service'])

        if options == 'Solar Power Generation':
            pred,train,df_dup,df = model.solar_generation(uploaded_files)

            train['TimeStamp'] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day, hour=df.Hour))

            train1 = train.to_numpy().tolist()
            pred1 = pred.tolist()
            train_vehID = [i[10] for i in train1]
            s1 = pd.Series(train_vehID, name='Date')
            s2 = pd.Series(pred1, name='Predicted_Generation')
            df_new = pd.concat([s1, s2], axis=1)

            st.info('Dataset')
            df_head = df_dup.head()
            st.dataframe(df_head)

            st.info("Predicted Response")
            st.write(df_new.T.to_html(escape=True), unsafe_allow_html=True)

            st.write(" ")

            st.info("PowerBI Dashboard")

            st.markdown('<iframe title="generated - Forecast" width="1000" height="650" src="https://app.powerbi.com/reportEmbed?reportId=30217de8-bd98-40af-8ba5-61a4d6a76991&autoAuth=true&ctid=0f8a5db0-6b60-4ca8-9fc9-8b2ae1cb809e&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLWluZGlhLWNlbnRyYWwtYS1wcmltYXJ5LXJlZGlyZWN0LmFuYWx5c2lzLndpbmRvd3MubmV0LyJ9" frameborder="0" allowFullScreen="false"></iframe>',unsafe_allow_html=True)



        elif options== 'Solar Plant Performance':

            df, decomposition,results_ARIMA= model.efficiency_pred(uploaded_files)

            st.info('Dataset')
            df_head = df.head()
            st.dataframe(df_head)

            st.info('Decompositional Plot')
            st.write( decomposition.plot())
            plt.show()

            st.info('Forecasted Result')
            st.write(results_ARIMA.plot_predict(1, 65))
            results_ARIMA.forecast(steps=12)

        elif options == 'RoI/Payback':
            pred, train, df= model.roi_pred(uploaded_files)
            st.info('Dataset')
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



        elif options == 'Solar Plant Service':
            pred, train, df = model.service_model(uploaded_files)


            st.info('Dataset')
            df_head = df.head()
            st.dataframe(df_head)

            train1 = train.to_numpy().tolist()
            pred1 = pred.tolist()
            train_vehID = [i[0] for i in train1]


            s1 = pd.Series(train_vehID, name='Plant ID')
            s2 = pd.Series(pred1, name='Predicted_Service')


            df_new = pd.concat([s1, s2], axis=1)
            df_new = df_new.replace(to_replace=1, value="Yes")
            df_new = df_new.replace(to_replace=0, value="No")

            st.info("Predicted Response")
            st.write(df_new.T.to_html(escape=False), unsafe_allow_html=True)

            st.write(" ")

            st.info("PowerBI Dashboard")

            st.markdown('<iframe title="service - Key Influencer" width="900" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=795dedf7-7510-4e64-bbbe-c05a904495df&autoAuth=true&ctid=0f8a5db0-6b60-4ca8-9fc9-8b2ae1cb809e&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLWluZGlhLWNlbnRyYWwtYS1wcmltYXJ5LXJlZGlyZWN0LmFuYWx5c2lzLndpbmRvd3MubmV0LyJ9" frameborder="0" allowFullScreen="true"></iframe>',unsafe_allow_html=True)



if __name__ == '__main__':
    main()
