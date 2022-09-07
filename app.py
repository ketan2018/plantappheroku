import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np 
from tensorflow.keras.preprocessing import image
st.markdown(""" <style> .mine {
font-size:48px ; font-family: 'Japanese';text-align:center;} 
</style> """, unsafe_allow_html=True)
st.markdown('<h1 class="mine"> Plant Seedling Classification App</p>', unsafe_allow_html=True)

st.markdown(""" <style> .label {
font-size:16px ; font-family: 'Japanese';text-align:center;} 
</style> """, unsafe_allow_html=True)



instructions = """
        The goal of this app is to prdict the category of the seedling using the provided plant Image.
        This app can classify your image only into the 12 categories which are shown below with thier
        respective image. This app identifies the Category by using the neural network model which is 
        trained on the dataset from kaggle competition.
        """
st.write(instructions)
col1, col2, col3,col4= st.columns(4)
bla_grs_image = Image.open('black-grass.jpg')
bla_grs_image_new= bla_grs_image.resize((375, 375))


cmn_chick_img=Image.open('common-chickweed.jpg')
cmn_chick_new= cmn_chick_img.resize((375, 375))
with col1:
    img1= st.image(bla_grs_image_new,use_column_width='auto')  
    st.markdown('<h4 class="label"> 1.Black-grass</p>', unsafe_allow_html=True)
with col2:
	img2=st.image('charlock-young.jpg'),st.markdown('<h4 class="label"> 2.Charlock</p>', unsafe_allow_html=True) 
    
with col3:
    img3 = st.image('cleavers-young.jpg'),st.markdown('<h4 class="label"> 3.Cleavers</p>', unsafe_allow_html=True)

with col4:
    img4= st.image(cmn_chick_new),st.markdown('<h4 class="label"> 4.Common Chickweed</p>', unsafe_allow_html=True)


col1, col2, col3,col4 = st.columns(4)


cmn_wht=Image.open('common_wheat.jpg')
cmn_wht_new=cmn_wht.resize((375,375))

loose=Image.open('loose-silky-bent.jpg')
loose_new=loose.resize((375,375))

maize=Image.open('corn-seedling.jpg')
maize_new=maize.resize((375,375))

with col1:
    img5 = st.image(cmn_wht_new),st.markdown('<h4 class="label"> 5.Common Wheat</p>', unsafe_allow_html=True) 
                                                                

with col2:
    img6 =st.image('fat-hen.jpg'),st.markdown('<h4 class="label"> 6.Fat Hen</p>', unsafe_allow_html=True)

with col3:
    img7= st.image(loose_new),st.markdown('<h4 class="label"> 7.Loose Silkybent</p>', unsafe_allow_html=True)

with col4:
    img8= st.image(maize_new),st.markdown('<h4 class="label"> 8.Maize</p>', unsafe_allow_html=True)

col1, col2, col3,col4 = st.columns(4)

sfd=Image.open('Shepherds-purse.jpg')
sfd_new=sfd.resize((375,375))

crn=Image.open('cransbil.jpeg')
crn_new=crn.resize((375,375))

sbt=Image.open('sugarbeet13.jpg')
sbt_new=sbt.resize((375,375))



with col1:
    img9= st.image('scentless-mayweed-young.jpg'),st.markdown('<h4 class="label"> 9.Scentless Mayweed</p>', unsafe_allow_html=True) 
                                                                

with col2:
    img10 =st.image(sfd_new),st.markdown('<h4 class="label"> 10.Shephers Purse</p>', unsafe_allow_html=True)

with col3:
    img11 = st.image(crn_new),st.markdown('<h4 class="label"> 11.Small Cranesbill</p>', unsafe_allow_html=True)

with col4:
    img12= st.image(sbt_new),st.markdown('<h4 class="label"> 12.Sugar Beet</p>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload your below(png only)")



if uploaded_file:
	st.title('Your Image:')
if uploaded_file:
    st.image(uploaded_file,width=300)

if uploaded_file:
    with st.spinner('Identifying...'):
        
        model=tf.keras.models.load_model('91_accuracy_model.h5')
        
        # from tensorflow.keras.utils import load_img
        # predicting images
        
        l=uploaded_file

        img=image.load_img(l, target_size=(200, 200))

        x=image.img_to_array(img)
        x /= 255
        x=np.expand_dims(x, axis=0)
        images = np.vstack([x])

        classes = model.predict(images, batch_size=10)
        cat=['Black-grass',
        'Charlock',
        'Cleavers',
        'Common Chickweed',
        'Common wheat',
        'Fat Hen',
        'Loose Silky-bent',
        'Maize',
        'Scentless Mayweed',
        'Shepherds Purse',
        'Small-flowered Cranesbill',
        'Sugar beet']

        pred_class=cat[classes.argmax()]
        st.write(f'This Image belongs to the {pred_class} category')