import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
from gensim import corpora, models, similarities
import re
import seaborn as sns
import streamlit as st
import pickle

############################################################################################
st.title("Content Based Filtering")
Product = pd.read_csv('Product_clean.csv', encoding="utf8", index_col=0)
pd.set_option('display.max_colwidth', None) # need this option to make sure cell content is displayed in full
Product['short_name'] = Product['name'].str.split('-').str[0]
product_map = Product.iloc[:,[0,-1]]
product_list = product_map['short_name'].values

############################################################################################

# Define functions to use for both methods
##### TEXT PROCESSING #####
def process_text(document):
    # Change to lower text
    document = document.lower()
    # Remove HTTP links
    document = document.replace(
        r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '')
    # Remove line break
    document = document.replace(r'[\r\n]+', ' ')
    # Change / by white space
    document = document.replace('/', ' ') 
    # Change , by white space
    document = document.replace(',', ' ') 
    # Remove punctuations
    document = document.replace('[^\w\s]', '')
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in punctuation:
        document = document.replace(char, '')
    # Replace mutiple spaces by single space
    document = document.replace('[\s]{2,}', ' ')
    # Word_tokenize
    document = word_tokenize(document, format="text")   
    # Pos_tag
    document = pos_tag(document)    
    # Remove stopwords
    STOP_WORD_FILE = 'vietnamese-stopwords.txt'   
    with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
        stop_words = file.read()
    stop_words = stop_words.split()  
    document = [[word[0] for word in document if not word[0] in stop_words]] 
    return document

##### TAKE URL OF AN IMAGE #####
def fetch_image(idx):
    selected_product = Product['image'].iloc[[idx]].reset_index(drop=True)
    url = selected_product[0]
    return url

##### CHECK PRODUCT SIMILARITIES BY GENSIM MODEL AND RETURN NAMES & IMAGES OF TOP PRODUCTS WITH HIGHEST SIMILARITY INDEX #####

with open('dictionary.pkl', 'rb') as file:
    dictionary = pickle.load(file)
tfidf = models.tfidfmodel.TfidfModel.load("tfidf.tfidfmodel")
index = similarities.docsim.SparseMatrixSimilarity.load("index.docsim")


def show_similar_products(dataframe, index, selected_product_index, num_similar):
    # Chọn sản phẩm đang xem dựa trên chỉ số
    product_selection = dataframe.iloc[[selected_product_index]]
    
    # Lấy thông tin sản phẩm
    name_description_pre = product_selection['content'].to_string(index=False)
    view_product = name_description_pre.lower().split()
    
    # Chuyển từ khóa tìm kiếm thành Sparse Vectors
    kw_vector = dictionary.doc2bow(view_product)
    
    # Tính toán độ tương tự
    sim = index[tfidf[kw_vector]]
    
    # Sắp xếp danh sách sim theo thứ tự giảm dần
    sorted_sim_indices = sorted(range(len(sim)), key=lambda k: sim[k], reverse=True)
    
    # Tạo DataFrame chứa thông tin các sản phẩm tương tự
    similar_products_info = []
    for i in range(num_similar+1):
        index = sorted_sim_indices[i]
        similarity = sim[index]
        similar_product = dataframe.iloc[[index]]
        similar_product_name = similar_product['name'].values[0]  # Lấy tên sản phẩm
        similar_product_image = similar_product['image'].values[0]  # Lấy đường dẫn hình ảnh sản phẩm
        similar_product_info = {
            "index": similar_product.index.values[0],
            "name": f"{similar_product_name}",
            "score": f"{similarity:.2f}",
            "image": f"{similar_product_image}"  # Thêm thông tin hình ảnh vào dictionary
        }
        similar_products_info.append(similar_product_info)
        result = pd.DataFrame(similar_products_info)
    
    n_highest_score = result.sort_values(by='score', ascending=False).head(num_similar)
    # Extract product_id of above request
    id_tolist = list(n_highest_score['index'])
    recommended_names = []
    recommended_images = []
    for i in id_tolist:
        # Fetch the product names
        product_name = dataframe['name'].iloc[[i]]
        recommended_names.append(product_name.to_string(index=False))
        # Fetch the product images
        recommended_images.append(fetch_image(i))
    return recommended_names, recommended_images



############################################################################################

# Define separate page to demo each method
##### CONTENT_BASED FILTERING BY FIXED LIST #####
def filter_list():
    # Markdown name of Content_based method
    st.markdown("### By Product List")

    # Select product from list
    selected_idx = st.selectbox("Chọn sản phẩm muốn xem: ", range(len(product_list)), format_func=lambda x: product_list[x])

    # Fetch image of selected product
    idx = selected_idx
    st.image(fetch_image(idx))

    # Choose maximum number of products that system will recommend
    n = st.slider(
    'Chọn số lượng sản phẩm tối đa tương tự như trên mà bạn muốn hệ thống giới thiệu (từ 1 đến 10)',
    1, 10, 5)
    st.write('Số lượng sản phẩm tối đa được giới thiệu:', n)

    
    # 'Recommend' button
    if st.button('Recommend'):
        selection = Product.iloc[[idx]]
        selection_str = selection['content'].to_string(index=False)
        document = selection_str
        index = similarities.docsim.SparseMatrixSimilarity.load("index.docsim")
        names, images = show_similar_products(dataframe=Product, index=index, selected_product_index=idx, num_similar = n+2)
        names = names[1:-1]
        images = images[1:-1]
        cols = st.columns(n)
        for c in range(n):
            with cols[c]:
                st.image(images[c], caption = names[c])
    
##### CONTENT_BASED FILTERING BY INPUTING DESCRIPTION #####

def input_description_product(dataframe, index, input_product_name, num_similar):
    # Chuyển từ khóa tìm kiếm thành Sparse Vectors
    view_product = input_product_name.lower().split()
    kw_vector = dictionary.doc2bow(view_product)
    
    # Tính toán độ tương tự
    sim = index[tfidf[kw_vector]]
    
    # Sắp xếp danh sách sim theo thứ tự giảm dần
    sorted_sim_indices = sorted(range(len(sim)), key=lambda k: sim[k], reverse=True)
    
    # Tạo DataFrame chứa thông tin các sản phẩm tương tự
    similar_products_info = []
    for i in range(num_similar+1):
        index = sorted_sim_indices[i]
        similarity = sim[index]
        similar_product = dataframe.iloc[[index]]
        similar_product_name = similar_product['name'].values[0]  # Lấy tên sản phẩm
        similar_product_image = similar_product['image'].values[0]  # Lấy đường dẫn hình ảnh sản phẩm
        similar_product_info = {
            "index": similar_product.index.values[0],
            "name": f"{similar_product_name}",
            "score": f"{similarity:.2f}",
            "image": f"{similar_product_image}" 
            
        }
        similar_products_info.append(similar_product_info)
        result = pd.DataFrame(similar_products_info)
    n_highest_score = result.sort_values(by='score', ascending=False).head(num_similar)
    # Extract product_id of above request
    id_tolist = list(n_highest_score['index'])
    recommended_names = []
    recommended_images = []
    for i in id_tolist:
        # Fetch the product names
        product_name = dataframe['name'].iloc[[i]]
        recommended_names.append(product_name.to_string(index=False))
        # Fetch the product images
        recommended_images.append(fetch_image(i))
    return recommended_names, recommended_images

def input_description():
    # Markdown name of Content_based method
    st.markdown("### By Inputing Description")

    # input product description
    text_input = st.text_input(
        "Nhập mô tả sản phẩm để tìm kiếm: "
    )

    if text_input:
        st.write("Mô tả sản phẩm của bạn: ", text_input)

    # Choose maximum number of products that system will recommend
    n = st.slider(
    'Chọn số lượng sản phẩm tối đa tương tự như trên mà bạn muốn hệ thống giới thiệu (từ 1 đến 10)',
    1, 10, 5)
    st.write('Số lượng sản phẩm tối đa được giới thiệu:', n)

    # 'Recommend' button
    if st.button('Recommend'):
        document = ' '.join(map(str,process_text(text_input)))
        with open('dictionary.pkl', 'rb') as file:
            dictionary = pickle.load(file)
        tfidf = models.tfidfmodel.TfidfModel.load("tfidf.tfidfmodel")
        index = similarities.docsim.SparseMatrixSimilarity.load("index.docsim")
        names, images = input_description_product(dataframe=Product, index=index, input_product_name = text_input, num_similar = n+1)
        names = names[:n]
        images = images[:n]
        cols = st.columns(n)
        for c in range(n):
            with cols[c]:
                st.image(images[c], caption = names[c])



    ##### CALLING PAGE  #####
page_names_to_funcs = {
    "Chọn sản phẩm": filter_list,
    "Nhập mô tả": input_description
    }
selected_page = st.sidebar.selectbox("Chọn hình thức gợi ý sản phẩm", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()