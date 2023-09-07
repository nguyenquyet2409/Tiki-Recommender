import streamlit as st
import pandas as pd
import pickle

############################################################################################
#app():
st.title("Collaborative Filtering")
Review = pd.read_csv("ReviewRaw.csv")
Product = pd.read_csv('ProductRaw.csv', encoding="utf8")
pd.set_option('display.max_colwidth', None) # need this option to make sure cell content is displayed in full
Product['short_name'] = Product['name'].str.split('-').str[0]

############################################################################################

new_user_recs = pd.read_parquet('Tiki_U.parquet')
df_product_product_idx = pd.read_parquet('Tiki_P.parquet')
############################################################################################

# Define functions
#TAKE URL OF AN IMAGE 
def fetch_image(idx):
    selected_product = Product['image'].iloc[[idx]].reset_index(drop=True)
    url = selected_product[0]
    return url

############################################################################################

def get_user_recommendations(customer_id, new_user_recs, df_product_product_idx):
    find_user_rec = new_user_recs.loc[new_user_recs['customer_id'] == customer_id]

    user = find_user_rec.iloc[0]
    lst = []
    
    for row in user['recommendations']:   
        row_f = df_product_product_idx.loc[df_product_product_idx.product_id_idx == row['product_id_idx']] 
        row_f_first = row_f.iloc[0]
        lst.append((row['product_id_idx'], row_f_first['product_id'], row['rating']))
        
    dic_user_rec = {'customer_id': user.customer_id, 'recommendations': lst}
    # Lấy danh sách các sản phẩm đề xuất
    recommended_products = dic_user_rec['recommendations']
    # Tạo danh sách sản phẩm đề xuất kèm tên
    recommended_with_names = []
    # Duyệt qua danh sách sản phẩm đề xuất và thêm tên sản phẩm vào danh sách mới
    for rec in recommended_products:
        product_id_idx = rec[1]  # Lấy mã sản phẩm từ recommendation
        product_row = Product[Product['item_id'] == product_id_idx]
        if not product_row.empty:
            product_name = product_row.iloc[0]['short_name']
            product_image = product_row.iloc[0]['image']
            recommended_with_names.append((rec[0], product_name, rec[2],product_image))
            
    # Tạo DataFrame từ danh sách sản phẩm đề xuất kèm tên
    result_df = pd.DataFrame(recommended_with_names, columns=['product_id_idx', 'product_name', 'rating','image'])
    # Hiển thị DataFrame kết quả
    recommended_names = result_df['product_name'].values.tolist()
    recommended_images = result_df['image'].values.tolist()
    return recommended_names, recommended_images

############################################################################################

# Input customer id
number = st.number_input("Nhập customer id:", min_value=0)
st.write("customer id bạn nhập là: ", number)

# Choose maximum number of products that system will recommend
n = st.slider(
    'Chọn số lượng sản phẩm tối đa mà bạn muốn hệ thống giới thiệu (từ 1 đến 10)',
    1, 10, 5)
st.write('Số lượng sản phẩm tối đa bạn muốn được giới thiệu là:', n)

# 'Recommend' button
if st.button('Recommend'):
    names, images = get_user_recommendations(customer_id=number, new_user_recs=new_user_recs, df_product_product_idx=df_product_product_idx)
    names = names[:n]
    images = images[:n]
    cols = st.columns(n)
    for c in range(n):
        if c < len(images):
            with cols[c]:
                st.image(images[c], caption=names[c])

    st.write('Các sản phẩm customer id:', number, 'đã từng mua là:')
    products_bought_by_customer = Review.loc[Review['customer_id'] == number]
    product1 = Product[['item_id','price','brand','group','image','short_name']]
    # Thực hiện phép join giữa Product và products_bought_by_customer để lấy thông tin tên sản phẩm
    result_df = products_bought_by_customer.merge(product1, left_on='product_id', right_on='item_id', how='inner')

    result_df = result_df[["product_id", "short_name", "rating"]]

    st.write(result_df)


