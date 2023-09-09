import streamlit as st
import pandas as pd

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
    if not find_user_rec.empty:
        user = find_user_rec.iloc[0]
    # Rest of your code for processing the user recommendations
    else:
        st.warning("Không tìm thấy thông tin đề xuất cho customer id này.")
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
number = st.number_input("Nhập customer id (ví dụ: 494671, 709310, 10701688, 9909549...):", min_value=0)
st.write("Customer id bạn nhập là: ", number)


# Check if customer id exists
if number not in new_user_recs['customer_id'].unique():
    st.warning("Có thể bạn thích những sản phẩm này")
    
    # Get top 10 products with the most reviews
    top_products = Review['product_id'].value_counts().nlargest(10).index.tolist()
    
    # Fetch product names and images
    top_product_names = []
    top_product_images = []
    for product_id in top_products:
        product_row = Product[Product['item_id'] == product_id]
        if not product_row.empty:
            product_name = product_row.iloc[0]['short_name']
            product_image = product_row.iloc[0]['image']
            top_product_names.append(product_name)
            top_product_images.append(product_image)
    
    # Display top products
    cols = st.columns(len(top_product_names))
    for c in range(len(top_product_names)):
        with cols[c]:
            st.image(top_product_images[c], caption=top_product_names[c])
else:
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

        st.write('Các sản phẩm customer id:', number, 'đã mua gần đây:')
        products_bought_by_customer = Review.loc[(Review['customer_id'] == number) & (Review['rating'] >= 4)]
        if not products_bought_by_customer.empty:
            product1 = Product[['item_id','price','brand','group','image','short_name']]
            # Thực hiện phép join giữa Product và products_bought_by_customer để lấy thông tin tên sản phẩm
            result_df = products_bought_by_customer.merge(product1, left_on='product_id', right_on='item_id', how='inner')
            result_df['name_rating'] = result_df.apply(lambda row: f"{row['short_name']} (rating: {row['rating']}⭐)", axis=1)
            result_df = result_df[["product_id", "short_name", "rating",'image','name_rating']]
            names = result_df['name_rating'].values.tolist()
            images = result_df['image'].values.tolist()
            names = names[:10]
            images = images[:10]
            if len(names) < 10:
                cols = st.columns(len(names))
            else:
                cols = st.columns(10)
            for c in range(len(names)):
                if c < len(images):
                    with cols[c]:
                        st.image(images[c], caption=names[c])

