import jieba
import pandas as pd


def save_data(data):
    df = pd.DataFrame(data)
    # Save as CSV
    file_path = 'webinfo/lab1-1/dataset/selected_book_top_1200_data_tag_jieba.csv'
    df.to_csv(file_path, index=False, encoding='utf-8')


# 读取CSV文件
file_path = 'webinfo/lab1-1/dataset/selected_book_top_1200_data_tag.csv'  # 替换为你的CSV文件路径
oringal_data = pd.read_csv(file_path)
data = {"Book": [], "Tags": []}
for book, booktags in zip(oringal_data['Book'], oringal_data['Tags']):
    booktags = booktags.strip("{}")
    booktags_list = [item.strip().strip("'") for item in booktags.split(",")]
    words = []
    for tag in booktags_list:
        words = words+jieba.lcut(tag)
    data['Book'].append(book)
    data['Tags'].append(words)
save_data(data)
