import sklearn.datasets
import os
# 打印内置图片的实际路径
img_path = os.path.join(sklearn.datasets.__path__[0], 'images', 'china.jpg')
print("图片路径：", img_path)