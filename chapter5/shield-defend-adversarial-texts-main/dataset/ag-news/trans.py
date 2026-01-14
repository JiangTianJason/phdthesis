import pandas as pd

#Load Data
data = pd.read_csv(r'train_source.csv')
testdata = pd.read_csv(r'test_source.csv')

#Set Column Names
data.columns = ['ClassIndex', 'Title', 'Description']
testdata.columns = ['ClassIndex', 'Title', 'Description']

#Combine Title and Description
X_train = data['Title'] + " " + data['Description'] # Combine title and description (better accuracy than using them as separate features)
y_train = data['ClassIndex'].apply(lambda x: x-1).values # Class labels need to begin from 0

x_test = testdata['Title'] + " " + testdata['Description'] # Combine title and description (better accuracy than using them as separate features)
y_test = testdata['ClassIndex'].apply(lambda x: x-1).values # Class labels need to begin from 0


# 创建包含文本和标签的数据框
train_data = pd.DataFrame({'text': X_train, 'label': y_train})
test_data = pd.DataFrame({'text': x_test, 'label': y_test})

# 为每个类别抽取2500个数据构成验证集
val_data = train_data.groupby('label').head(2500)

# 从训练集中删除验证集数据
train_data = train_data[~train_data.index.isin(val_data.index)]

# 保存验证集数据到val.csv文件
val_data.to_csv('val.csv', index=False)

# 保存更新后的训练集数据到train.csv文件
train_data.to_csv('train.csv', index=False)

test_data.to_csv('test.csv', index=False)
