from cmath import nan
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, Normalizer, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


#0.太多空值的欄位先刪掉、1.補值(mean/majority)、2.正規化(把值改成0~1之間)、3.feature selection

df = pd.read_csv('secondary_data.csv', delimiter = ";", header = 'infer')
print(df.isnull().sum()) #看null有多少個，超過一半的欄位(30534個)就刪掉
print(df.shape)
df = df.drop(['spore-print-color', 'stem-root','veil-color','veil-type','stem-surface','stem-root'], axis=1)
print(df.isnull().sum()) #查看空值的欄位
print(df.shape)
print(df)

#補值：以最高出現次數取代nan
df['cap-surface'].fillna(df['cap-surface'].mode()[0], inplace=True)
df['gill-attachment'].fillna(df['gill-attachment'].mode()[0], inplace=True)
df['gill-spacing'].fillna(df['gill-spacing'].mode()[0], inplace=True)
df['ring-type'].fillna(df['ring-type'].mode()[0], inplace=True)
print(df.isnull().sum())

#把string轉換成數值
lb = LabelEncoder()
df.iloc[:,0] = lb.fit_transform(df.iloc[:,0])
df.iloc[:,2] = lb.fit_transform(df.iloc[:,2])
df.iloc[:,3] = lb.fit_transform(df.iloc[:,3])
df.iloc[:,4] = lb.fit_transform(df.iloc[:,4])
df.iloc[:,5] = lb.fit_transform(df.iloc[:,5])
df.iloc[:,6] = lb.fit_transform(df.iloc[:,6])
df.iloc[:,7] = lb.fit_transform(df.iloc[:,7])
df.iloc[:,8] = lb.fit_transform(df.iloc[:,8])
df.iloc[:,11] = lb.fit_transform(df.iloc[:,11])
df.iloc[:,12] = lb.fit_transform(df.iloc[:,12])
df.iloc[:,13] = lb.fit_transform(df.iloc[:,13])
df.iloc[:,14] = lb.fit_transform(df.iloc[:,14])
df.iloc[:,15] = lb.fit_transform(df.iloc[:,15])

for i in range(0,16):
    '''迴圈裡無法排除非string欄位
    print(df[df.columns[i]].unique())
    if i == 1 or 9 or 10:
        pass
    if i == 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 11 or 12 or 13 or 14 or 15:
        df.iloc[:,i] = lb.fit_transform(df.iloc[:,i])
    else:
        pass
    if df.columns[i] == 'cap-diameter' or 'stem-height' or 'stem-width':
        pass
    elif df.columns[i] == 'class' or 'cap-shape' or 'cap-surface' or 'cap-color' or 'does-bruise-or-bleed' or 'gill-attachment' or 'gill-color' or 'stem-root' or 'stem-surface' or 'stem-color' or 'veil-type' or 'veil-color' or 'has-ring' or 'ring-type' or 'habitat' or 'season':
        df.iloc[:,i] = lb.fit_transform(df.iloc[:,i])
    else:
        pass
    '''
    print(df.columns[i], df[df.columns[i]].unique())

print(df)
#print(df.isnull().sum()) #沒空值了

'''#正規化 norm max/L1/L2
norm_max = Normalizer(norm='max')
norm_L1 = Normalizer(norm='l1')
norm_L2 = Normalizer(norm='l2')
#scaled_data = norm_max.fit_transform(np.array(df).reshape(-1,len(df.columns)))
scaled_data = norm_L1.fit_transform(np.array(df).reshape(-1,len(df.columns)))
#scaled_data = norm_L2.fit_transform(np.array(df).reshape(-1,len(df.columns)))
print('normalization')
print(scaled_data, scaled_data.shape)
'''

#標準化
ss = StandardScaler()
rb1 = RobustScaler(quantile_range=(25, 75))
scaled_data = ss.fit_transform(np.array(df).reshape(-1,16))
#scaled_data = rb1.fit_transform(np.array(df).reshape(-1,16))
print('standardization')
print(scaled_data, scaled_data.shape)



'''#norm max時
#觀察variance x12 < x13, x1 < x10 且 correlation >0.5的feature x刪掉
scaled_data = np.delete(scaled_data, 12, 1)
scaled_data = np.delete(scaled_data, 1, 1)
print(scaled_data, scaled_data.shape)
'''

'''#norm L1/norm L2時variance x1 < x7 and correlation with 7 <0.5
scaled_data = np.delete(scaled_data, 0, 1)
scaled_data = np.delete(scaled_data, 0, 1)
print(scaled_data, scaled_data.shape)
'''

#variance對於StandardScaler沒用,所以要看standardization前的變異
print('raw data variance')
print(df.var(axis=0), df.shape)
scaled_data = np.delete(scaled_data, 1, 1)
scaled_data = np.delete(scaled_data, 12, 1)
print(scaled_data.shape)


'''#RobustScaler時
scaled_data = np.delete(scaled_data, 10, 1)
scaled_data = np.delete(scaled_data, 12, 1)
print(scaled_data.shape)
'''

#計算每欄feature中的變異 變異太小 刪掉
vt = VarianceThreshold(threshold=1e-03)
x_variance = vt.fit_transform(scaled_data)
print('x variance')
print(x_variance.var(axis=0), x_variance.shape)
#print(X_t.var(axis=0))


#變異數間相關性 >0.5~0.6可考慮刪其一變數
cor = pd.DataFrame(x_variance)
cor = cor.corr()
print('correlation')
print(cor)

#檢查correlation >0.5的feature(x)
for i in range(0, len(x_variance[0])):
    for j in range(0, len(x_variance[0])):
        if cor[i][j] >=0.5 and cor[i][j]<1:
            print(cor.columns[i], cor.columns[j])


#PCA 取前K大variance的feature
pca = PCA(n_components=5)
Y_pca = pca.fit_transform(x_variance)
print('pca')
print(Y_pca)
print('pca variance')
print(Y_pca.var(axis=0))
#print(pca.explained_variance_ratio_)

