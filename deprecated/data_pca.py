import numpy as np
import pandas as pd
import sklearn
import sklearn.decomposition
import sys

print "Reading data..."
is_test = int(sys.argv[2])
data_name = ['train', 'test'][is_test]
rawdata = pd.read_csv('mnist_'+data_name+'.csv')

print "Converting to dataframe..."
X_train = rawdata.ix[:, 1:].as_matrix()
Y_train = rawdata.ix[:, 0].as_matrix()

print "Doing PCA..."
pca_dim = int(sys.argv[1])
svd = sklearn.decomposition.TruncatedSVD(n_components=pca_dim)
X_PCA = svd.fit_transform(X_train)

print "Merging data..."
train_pca = np.column_stack((X_PCA, Y_train))
train_pca_df = pd.DataFrame(train_pca)
train_pca_df.head()
colnames = ['dim'+str(i) for i in xrange(1, pca_dim+1)] + ['label']
train_pca_df.columns = colnames
#train_pca_df.head()

print "Saving data..."
train_pca_df.to_csv(data_name+'_pca.csv', index=False)

print "Done!"