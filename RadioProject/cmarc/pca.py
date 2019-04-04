from sklearn import decomposition
import common.io_operations as io
import pandas as pd

df: pd.DataFrame = io.load_data()
pca = decomposition.PCA(n_components=3)
pca.fit(df)
