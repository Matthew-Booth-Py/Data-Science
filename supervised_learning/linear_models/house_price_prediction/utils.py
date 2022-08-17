import pandas as pd 

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as P
from sklearn.preprocessing import StandardScaler 

def preprocessing(df): 

    int_cols = df.select_dtypes(['int', 'float'])
    object_cols  = df.select_dtypes('object').columns.tolist()    
    
    try: 
        int_cols = int_cols.drop(columns=['SalePrice']).columns.tolist()
    except: 
        int_cols = int_cols.columns.tolist()
    

    df[int_cols] = df[int_cols].fillna(0)
    df[object_cols] = df[object_cols].fillna('NaN')

    # Numerical
    num_pipeline = P([
                    ('std_scaler', StandardScaler())
                        ])
    # Categoriacal 
    categorical_pipeline = P(steps=[
                                ('impute', SimpleImputer(strategy='most_frequent')),
                                # ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
                                ])

    full_processor = ColumnTransformer(transformers=[
                        ('number', num_pipeline, int_cols),
                        # ('category', categorical_pipeline,r object_cols)
                        ])
    
    data_pipe = P([
                    ('full_processor', full_processor),
                    # ('model', LinearRegression())
                    ])

    return data_pipe