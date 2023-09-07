import random
from typing import Optional, Union
import string
import numpy as np
import pandas as pd
import cudf

def generate_data(
    n_samples,
    as_list=False,
    random_state: Optional[Union[int, float, str, bytes, bytearray]] = None,
    sample_length: int = 100,
) -> np.ndarray:
    if random_state is not None:
        random.seed(random_state)
    MAX_LIMIT = 255  # extended ASCII Character set
    str_list = []
    for i in range(n_samples):
        random_string = "category "
        for _ in range(sample_length):
            random_integer = random.randint(1, MAX_LIMIT)
            random_string += chr(random_integer)
            if random_integer < 50:
                random_string += "  "
        str_list += [random_string]
    if as_list is True:
        X = str_list
    else:
        X = np.array(str_list)
    return cudf.DataFrame(cudf.Series.from_pandas(pd.Series(X),nan_as_null=False))

def id_generator(size=12, chars=string.ascii_uppercase + string.digits + ' '):
    return ''.join(random.choice(chars) for _ in range(size))

def generate_cudata(
    n_samples,
    random_state: Optional[Union[int, float, str, bytes, bytearray]] = None,
):
    # start_u = pd.to_datetime('2012-01-01').value//10**9
    # end_u = pd.to_datetime('2023-01-01').value//10**9

#     aa = np.random.randint(18,75,size=(n_samples)).astype(float)
#     mask = np.random.choice([1, 0], aa.shape, p=[.1, .9]).astype(bool)
#     aa[mask] = np.nan
#     bb = np.random.randint(0,200,size=(n_samples)).astype(float)
#     mask = np.random.choice([1, 0], bb.shape, p=[.1, .9]).astype(bool)
#     bb[mask] = np.nan
#     cc = np.random.randint(0,1000,size=(n_samples)).astype(float)
#     mask = np.random.choice([1, 0], cc.shape, p=[.1, .9]).astype(bool)
#     cc[mask] = np.nan
#     ff=pd.to_datetime(np.random.randint(start_u, end_u, n_samples), unit='s').date

#     dd = np.round(np.random.uniform(20, 24,size=(n_samples)), 2)
#     ee = np.round(np.random.uniform(110, 120,size=(n_samples)), 2)

    df = pd.DataFrame()#{
        # 'age': aa,
        # 'user_id': bb,
        # 'profile': cc,
        # 'lon':dd,
        # 'date':ff,
        # 'lat':ee
    # }

    # )

    id_generator()
    df['str0'] = np.array([id_generator(5) for i in range(n_samples)])#.reshape(-1,2)
    df['str1'] = np.array([id_generator(10) for i in range(n_samples)])#.reshape(-1,2)
    
    words = [''.join(random.choice(string.ascii_uppercase + string.digits + ' ') for j in range(20)) for i in range(n_samples)]
    df['str2']=words

    words = [''.join(random.choice(string.ascii_uppercase + string.digits + ' ') for j in range(50)) for i in range(n_samples)]
    df['str3']=words
    
    return cudf.DataFrame(df)
