import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
os.environ["KERAS_BACKEND"] = "tensorflow"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------------------------------------------------------- #
BASE_DATA_DIR = 'DATA/Banawre'
RESULTS_DIR   = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

ALLOWED_SCENARIOS = {
    'ID5':  [('Melon', 'Crave')],
    'ID9':  [('Melon', 'Crave')],
    'ID10': [('Carrot','Crave'), ('Nectarine','Crave'),
             ('Nectarine','Use'), ('Carrot','Use')],
    'ID11': [('Carrot','Crave'), ('Nectarine','Crave'), ('Almond','Crave'),
             ('Carrot','Use'), ('Nectarine','Use'), ('Almond','Use')],
    'ID12': [('Melon','Crave'), ('Nectarine','Crave'),
             ('Melon','Use'), ('Nectarine','Use'), ('GHB','Use')],
    'ID13': [('Nectarine','Use'), ('Carrot','Use'), ('Almond','Use')],
    'ID14': [('Carrot','Crave'), ('Carrot','Use')],
    'ID15': [('Carrot','Crave'), ('Carrot','Use')],
    'ID18': [('Carrot','Use'), ('Carrot','Crave')],
    'ID19': [('Melon','Crave'), ('Almond','Crave'),
             ('Melon','Use'), ('Almond','Use')],
    'ID20': [('Melon','Use'), ('Nectarine','Use'),
             ('Melon','Crave'), ('Nectarine','Crave')],
    'ID21': [('Nectarine','Use'),
             ('Melon','Crave'), ('Nectarine','Crave')],
    'ID25': [('Almond','Crave'), ('Carrot', 'Crave'), 
             ('Almond','Use')],
    'ID26': [('Carrot','Use')],
    'ID27': [('Melon','Use'), ('Nectarine','Use'),
             ('Melon','Crave'), ('Nectarine','Crave')],
    'ID28': [('Coffee','Use'), ('Almond','Use')]      
}

WINDOW_HOURS   = 1            # before/after event
RESAMPLE_MIN   = '4min'       # 30 points/hour
FEATURE_POINTS = 30

# ----------------------------------------------------------------------------- #
def _safe_div(num, denom):
    return num / denom if denom else 0.0


# ----------------------------------------------------------------------------- #
# Data loaders
# ----------------------------------------------------------------------------- #
def load_signal_data(user_dir):
    """
    Read '<numeric>.csv' inside user_dir and return two pre‑processed
    pandas.Series indexed by 1‑minute timestamps:
        hr_df['value']  and  st_df['value']
    """
    uid = os.path.basename(user_dir)
    numeric = uid.split('ID')[-1]
    path = os.path.join(user_dir, f"{numeric}.csv")
    data = pd.read_csv(path)

    hr_df = data.loc[data['data_type'] == 'hr',    ['time', 'value']].copy()
    st_df = data.loc[data['data_type'] == 'steps', ['time', 'value']].copy()

    hr_df['value'] = pd.to_numeric(hr_df['value'], errors='coerce')
    st_df['value'] = pd.to_numeric(st_df['value'], errors='coerce')

    hr_df['time'] = pd.to_datetime(hr_df['time']).dt.tz_localize(None)
    st_df['time'] = pd.to_datetime(st_df['time']).dt.tz_localize(None)
    hr_df.sort_values('time', inplace=True)
    st_df.sort_values('time', inplace=True)

    # aggregate duplicate timestamps
    hr_df = hr_df.groupby('time', as_index=False)['value'].mean()
    st_df = st_df.groupby('time', as_index=False)['value'].mean()

    hr_df['value'] = hr_df['value'].ffill()
    st_df['value'] = st_df['value'].ffill()

    hr_df.set_index('time', inplace=True)
    st_df.set_index('time', inplace=True)

    idx = pd.date_range(start=hr_df.index.min(),
                        end=hr_df.index.max(), freq='min')
    hr_df = hr_df.reindex(idx, method='ffill')
    st_df = st_df.reindex(idx, method='ffill').fillna(0)

    return hr_df, st_df

def load_label_data(user_dir, fruit, scenario):
    uid   = os.path.basename(user_dir)
    fname = f"{uid}_{scenario}.csv"
    path  = os.path.join(user_dir, fname)

    # 1) file doesn’t exist → empty DF
    if not os.path.exists(path):
        return pd.DataFrame()

    # 2) file is literally zero bytes → empty DF
    if os.path.getsize(path) == 0:
        return pd.DataFrame()

    # 3) catch “no columns” error
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    # 4) got a DF with columns but no rows → empty DF
    if df.empty:
        return pd.DataFrame()

    # now parse your timestamp column
    if scenario == 'Use':
        raw = df['hawaii_use_time'].astype(str).str.split('.', n=1).str[0]
    else:
        raw = df['hawaii_createdat_time'].astype(str).str.split('.', n=1).str[0]

    df['hawaii_createdat_time'] = (
        pd.to_datetime(raw, errors='coerce')
          .dt.tz_localize(None)
    )

    df = df.dropna(subset=['hawaii_createdat_time'])

    # if you’re filtering by fruit
    if scenario != 'None':
        df = df[df['substance_fruit_label'] == fruit]

    if 'hawaii_createdat_time' in df.columns:
        df = df.sort_values('hawaii_createdat_time').reset_index(drop=True)
    return df


def process_label_window(df_label, hr_df, st_df, val):
    '''
    Build fixed-length HR/steps sequences around each label timestamp.
    Returns a DataFrame of scaled windows with a binary state label.
    '''
    scaler  = StandardScaler()
    records = []
    half    = pd.Timedelta(hours=WINDOW_HOURS)

    # For each label timestamp, extract a ±1h HR/steps window and resample to fixed length.
    for _, row in df_label.iterrows():
        t0      = row['hawaii_createdat_time']
        hr_win  = hr_df .loc[t0 - half : t0 + half]['value']
        st_win  = st_df .loc[t0 - half : t0 + half]['value']

        # require raw ≥ FEATURE_POINTS minutes
        if len(hr_win) < FEATURE_POINTS or len(st_win) < FEATURE_POINTS:
            continue

        # resample into exactly FEATURE_POINTS bins
        hr_means = hr_win.resample(RESAMPLE_MIN).mean().iloc[:FEATURE_POINTS]
        st_means = st_win.resample(RESAMPLE_MIN).mean().iloc[:FEATURE_POINTS]

        # STRICT: must have exactly FEATURE_POINTS bins AND no NaNs
        if (len(hr_means) != FEATURE_POINTS or hr_means.isna().any() or
            len(st_means) != FEATURE_POINTS or st_means.isna().any()):
            continue

        # scale
        hr_scaled = scaler.fit_transform(hr_means.values.reshape(-1,1)).flatten().tolist()
        st_scaled = scaler.fit_transform(st_means.values.reshape(-1,1)).flatten().tolist()

        records.append({
            # Preserve source label time so downstream split/date checks can run on windowed frames.
            'hawaii_createdat_time': t0,
            'hr_seq':    hr_scaled,
            'st_seq':    st_scaled,
            'state_val': val, 
        })

    return pd.DataFrame(records)

def generate_embeddings(df_feat, enc_hr, enc_st):
    """Use pretrained encoders to obtain concatenated embeddings."""
    hr_arr = np.stack(df_feat['hr_seq'].values)[:, :, None]
    st_arr = np.stack(df_feat['st_seq'].values)[:, :, None]
    hr_emb = enc_hr.predict(hr_arr, verbose=0)
    st_emb = enc_st.predict(st_arr, verbose=0)
    return np.concatenate([hr_emb, st_emb], axis=1)
