# import pandas as pd

# def _calculate_features(df_user: pd.DataFrame) -> pd.Series:
#     """
#     تحسب الميزات لمجموعة بيانات مستخدم واحد.
#     هذه هي الدالة التي يتم تطبيقها على كل مستخدم.
#     """
#     # --- حساب الميزات بشكل مباشر ---
#     is_churned = df_user['page'].eq('Cancellation Confirmation').any()
#     num_songs = df_user['page'].eq('NextSong').sum()
#     num_sessions = df_user['sessionId'].nunique()
#     thumbs_up = df_user['page'].eq('Thumbs Up').sum()
#     thumbs_down = df_user['page'].eq('Thumbs Down').sum()
#     downgraded = df_user['page'].eq('Downgrade').any()
#     upgraded = df_user['page'].eq('Upgrade').any()

#     # حساب مدة حياة الحساب بشكل صحيح
#     account_lifetime_days = (df_user['datetime'].max() - df_user['registration'].min()).days

#     avg_songs_per_session = num_songs / num_sessions if num_sessions > 0 else 0.0
#     level_changes = df_user['level'].nunique()

#     # --- معلومات ثابتة للمستخدم ---
#     gender = df_user['gender'].iloc[0]
#     location = df_user['location'].iloc[0]

#     # إنشاء Series بالنتائج
#     return pd.Series({
#         'is_churned': int(is_churned),
#         'num_songs': num_songs,
#         'num_sessions': num_sessions,
#         'thumbs_up': thumbs_up,
#         'thumbs_down': thumbs_down,
#         'downgraded': int(downgraded),
#         'upgraded': int(upgraded),
#         'account_lifetime_days': account_lifetime_days,
#         'avg_songs_per_session': avg_songs_per_session,
#         'level_changes': level_changes,
#         'gender': gender,
#         'location': location
#     })
# def extract_user_features(df_user):
#     df_user = df_user.copy()

#     # تحويل الوقت
#     df_user['datetime'] = pd.to_datetime(df_user['ts'], unit='ms')

#     # التصفية حسب الأحداث
#     is_churned = df_user['page'].isin(['Cancellation Confirmation']).any()
#     num_songs = df_user[df_user['page'] == 'NextSong'].shape[0]
#     num_sessions = df_user['sessionId'].nunique()
#     thumbs_up = df_user[df_user['page'] == 'Thumbs Up'].shape[0]
#     thumbs_down = df_user[df_user['page'] == 'Thumbs Down'].shape[0]
#     downgraded = df_user['page'].isin(['Downgrade']).any()
#     upgraded = df_user['page'].isin(['Upgrade']).any()
#     account_lifetime_days = (df_user['datetime'].max() - df_user['datetime'].min()).days
#     avg_songs_per_session = num_songs / num_sessions if num_sessions else 0
#     level_changes = df_user['level'].nunique()

#     # معلومات ثابتة للمستخدم
#     user_id = df_user['userId'].iloc[0]
#     gender = df_user['gender'].iloc[0] if pd.notna(df_user['gender'].iloc[0]) else 'Unknown'
#     location = df_user['location'].iloc[0] if pd.notna(df_user['location'].iloc[0]) else 'Unknown'

#     return {
#         'userId': user_id,
#         'is_churned': int(is_churned),
#         'num_songs': num_songs,
#         'num_sessions': num_sessions,
#         'thumbs_up': thumbs_up,
#         'thumbs_down': thumbs_down,
#         'downgraded': int(downgraded),
#         'upgraded': int(upgraded),
#         'account_lifetime_days': account_lifetime_days,
#         'avg_songs_per_session': avg_songs_per_session,
#         'level_changes': level_changes,
#         'gender': gender,
#         'location': location
#     }
# def create_features(df: pd.DataFrame, balance: bool = False) -> pd.DataFrame:
#     """
#     Processes the raw event data to create a feature set for all users.
#     Includes cleaning for invalid user IDs and an optional balancing step.
#     """
#     df = df.copy()

#     # --- (جديد) تنظيف معرفات المستخدمين غير الصالحة ---
#     # التأكد من أن userId هو نصي قبل التنظيف
#     df['userId'] = df['userId'].astype(str)

#     initial_rows = len(df)
#     print(f"Initial number of rows: {initial_rows}")

#     # إزالة الصفوف التي يكون فيها userId فارغًا أو يحتوي على مسافات فقط
#     df = df[df['userId'].str.strip() != '']
#     # إزالة الصفوف التي يكون فيها userId هو '0'
#     df = df[df['userId'] != '0']

#     cleaned_rows = len(df)
#     print(f"Number of rows after cleaning invalid userIDs: {cleaned_rows} ({initial_rows - cleaned_rows} rows removed).")

#     # --- 1. تجهيز البيانات الأساسي ---
#     df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
#     df["registration"] = pd.to_datetime(df["registration"], unit="ms")

#     # --- 2. تطبيق حساب الميزات لكل مستخدم ---
#     # هذا السطر يؤدي نفس وظيفة حلقة for التي عرضتها ولكن بكفاءة أكبر
#     #features_df = df.groupby('userId').apply(_calculate_features)
#     #print('Features created with shape:', features_df['is_churned'].value_counts())  # لمعرفة كم مستخدم قام بالإلغاء
#     user_features = []
#     df_clean = df.copy()
#     for user_id, df_user in df_clean.groupby('userId'):
#         features = extract_user_features(df_user)
#         user_features.append(features)

#     # تحويل النتائج إلى DataFrame
#     df_features = pd.DataFrame(user_features)
#     print('Features created with shape:', df_features['is_churned'].value_counts())  # لمعرفة كم مستخدم قام بالإلغاء
#     # --- 3. تنظيف البيانات بعد التجميع ---
#     features_df = df_features.copy()
#     features_df = features_df.reset_index()
#     features_df["gender"] = features_df["gender"].fillna("Unknown")
#     features_df["location"] = features_df["location"].fillna("Unknown")

#     # # --- 4. (اختياري) موازنة البيانات بشكل ديناميكي ---
#     # if balance:
#     #     print("Balancing data by undersampling the majority class...")
#     #     class_counts = features_df['is_churned'].value_counts()

#     #     if len(class_counts) == 2 and class_counts.iloc[0] != class_counts.iloc[1]:
#     #         minority_class_label = class_counts.idxmin()
#     #         majority_class_label = class_counts.idxmax()
#     #         minority_count = class_counts.min()

#     #         df_minority = features_df[features_df['is_churned'] == minority_class_label]
#     #         df_majority = features_df[features_df['is_churned'] == majority_class_label]

#     #         df_majority_undersampled = df_majority.sample(n=minority_count, random_state=42)

#     #         features_df = pd.concat([df_minority, df_majority_undersampled])
#     #         features_df = features_df.sample(frac=1, random_state=42).reset_index(drop=True)
#     #         print(f"Data balanced. New shape: {features_df.shape}")
#     #     else:
#     #         print("Balancing not needed or not possible.")

#     # --- 5. إعادة ترتيب الأعمدة للهيكل النهائي ---
#     column_order = [
#         "userId", "is_churned", "num_songs", "num_sessions", "thumbs_up",
#         "thumbs_down", "downgraded", "upgraded", "account_lifetime_days",
#         "avg_songs_per_session", "level_changes", "gender", "location"
#     ]
#     final_cols = [col for col in column_order if col in features_df.columns]

#     return features_df[final_cols]

import pandas as pd
import numpy as np

# ===============================
# Feature Engineering Functions
# ===============================
# This file contains all logic for extracting features from user event data for churn prediction.
# All main steps and calculations are explained in English below.


def _calculate_features(df_user: pd.DataFrame) -> pd.Series:
    """
    Calculate features for a single user's event data after applying a cutoff time to prevent data leakage.
    This function is applied to each user's data separately.
    """
    # --- Golden rule: Define cutoff time and filter data to prevent leakage ---
    churn_event = df_user[df_user["page"] == "Cancellation Confirmation"]

    if not churn_event.empty:
        # If user churned, cutoff time is the first churn event
        cutoff_time = churn_event["datetime"].min()
        # Only keep data up to and including the cutoff time
        df_safe = df_user[df_user["datetime"] <= cutoff_time]
        is_churned = 1
    else:
        # If user did not churn, all data is safe to use
        df_safe = df_user
        is_churned = 0

    # If there is no safe data (rare case), return empty Series to be ignored later
    if df_safe.empty:
        return pd.Series(dtype="object")

    # --- Calculate core features from safe data only ---
    num_songs = df_safe["page"].eq("NextSong").sum()  # Total number of songs played
    num_sessions = df_safe["sessionId"].nunique()  # Number of unique sessions
    thumbs_up = df_safe["page"].eq("Thumbs Up").sum()  # Number of thumbs up
    thumbs_down = df_safe["page"].eq("Thumbs Down").sum()  # Number of thumbs down
    downgraded = df_safe["page"].eq("Downgrade").any()  # Did user downgrade?
    upgraded = df_safe["page"].eq("Upgrade").any()  # Did user upgrade?
    level_changes = df_safe["level"].nunique()  # Number of unique subscription levels

    account_lifetime_days = (
        df_safe["datetime"].max() - df_safe["registration"].min()
    ).days  # Account age
    avg_songs_per_session = (
        num_songs / num_sessions if num_sessions > 0 else 0.0
    )  # Average songs per session

    # --- New and strong features ---
    # 1. Thumbs up ratio (strong satisfaction indicator)
    total_thumbs = thumbs_up + thumbs_down
    thumbs_up_ratio = thumbs_up / total_thumbs if total_thumbs > 0 else 0.0

    # 2. Average session duration in minutes (engagement indicator)
    if num_sessions > 0:
        session_durations = df_safe.groupby("sessionId")["datetime"].agg(
            lambda x: x.max() - x.min()
        )
        avg_session_duration_min = session_durations.mean().total_seconds() / 60
    else:
        avg_session_duration_min = 0.0

    # 3. Activity in last 7 days (recency indicator)
    last_event_time = df_safe["datetime"].max()
    seven_days_prior = last_event_time - pd.Timedelta(days=7)
    songs_last_7_days = df_safe[
        (df_safe["page"] == "NextSong") & (df_safe["datetime"] >= seven_days_prior)
    ].shape[0]

    # 4. Account tenure at first downgrade (behavioral indicator)
    downgrade_events = df_safe[df_safe["page"] == "Downgrade"]
    if not downgrade_events.empty:
        first_downgrade_time = downgrade_events["datetime"].min()
        tenure_at_downgrade = (
            first_downgrade_time - df_safe["registration"].min()
        ).days
    else:
        # Use -1 to indicate event did not occur
        tenure_at_downgrade = -1

    # 5. Usage density (songs per day)
    songs_per_day = (
        num_songs / account_lifetime_days if account_lifetime_days > 0 else 0.0
    )

    # --- Static user information ---
    gender = df_safe["gender"].iloc[0]
    location = df_safe["location"].iloc[0]

    # Return all features as a Series
    return pd.Series(
        {
            # Target
            "is_churned": is_churned,
            # Core features
            "num_songs": num_songs,
            "num_sessions": num_sessions,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "downgraded": int(downgraded),
            "upgraded": int(upgraded),
            "account_lifetime_days": account_lifetime_days,
            "avg_songs_per_session": avg_songs_per_session,
            "level_changes": level_changes,
            # Demographic features
            "gender": gender,
            "location": location,
            # New features
            "thumbs_up_ratio": thumbs_up_ratio,
            "avg_session_duration_min": avg_session_duration_min,
            "songs_last_7_days": songs_last_7_days,
            "tenure_at_downgrade": tenure_at_downgrade,
            "songs_per_day": songs_per_day,
        }
    )


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes raw event data to create a rich and safe feature set for all users.
    This function prevents data leakage by respecting a cutoff time for churned users.
    Steps:
      1. Clean and prepare the data (remove invalid userIds, convert time columns).
      2. Apply feature calculation for each user using _calculate_features.
      3. Clean and finalize the resulting features DataFrame.
    """
    df = df.copy()

    # --- 1. Initial cleaning and preparation ---
    df["userId"] = (
        df["userId"].astype(str).str.strip()
    )  # Ensure userId is string and strip whitespace
    df = df.dropna(subset=["userId"])  # Drop rows with missing userId
    df = df[df["userId"] != ""]  # Drop rows with empty userId
    df = df[df["userId"] != "0"]  # Drop rows where userId is '0'

    # Convert time columns once to avoid repeated conversions
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    df["registration"] = pd.to_datetime(df["registration"], unit="ms")

    # --- 2. Apply feature calculation for each user ---
    # .apply is an efficient way to perform this type of aggregation
    features_df = df.groupby("userId").apply(_calculate_features)

    # Remove any users where features could not be calculated (rare case)
    features_df.dropna(how="all", inplace=True)

    # --- 3. Post-aggregation cleaning ---
    features_df = features_df.reset_index()
    features_df["gender"] = features_df["gender"].fillna("Unknown")
    features_df["location"] = features_df["location"].fillna("Unknown")

    # Ensure all numeric columns are of numeric type
    numeric_cols = features_df.select_dtypes(include=np.number).columns
    features_df[numeric_cols] = features_df[numeric_cols].astype("float64")

    print(f"Features created for {len(features_df)} users.")

    return features_df
