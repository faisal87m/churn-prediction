# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report, f1_score
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# import mlflow
# import mlflow.sklearn
# from typing import Tuple, Any, Dict

# def train_model(df_features: pd.DataFrame) -> Tuple[Any, float]:
#     """
#     Trains multiple classification models, logs them with MLflow in nested runs,
#     selects the best one based on F1-score, and returns it.
#     """


#     X = df_features.drop(["userId", "is_churned"], axis=1)
#     y = df_features["is_churned"]


#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )


#     # الخطوة 2: قم بتطبيق الموازنة (Undersampling) على مجموعة التدريب فقط
#     print("--- Step 2: Applying Undersampling to the Training set ONLY ---")

#     # ندمج X_train و y_train مؤقتًا لتسهيل عملية أخذ العينات
#     train_data = pd.concat([X_train, y_train], axis=1)

#     df_churned = train_data[train_data['is_churned'] == 1]
#     df_not_churned = train_data[train_data['is_churned'] == 0]

#     # تحديد الحجم الأصغر من بين الفئتين في مجموعة التدريب
#     min_train_size = min(len(df_churned), len(df_not_churned))

#     # أخذ عينة متساوية من كل فئة
#     df_churned_bal = df_churned.sample(n=min_train_size, random_state=42)
#     df_not_churned_bal = df_not_churned.sample(n=min_train_size, random_state=42)

#     # دمج البيانات الموزونة وخلطها
#     balanced_train_df = pd.concat([df_churned_bal, df_not_churned_bal]).sample(frac=1, random_state=42)

#     # فصل البيانات الموزونة مرة أخرى إلى X و y
#     X_train_balanced = balanced_train_df.drop("is_churned", axis=1)
#     y_train_balanced = balanced_train_df["is_churned"]


#     # Handle class imbalance for XGBoost
#     scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

#     # Define models
#     models: Dict[str, Any] = {
#         'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
#         'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
#         'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
#     }

#     # Define preprocessing steps
#     numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     categorical_features = X.select_dtypes(include=['object']).columns.tolist()

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), numeric_features),
#             ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
#         ],
#         remainder="passthrough",
#     )

#     best_model = None
#     best_f1_score = -1.0
#     best_model_name = ""
#     last_run = None

#     # Main MLflow run
#     for model_name, model in models.items():
#         # Use nested runs for each model
#         with mlflow.start_run(run_name=f"Train_{model_name}", nested=True) as run:
#             last_run = run
#             print(f"--- Training {model_name} ---")

#             pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
#             pipeline.fit(X_train_balanced, y_train_balanced)

#             y_pred = pipeline.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             f1_churn = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

#             print(f"Model: {model_name}")
#             print(f"Accuracy: {accuracy:.4f}")
#             print(f"F1-Score (Churn): {f1_churn:.4f}")
#             print(classification_report(y_test, y_pred, zero_division=0))

#             # Log parameters, metrics, and model
#             mlflow.log_param("model_name", model_name)
#             mlflow.log_metric("accuracy", accuracy)
#             mlflow.log_metric("f1_score_churn", f1_churn)
#             mlflow.sklearn.log_model(pipeline, model_name)
#             mlflow.set_tag("mlflow.runName", f"Train_{model_name}")

#             # Check for the best model based on F1 score for the positive class
#             if f1_churn > best_f1_score:
#                 best_f1_score = f1_churn
#                 best_model = pipeline
#                 best_model_name = model_name

#     # Log the best model choice to the parent run
#     if mlflow.active_run() and last_run and mlflow.active_run().info.run_id != last_run.info.run_id:
#          mlflow.set_tag("best_model_name", best_model_name)
#          mlflow.log_metric("best_model_f1_score", best_f1_score)

#     print(f"\nBest model is: {best_model_name} with F1-Score (Churn): {best_f1_score:.4f}")

#     return best_model, best_f1_score

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from typing import Tuple, Any, Dict

# تعديل مسار النظام للسماح بالاستيراد من المجلد الرئيسي
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_model(df_features: pd.DataFrame) -> Tuple[Any, float]:
    """
    Trains and evaluates multiple models using the recommended best practice:
    - Handles class imbalance using model weights to preserve all data.
    - Automatically detects numeric/categorical features.
    - Splits data first, then calculates weights from the training set only.
    - Logs all results and models to MLflow.
    """
    # التحقق من وجود بيانات كافية
    if df_features.shape[0] < 20:
        print("Not enough data to train the model.")
        return None, -1.0

    # فصل الميزات عن الهدف
    X = df_features.drop(["userId", "is_churned"], axis=1)
    y = df_features["is_churned"]

    if y.nunique() < 2:
        print("The target variable has less than 2 unique classes. Cannot train model.")
        return None, -1.0

    # الخطوة 1: قسّم البيانات إلى تدريب واختبار أولاً
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"Data split into {len(X_train)} training samples and {len(X_test)} test samples."
    )
    print(f"Training data distribution:\n{y_train.value_counts(normalize=True)}\n")

    # الخطوة 2: حساب معامل الوزن لـ XGBoost من بيانات التدريب فقط (لمنع تسرب البيانات)
    scale_pos_weight = (y_train == 0).sum() / (
        y_train == 1
    ).sum()  # if (y_train == 1).sum() > 0 else 1
    print(
        f"Using weight-based balancing. Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}"
    )

    # الخطوة 3: تعريف النماذج مع استراتيجيات الموازنة المدمجة
    models: Dict[str, Any] = {
        # class_weight='balanced' يطلب من النموذج موازنة الفئات داخليًا
        "Logistic Regression": LogisticRegression(
            random_state=42, class_weight="balanced", max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_estimators=150,
            max_depth=15,
            min_samples_leaf=2,
        ),
        # نمرر معامل الوزن الذي حسبناه لـ XGBoost
        "XGBoost": XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        ),
    }

    # الخطوة 4: تعريف المعالج المسبق (Preprocessor) الذي يكتشف الميزات تلقائيًا
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"Identified {len(numeric_features)} numeric features.")
    print(
        f"Identified {len(categorical_features)} categorical features: {categorical_features}"
    )

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # الوسيط أكثر قوة ضد القيم المتطرفة
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    best_model = None
    best_f1_score = -1.0
    best_model_name = ""

    # الخطوة 5: حلقة التدريب والتقييم
    for model_name, model in models.items():
        # استخدام nested runs لتنظيم التجارب في MLflow
        with mlflow.start_run(run_name=f"Train_{model_name}", nested=True):
            print(f"--- Training {model_name} ---")

            # إنشاء الـ Pipeline الكامل الذي يدمج المعالجة المسبقة مع النموذج
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("classifier", model)]
            )

            # تدريب النموذج على بيانات التدريب الأصلية الكاملة
            pipeline.fit(X_train, y_train)

            # التقييم على بيانات الاختبار لضمان تقييم صادق
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            f1_churn = report.get("1.0", {}).get(
                "f1-score", 0
            )  # تم التعديل: استخدام '1.0' بدلاً من '1'

            print(f"Model: {model_name}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score (Churn): {f1_churn:.4f}")
            print(classification_report(y_test, y_pred, zero_division=0))

            # تسجيل كل النتائج في MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score_churn", f1_churn)
            mlflow.sklearn.log_model(pipeline, model_name)

            # التحقق من أفضل نموذج بناءً على F1-Score
            if f1_churn > best_f1_score:
                best_f1_score = f1_churn
                best_model = pipeline
                best_model_name = model_name

    # تسجيل ملخص أفضل نموذج في التشغيلة الرئيسية
    # بعد انتهاء التشغيلات المتداخلة، نكون قد عدنا إلى التشغيلة الرئيسية
    # لذا يمكننا التسجيل مباشرة هنا
    if mlflow.active_run() and best_model_name:
        mlflow.set_tag("best_model_name", best_model_name)
        mlflow.log_metric("best_model_f1_score", best_f1_score)

    print(
        f"\nBest model is: {best_model_name} with F1-Score (Churn): {best_f1_score:.4f}"
    )

    return best_model, best_f1_score
