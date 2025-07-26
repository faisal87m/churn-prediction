# -*- coding: utf-8 -*-
"""
ملف الإعدادات المركزي لمشروع التنبؤ بانسحاب العملاء.

أفضل الممارسات المتبعة:
- **مركزي وموحد**: جميع الإعدادات في مكان واحد.
- **منظم**: الإعدادات مقسمة إلى أقسام منطقية
(paths, mlflow, data, training, monitoring, retraining).
- **سهل الاستيراد**: يمكن استيراد قاموس `config` بسهولة في أي مكان بالمشروع.
"""

# عدد الأيام بين كل إعادة تدريب تلقائية (للاستخدام المستقبلي مع أنظمة الجدولة)
retrain_interval_days = 7  # يمكن تعديله بسهولة لاحقًا

config = {
    # 1. إعدادات المسارات والبيانات
    "paths": {
        "data_directory": "data/",  # المجلد الذي يحتوي على ملفات البيانات
        "training_data_pattern": "customer_churn_*.json",  # نمط للبحث عن ملفات التدريب
        "model_path": "models/churn_model.joblib",  # مسار لحفظ النموذج
    },
    # إعدادات خاصة بتحديث البيانات
    "data_update": {
        "enabled": False,  # إذا كانت True، سيتم تشغيل سكربت تحديث البيانات
        "script_name": "update_data.py",  # اسم السكربت الذي سيتم تشغيله
    },
    # 2. إعدادات MLflow
    "mlflow": {
        "experiment_name": "Churn_Prediction_Unified_Pipeline",
        "model_artifact_path": "churn_model",
        "tracking_uri": "./mlruns",  # Store MLflow runs in the local 'mlruns' directory
    },
    # 3. إعدادات هندسة الميزات والبيانات
    "data": {
        "target_column": "is_churned",
        "user_id_column": "userId",
    },
    # 4. إعدادات تدريب النموذج
    "training": {
        "test_size": 0.2,
        "random_state": 42,
        "stratify_column": "is_churned",
        # المعلمات الخاصة بكل نموذج
        "models": {
            "logistic_regression": {"class_weight": "balanced", "max_iter": 1000},
            "random_forest": {
                "class_weight": "balanced",
                "n_estimators": 150,
                "max_depth": 15,
                "min_samples_leaf": 2,
            },
            "xgboost": {"use_label_encoder": False, "eval_metric": "logloss"},
        },
    },
    # 5. إعدادات مراقبة النموذج (Drift)
    "monitoring": {
        "data_drift": {
            # مستوى الأهمية الإحصائية (alpha) لاختبار KS
            "alpha": 0.05
        },
        "concept_drift": {
            # عتبة انخفاض F1-score التي تعتبر انحرافًا في المفهوم
            "f1_score_threshold_ratio": 0.8
        },
    },
}
