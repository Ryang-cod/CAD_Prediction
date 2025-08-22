from flask import Flask, request, render_template
import joblib
import pandas as pd
import shap

# 加载模型
model = joblib.load("rf_model.pkl")

# 特征名称
feature_names = ['RDW', 'Bun', 'CTI', 'ALP', 'Lymphocytes', 'Chloride',
                 'WBC', 'Los_Icu', 'RBC', 'RR', 'APSIII', 'Sodium',
                 'Platelet', 'Phosphate', 'Age']

# baseline 数据 (a.xlsx 前50行)
baseline = pd.read_excel("a.xlsx").iloc[:10][feature_names]

# 创建 Flask 应用
app = Flask(__name__)

def explain_patient(patient_data: pd.DataFrame):
    """计算预测概率和 shap 力图 (返回HTML字符串)"""
    # 拼接数据（患者 + baseline）
    X_explain = pd.concat([patient_data, baseline], ignore_index=True)

    # 构建 explainer (背景=baseline)
    explainer = shap.Explainer(lambda X: model.predict_proba(X)[:, 1], X_explain.iloc[1:])

    shap_values = explainer(X_explain)

    # 预测概率
    pred_prob = model.predict_proba(patient_data)[:, 1][0]

    # 生成 shap 力图 (返回 html)
    force_plot = shap.force_plot(
        shap_values.base_values[0],
        shap_values.values[0],
        X_explain.iloc[0]
    )
    return pred_prob, force_plot

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    shap_html = None

    if request.method == "POST":
        try:
            # 获取用户输入
            data = [float(request.form[feat]) for feat in feature_names]
            patient_df = pd.DataFrame([data], columns=feature_names)

            # 模型预测
            pred = model.predict(patient_df)[0]
            prediction = "Death" if pred == 1 else "Survive"

            # SHAP 解释
            probability, force_plot = explain_patient(patient_df)

            # 将 shap 力图转成 HTML 字符串
            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

        except Exception as e:
            prediction = f"输入错误: {e}"

    return render_template("index.html",
                           feature_names=feature_names,
                           prediction=prediction,
                           probability=probability,
                           shap_html=shap_html)

if __name__ == "__main__":
    app.run(debug=False)
