import joblib
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import poisson

# 尝试导入joblib
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib
    
class FootballModelManager:
    def __init__(self, model_dir="./models/"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.xgb_model = None
        self.lgb_model = None
        self.lr_model = None
        self.mlp_model = None
        self.score_predictor = None
        self.scaler = StandardScaler()
        self.label_encoder = {"H": 2, "D": 1, "A": 0}
        self.inverse_label = {2: "主胜", 1: "平局", 0: "客胜"}
        self.training_data_path = os.path.join(model_dir, "training_data.csv")
        self.is_trained = False

    def train(self, X, y, home_goals, away_goals):
        """训练完整模型"""
        # 数据标准化
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # 1. 训练XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.03, 
            random_state=42, use_label_encoder=False, eval_metric='mlogloss'
        )
        self.xgb_model.fit(X, y)

        # 2. 训练LightGBM
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.03, 
            random_state=42, verbose=-1
        )
        self.lgb_model.fit(X, y)

        # 3. 训练逻辑回归
        self.lr_model = LogisticRegression(max_iter=2000, random_state=42)
        self.lr_model.fit(X_scaled, y)

        # 4. 训练MLP神经网络
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=1000, 
            random_state=42, early_stopping=True
        )
        self.mlp_model.fit(X_scaled, y)

        # 训练比分预测模型
        self.score_predictor = {
            "home": xgb.XGBRegressor(n_estimators=300, random_state=42),
            "away": xgb.XGBRegressor(n_estimators=300, random_state=42)
        }
        self.score_predictor["home"].fit(X, home_goals)
        self.score_predictor["away"].fit(X, away_goals)

        # 保存训练数据
        training_df = pd.DataFrame(X)
        training_df["y"] = y
        training_df["home_goals"] = home_goals
        training_df["away_goals"] = away_goals
        training_df.to_csv(self.training_data_path, index=False)

        self.is_trained = True
        self.save()

    def predict(self, X):
        """预测赛果和比分"""
        if not self.is_trained:
            raise Exception("模型未训练，请先训练模型")

        X_scaled = self.scaler.transform(X)

        # 赛果预测 - 软投票
        probas = []
        probas.append(self.xgb_model.predict_proba(X))
        probas.append(self.lgb_model.predict_proba(X))
        probas.append(self.lr_model.predict_proba(X_scaled))
        probas.append(self.mlp_model.predict_proba(X_scaled))
        
        result_proba = np.mean(probas, axis=0)
        pred_result = self.inverse_label[np.argmax(result_proba)]
        result_prob_dict = {self.inverse_label[i]: round(result_proba[0][i], 4) for i in range(3)}

        # 比分预测
        lambda_home = max(0.1, self.score_predictor["home"].predict(X)[0])
        lambda_away = max(0.1, self.score_predictor["away"].predict(X)[0])
        
        score_probs = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                score_probs[i][j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
        most_prob_idx = np.unravel_index(score_probs.argmax(), score_probs.shape)
        most_prob_score = f"{most_prob_idx[0]}-{most_prob_idx[1]}"

        return {
            "result": pred_result,
            "result_probs": result_prob_dict,
            "score": most_prob_score,
            "lambda_home": round(lambda_home, 2),
            "lambda_away": round(lambda_away, 2)
        }

    def online_update(self, X_new, y_new, home_goals_new, away_goals_new):
        """在线更新模型 (增量学习)"""
        # 加载旧训练数据
        if os.path.exists(self.training_data_path):
            old_data = pd.read_csv(self.training_data_path)
            X_old = old_data.drop(["y", "home_goals", "away_goals"], axis=1).values
            y_old = old_data["y"].values
            home_goals_old = old_data["home_goals"].values
            away_goals_old = old_data["away_goals"].values

            # 合并新数据
            X_combined = np.vstack([X_old, X_new])
            y_combined = np.hstack([y_old, y_new])
            home_goals_combined = np.hstack([home_goals_old, home_goals_new])
            away_goals_combined = np.hstack([away_goals_old, away_goals_new])
        else:
            X_combined = X_new
            y_combined = y_new
            home_goals_combined = home_goals_new
            away_goals_combined = away_goals_new

        # 重新训练
        self.train(X_combined, y_combined, home_goals_combined, away_goals_combined)

    def save(self):
        """保存模型到本地"""
        joblib.dump(self.xgb_model, os.path.join(self.model_dir, "xgb_model.pkl"))
        joblib.dump(self.lgb_model, os.path.join(self.model_dir, "lgb_model.pkl"))
        joblib.dump(self.lr_model, os.path.join(self.model_dir, "lr_model.pkl"))
        joblib.dump(self.mlp_model, os.path.join(self.model_dir, "mlp_model.pkl"))
        joblib.dump(self.score_predictor, os.path.join(self.model_dir, "score_predictor.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))
        joblib.dump(self.is_trained, os.path.join(self.model_dir, "is_trained.pkl"))

    def load(self):
        """加载本地模型"""
        try:
            self.xgb_model = joblib.load(os.path.join(self.model_dir, "xgb_model.pkl"))
            self.lgb_model = joblib.load(os.path.join(self.model_dir, "lgb_model.pkl"))
            self.lr_model = joblib.load(os.path.join(self.model_dir, "lr_model.pkl"))
            self.mlp_model = joblib.load(os.path.join(self.model_dir, "mlp_model.pkl"))
            self.score_predictor = joblib.load(os.path.join(self.model_dir, "score_predictor.pkl"))
            self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
            self.is_trained = joblib.load(os.path.join(self.model_dir, "is_trained.pkl"))
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
