import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.team_stats_cache = {}

    def compute_team_features(self, df, team_id, date, lookback=10):
        """计算球队历史表现特征"""
        cache_key = (team_id, str(date))
        if cache_key in self.team_stats_cache:
            return self.team_stats_cache[cache_key]

        # 确保日期格式一致
        df['date'] = pd.to_datetime(df['date'])
        target_date = pd.to_datetime(date)

        # 筛选该球队在指定日期前的比赛
        team_matches = df[
            ((df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)) &
            (df["date"] < target_date)
        ].tail(lookback)

        if len(team_matches) == 0:
            return [0]*12

        # 计算统计量
        goals_for = []
        goals_against = []
        points = []
        wins = 0

        for _, row in team_matches.iterrows():
            is_home = row["home_team_id"] == team_id
            gf = row["home_goals"] if is_home else row["away_goals"]
            ga = row["away_goals"] if is_home else row["home_goals"]
            
            goals_for.append(gf)
            goals_against.append(ga)
            
            if gf > ga:
                points.append(3)
                wins += 1
            elif gf == ga:
                points.append(1)
            else:
                points.append(0)

        features = [
            np.mean(goals_for), np.std(goals_for),
            np.mean(goals_against), np.std(goals_against),
            np.mean(points), wins/len(team_matches),
            # 主客场拆分
            np.mean([gf for i, gf in enumerate(goals_for) if team_matches.iloc[i]["home_team_id"] == team_id] or [0]),
            np.mean([ga for i, ga in enumerate(goals_against) if team_matches.iloc[i]["home_team_id"] == team_id] or [0]),
            np.mean([gf for i, gf in enumerate(goals_for) if team_matches.iloc[i]["away_team_id"] == team_id] or [0]),
            np.mean([ga for i, ga in enumerate(goals_against) if team_matches.iloc[i]["away_team_id"] == team_id] or [0]),
            # 近期势头 (最近3场平均积分)
            np.mean(points[-3:]) if len(points)>=3 else np.mean(points),
            len(team_matches)
        ]

        self.team_stats_cache[cache_key] = features
        return features

    def create_match_features(self, historical_df, match_row, lineup_influence_home=0, lineup_influence_away=0):
        """创建单场比赛的完整特征向量 (25维)"""
        home_features = self.compute_team_features(historical_df, match_row["home_team_id"], match_row["date"])
        away_features = self.compute_team_features(historical_df, match_row["away_team_id"], match_row["date"])

        # 特征组合: 主队特征 - 客队特征 + 绝对特征 + 阵容影响 + 赔率
        features = [
            # 相对实力 (10维)
            home_features[0] - away_features[0],  # 进球差
            home_features[1] - away_features[1],  # 进球波动差
            home_features[2] - away_features[2],  # 失球差
            home_features[3] - away_features[3],  # 失球波动差
            home_features[4] - away_features[4],  # 积分差
            home_features[5] - away_features[5],  # 胜率差
            home_features[6] - away_features[8],  # 主场进攻 vs 客场进攻
            home_features[7] - away_features[9],  # 主场防守 vs 客场防守
            home_features[10] - away_features[10], # 近期势头差
            # 绝对特征 (4维)
            home_features[6], home_features[7],  # 主队主场攻防
            away_features[8], away_features[9],  # 客队客场攻防
            # 阵容/伤病影响 (2维)
            lineup_influence_home, lineup_influence_away,
            # 赔率特征 (6维) - 使用get防止缺失
            float(match_row.get("William Hill_home", 2.5)),
            float(match_row.get("William Hill_draw", 3.2)),
            float(match_row.get("William Hill_away", 2.5)),
            float(match_row.get("Ladbrokes_home", 2.5)),
            float(match_row.get("Ladbrokes_draw", 3.2)),
            float(match_row.get("Ladbrokes_away", 2.5))
        ]

        return np.array(features).reshape(1, -1)