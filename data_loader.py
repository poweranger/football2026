import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

class FootballDataLoader:
    def __init__(self, api_key):
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {"x-apisports-key": api_key}
        self.cache_dir = "./data_cache/"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _make_request(self, endpoint, params, max_retries=3):
        """带重试的API请求"""
        for i in range(max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    params=params,
                    timeout=15
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if i == max_retries - 1:
                    raise e
                time.sleep(2)
        return None

    def get_teams(self, league_id, season):
        """获取联赛球队列表"""
        data = self._make_request("/teams", {"league": league_id, "season": season})
        if data and "response" in data:
            return {team["team"]["name"]: team["team"]["id"] for team in data["response"]}
        return {}

    def download_historical_data(self, league_id, seasons):
        """下载多个赛季的历史比赛数据+赔率"""
        all_matches = []
        for season in seasons:
            cache_file = f"{self.cache_dir}league_{league_id}_season_{season}.csv"
            if os.path.exists(cache_file):
                season_data = pd.read_csv(cache_file)
                all_matches.append(season_data)
                continue

            # 下载比赛
            fixtures = self._make_request("/fixtures", {"league": league_id, "season": season})
            if not fixtures or "response" not in fixtures:
                continue

            # 处理每场比赛
            for fixture in fixtures["response"]:
                fixture_id = fixture["fixture"]["id"]
                match_data = {
                    "date": fixture["fixture"]["date"],
                    "home_team_id": fixture["teams"]["home"]["id"],
                    "home_team": fixture["teams"]["home"]["name"],
                    "away_team_id": fixture["teams"]["away"]["id"],
                    "away_team": fixture["teams"]["away"]["name"],
                    "home_goals": fixture["goals"]["home"],
                    "away_goals": fixture["goals"]["away"],
                    "result": "H" if fixture["goals"]["home"] > fixture["goals"]["away"] else ("A" if fixture["goals"]["home"] < fixture["goals"]["away"] else "D")
                }

                # 下载赔率 (威廉希尔+立博)
                try:
                    odds = self._make_request("/odds", {"fixture": fixture_id})
                    if odds and "response" in odds and len(odds["response"]) > 0:
                        for bookmaker in odds["response"][0]["bookmakers"]:
                            if bookmaker["name"] in ["William Hill", "Ladbrokes"]:
                                for bet in bookmaker["bets"]:
                                    if bet["name"] == "Match Winner":
                                        match_data[f"{bookmaker['name']}_home"] = bet["values"][0]["odd"]
                                        match_data[f"{bookmaker['name']}_draw"] = bet["values"][1]["odd"]
                                        match_data[f"{bookmaker['name']}_away"] = bet["values"][2]["odd"]
                except:
                    pass

                all_matches.append(match_data)
                time.sleep(0.3)  # 避免API限流

            # 保存缓存
            if all_matches:
                pd.DataFrame(all_matches).to_csv(cache_file, index=False)

        return pd.concat([pd.DataFrame(x) for x in all_matches], ignore_index=True) if all_matches else None

    def get_lineup_influence(self, fixture_id, team_id):
        """计算主力球员缺席影响分数 (0-10, 越高影响越大)"""
        try:
            # 这里简化处理，因为需要真实fixture_id才能获取阵容
            # 实际生产中可传入真实比赛ID
            return 0
        except:
            return 0

    def get_injury_count(self, league_id, team_id):
        """获取伤病球员数量"""
        try:
            injuries = self._make_request("/injuries", {"league": league_id, "team": team_id})
            return len(injuries["response"]) if injuries and "response" in injuries else 0
        except:
            return 0