import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# 导入自定义模块
from data_loader import FootballDataLoader
from feature_engineer import FeatureEngineer
from model_manager import FootballModelManager

# ===================== 页面配置 =====================
st.set_page_config(page_title="⚽ 专业足球预测系统", layout="wide", page_icon="⚽")
st.title("⚽ 专业足球比赛预测系统 (企业优化版)")

# ===================== 全局状态管理 =====================
if "model_manager" not in st.session_state:
    st.session_state.model_manager = FootballModelManager()
    st.session_state.data_loader = None
    st.session_state.feature_engineer = FeatureEngineer()
    st.session_state.historical_data = None

# ===================== 侧边栏配置 =====================
st.sidebar.header("🔧 系统配置")
api_key = st.sidebar.text_input("API-Football Key", type="password", value=st.secrets.get("API_KEY", ""))
selected_league = st.sidebar.selectbox("选择联赛", ["英超", "德甲", "西甲", "意甲", "法甲"])
league_config = {
    "英超": {"id": 39, "seasons": [2020, 2021, 2022, 2023, 2024]},
    "德甲": {"id": 78, "seasons": [2020, 2021, 2022, 2023, 2024]},
    "西甲": {"id": 140, "seasons": [2020, 2021, 2022, 2023, 2024]},
    "意甲": {"id": 135, "seasons": [2020, 2021, 2022, 2023, 2024]},
    "法甲": {"id": 61, "seasons": [2020, 2021, 2022, 2023, 2024]}
}

# 初始化数据加载器
if api_key and (not st.session_state.data_loader or st.session_state.data_loader.headers["x-apisports-key"] != api_key):
    st.session_state.data_loader = FootballDataLoader(api_key)

# 页面导航
page = st.sidebar.radio("📱 功能模块", ["单场预测", "批量预测", "模型管理", "数据管理", "阵容分析"])

# ===================== 页面1: 单场预测 =====================
if page == "单场预测":
    st.header("🎯 单场比赛智能预测")
    
    if not st.session_state.data_loader:
        st.warning("请先在左侧配置 API Key")
    else:
        # 检查模型是否已训练
        if not st.session_state.model_manager.ensemble_model:
            if not st.session_state.model_manager.load():
                st.info("模型未训练，请先前往「模型管理」页面训练模型")
        
        # 获取球队列表
        with st.spinner("加载球队列表..."):
            teams = st.session_state.data_loader.get_teams(
                league_config[selected_league]["id"],
                league_config[selected_league]["seasons"][-1]
            )
        
        if teams:
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox("选择主队", sorted(teams.keys()))
            with col2:
                away_team = st.selectbox("选择客队", sorted(teams.keys()), index=1)
            
            if st.button("开始预测", type="primary"):
                if home_team == away_team:
                    st.error("主队和客队不能相同")
                else:
                    with st.spinner("正在获取实时数据并执行预测..."):
                        try:
                            # 准备模拟比赛行 (使用当前日期)
                            match_row = {
                                "date": datetime.now().isoformat(),
                                "home_team_id": teams[home_team],
                                "away_team_id": teams[away_team],
                                "home_team": home_team,
                                "away_team": away_team
                            }
                            
                            # 获取阵容影响和伤病
                            lineup_home = st.session_state.data_loader.get_lineup_influence(
                                0, teams[home_team]
                            )  # 0为模拟fixture_id
                            lineup_away = st.session_state.data_loader.get_lineup_influence(
                                0, teams[away_team]
                            )
                            injury_home = st.session_state.data_loader.get_injury_count(
                                league_config[selected_league]["id"], teams[home_team]
                            )
                            injury_away = st.session_state.data_loader.get_injury_count(
                                league_config[selected_league]["id"], teams[away_team]
                            )
                            
                            # 加载历史数据
                            if st.session_state.historical_data is None:
                                st.session_state.historical_data = st.session_state.data_loader.download_historical_data(
                                    league_config[selected_league]["id"],
                                    league_config[selected_league]["seasons"]
                                )
                            
                            # 特征工程
                            X = st.session_state.feature_engineer.create_match_features(
                                st.session_state.historical_data,
                                match_row,
                                lineup_influence_home=lineup_home + injury_home,
                                lineup_influence_away=lineup_away + injury_away
                            )
                            
                            # 预测
                            result = st.session_state.model_manager.predict(X)
                            
                            # 展示结果
                            st.success("✅ 预测完成！")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("预测赛果", result["result"])
                            with col2:
                                st.metric("最可能比分", result["score"])
                            with col3:
                                st.metric("期望进球", f"{result['lambda_home']} - {result['lambda_away']}")
                            
                            # 概率分布
                            st.subheader("赛果概率分布")
                            prob_df = pd.DataFrame.from_dict(result["result_probs"], orient="index", columns=["概率"])
                            st.bar_chart(prob_df)
                            
                            # 阵容影响
                            st.subheader("阵容与伤病影响")
                            st.write(f"主队影响分数: {lineup_home + injury_home}/10")
                            st.write(f"客队影响分数: {lineup_away + injury_away}/10")
                            
                        except Exception as e:
                            st.error(f"预测失败: {str(e)}")

# ===================== 页面2: 批量预测 =====================
elif page == "批量预测":
    st.header("📊 批量预测")
    
    # 下载模板
    template_df = pd.DataFrame(columns=["主队", "客队"])
    st.download_button("下载预测模板", template_df.to_csv(index=False).encode('utf-8-sig'), "预测模板.csv")
    
    # 上传文件
    uploaded_file = st.file_uploader("上传对阵列表", type="csv")
    if uploaded_file and st.session_state.model_manager.ensemble_model:
        input_df = pd.read_csv(uploaded_file)
        st.dataframe(input_df)
        
        if st.button("开始批量预测"):
            with st.spinner("批量预测中..."):
                results = []
                teams = st.session_state.data_loader.get_teams(
                    league_config[selected_league]["id"],
                    league_config[selected_league]["seasons"][-1]
                )
                
                for _, row in input_df.iterrows():
                    try:
                        match_row = {
                            "date": datetime.now().isoformat(),
                            "home_team_id": teams[row["主队"]],
                            "away_team_id": teams[row["客队"]]
                        }
                        X = st.session_state.feature_engineer.create_match_features(
                            st.session_state.historical_data, match_row
                        )
                        res = st.session_state.model_manager.predict(X)
                        results.append({
                            "主队": row["主队"], "客队": row["客队"],
                            "预测赛果": res["result"], "最可能比分": res["score"],
                            "主胜概率": res["result_probs"]["主胜"],
                            "平局概率": res["result_probs"]["平局"],
                            "客胜概率": res["result_probs"]["客胜"]
                        })
                    except:
                        results.append({"主队": row["主队"], "客队": row["客队"], "预测赛果": "失败"})
                
                result_df = pd.DataFrame(results)
                st.dataframe(result_df)
                st.download_button("下载预测结果", result_df.to_csv(index=False).encode('utf-8-sig'), "预测结果.csv")

# ===================== 页面3: 模型管理 =====================
elif page == "模型管理":
    st.header("🤖 模型训练与管理")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("训练/重新训练模型", type="primary"):
            if not st.session_state.data_loader:
                st.error("请先配置 API Key")
            else:
                with st.spinner("正在下载历史数据并训练模型 (约5-10分钟)..."):
                    # 下载历史数据
                    st.session_state.historical_data = st.session_state.data_loader.download_historical_data(
                        league_config[selected_league]["id"],
                        league_config[selected_league]["seasons"]
                    )
                    
                    if st.session_state.historical_data is not None:
                        # 构建训练集
                        X_list = []
                        y_list = []
                        home_goals_list = []
                        away_goals_list = []
                        
                        for _, row in st.session_state.historical_data.iterrows():
                            try:
                                X = st.session_state.feature_engineer.create_match_features(
                                    st.session_state.historical_data, row
                                )
                                X_list.append(X.flatten())
                                y_list.append(st.session_state.model_manager.label_encoder[row["result"]])
                                home_goals_list.append(row["home_goals"])
                                away_goals_list.append(row["away_goals"])
                            except:
                                continue
                        
                        X_train = np.array(X_list)
                        y_train = np.array(y_list)
                        
                        # 训练模型
                        st.session_state.model_manager.train(
                            X_train, y_train,
                            np.array(home_goals_list),
                            np.array(away_goals_list)
                        )
                        st.success("✅ 模型训练完成并已保存！")
    
    with col2:
        if st.button("加载本地模型"):
            if st.session_state.model_manager.load():
                st.success("✅ 模型加载成功！")
            else:
                st.error("❌ 未找到本地模型文件")
    
    # 在线更新
    st.subheader("🔄 在线模型更新")
    st.write("输入比赛真实结果以更新模型:")
    update_col1, update_col2, update_col3 = st.columns(3)
    with update_col1:
        update_home = st.text_input("主队")
        update_home_goals = st.number_input("主队进球", min_value=0, step=1)
    with update_col2:
        update_away = st.text_input("客队")
        update_away_goals = st.number_input("客队进球", min_value=0, step=1)
    with update_col3:
        update_date = st.date_input("比赛日期")
    
    if st.button("更新模型"):
        if update_home and update_away:
            teams = st.session_state.data_loader.get_teams(
                league_config[selected_league]["id"],
                league_config[selected_league]["seasons"][-1]
            )
            match_row = {
                "date": update_date.isoformat(),
                "home_team_id": teams[update_home],
                "away_team_id": teams[update_away]
            }
            X = st.session_state.feature_engineer.create_match_features(
                st.session_state.historical_data, match_row
            )
            result = "H" if update_home_goals > update_away_goals else ("A" if update_home_goals < update_away_goals else "D")
            y = st.session_state.model_manager.label_encoder[result]
            
            st.session_state.model_manager.online_update(
                X, np.array([y]),
                np.array([update_home_goals]),
                np.array([update_away_goals])
            )
            st.success("✅ 模型在线更新完成！")

# ===================== 页面4: 数据管理 =====================
elif page == "数据管理":
    st.header("💾 数据管理")
    
    if st.session_state.data_loader:
        if st.button("刷新历史数据缓存"):
            with st.spinner("正在刷新数据..."):
                st.session_state.historical_data = st.session_state.data_loader.download_historical_data(
                    league_config[selected_league]["id"],
                    league_config[selected_league]["seasons"]
                )
                st.success("✅ 数据刷新完成！")
        
        if st.session_state.historical_data is not None:
            st.subheader("历史数据预览")
            st.dataframe(st.session_state.historical_data.tail(20))
            st.write(f"总数据量: {len(st.session_state.historical_data)} 场比赛")

# ===================== 页面5: 阵容分析 =====================
elif page == "阵容分析":
    st.header("👥 阵容深度分析")
    
    if st.session_state.data_loader:
        teams = st.session_state.data_loader.get_teams(
            league_config[selected_league]["id"],
            league_config[selected_league]["seasons"][-1]
        )
        team_name = st.selectbox("选择球队", sorted(teams.keys()))
        
        if st.button("分析阵容"):
            with st.spinner("正在获取阵容数据..."):
                injury_count = st.session_state.data_loader.get_injury_count(
                    league_config[selected_league]["id"], teams[team_name]
                )
                st.metric("当前伤病球员数", injury_count)
                st.info("阵容分析功能需要真实 fixture_id 才能获取完整阵容，此处展示伤病统计")

# ===================== 页脚 =====================
st.sidebar.markdown("---")
st.sidebar.info("""
⚠️ **免责声明**  
本系统仅供技术学习和学术研究使用，预测结果不构成任何投注建议。
""")