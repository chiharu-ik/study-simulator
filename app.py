import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

SUBJECTS = ["国語", "数学", "英語"]

COLOR_DOMAIN = ["国語", "数学", "英語"]
COLOR_RANGE = ["red", "blue", "green"]

EPS = 0.1
MAX_TIME = 40


def safe_limit(target, minimum_limit):
    return min(100, max(target + 1, minimum_limit))


def approach_exp(t, start, limit, k):
    return limit - (limit - start) * np.exp(-k * t)


def approach_sqrt(t, start, limit, k):
    return start + (limit - start) * (1 - np.exp(-k * np.sqrt(t)))


def approach_linear(t, start, limit, slope):
    return np.minimum(start + slope * t, limit)


def english_plateau_sqrt(t, start, limit, first_k, second_k):
    plateau_score = limit * 3 / 4

    if plateau_score <= start:
        return approach_sqrt(t, start, limit, second_k)

    t1 = ((-np.log((limit - plateau_score) / (limit - start))) / first_k) ** 2

    if t <= t1:
        return approach_sqrt(t, start, limit, first_k)
    elif t <= t1 + 5:
        return plateau_score
    else:
        return approach_sqrt(t - t1 - 5, plateau_score, limit, second_k)


def get_model_config(subject, proficiency, target):
    if subject == "国語":
        if proficiency <= 2:
            return {"start": 40, "limit": safe_limit(target, target), "type": "exp", "k": 0.06}
        elif proficiency <= 4:
            return {"start": 50, "limit": safe_limit(target, target), "type": "exp", "k": 0.09}
        else:
            return {"start": 70, "limit": safe_limit(target, 95), "type": "sqrt", "k": 0.22}

    elif subject == "英語":
        if proficiency <= 2:
            if target >= 60:
                return {"start": 20, "limit": safe_limit(target, 80), "type": "english_plateau", "first_k": 0.45, "second_k": 0.18}
            else:
                return {"start": 20, "limit": safe_limit(target, target), "type": "linear", "slope": 0.7}

        elif proficiency <= 4:
            return {"start": 40, "limit": safe_limit(target, 95), "type": "english_plateau", "first_k": 0.38, "second_k": 0.16}

        else:
            return {"start": 70, "limit": safe_limit(target, 90), "type": "exp", "k": 0.12}

    elif subject == "数学":
        if proficiency <= 2:
            return {"start": 20, "limit": safe_limit(target, 60), "type": "exp", "k": 0.04}
        elif proficiency == 3:
            return {"start": 40, "limit": safe_limit(target, 80), "type": "exp", "k": 0.08}
        elif proficiency == 4:
            return {"start": 60, "limit": safe_limit(target, 90), "type": "exp", "k": 0.12}
        else:
            return {"start": 80, "limit": safe_limit(target, 98), "type": "linear", "slope": 0.7}


def calc_score(t, config):
    start = config["start"]
    limit = config["limit"]

    if config["type"] == "exp":
        score = approach_exp(t, start, limit, config["k"])
    elif config["type"] == "sqrt":
        score = approach_sqrt(t, start, limit, config["k"])
    elif config["type"] == "linear":
        score = approach_linear(t, start, limit, config["slope"])
    elif config["type"] == "english_plateau":
        score = english_plateau_sqrt(t, start, limit, config["first_k"], config["second_k"])

    return np.clip(score, 0, 100)


def calc_time(target, config):
    if target <= config["start"]:
        return 0

    for t in np.arange(0, MAX_TIME + 0.5, 0.5):
        if calc_score(t, config) >= target - EPS:
            return t

    return None


st.set_page_config(page_title="学習時間シミュレーター", layout="wide")
st.title("🎯 学習時間シミュレーター")

tab1, tab2 = st.tabs(["📊 目標達成予測", "👥 シミュレーション"])


with tab1:
    st.header("科目別学習設計")
    st.caption("1教科最大40時間、3教科合計で約120時間を想定しています。")

    cols = st.columns(3)
    user_profile = {}

    total_time = 0
    all_reached = True

    for i, sub in enumerate(SUBJECTS):
        with cols[i]:
            st.subheader(sub)

            prof = st.slider(f"{sub}の得意度", 1, 5, 3, key=f"p_{sub}")
            target = st.number_input(f"{sub}の目標点数", 0, 100, 75, key=f"t_{sub}")

            config = get_model_config(sub, prof, target)
            user_profile[sub] = {"config": config, "target": target}

            req_time = calc_time(target, config)

            st.write(f"初期点数：{config['start']}点")
            st.write(f"収束点：{config['limit']}点")

            if req_time is None:
                all_reached = False
                st.warning("40時間以内には未達成")
            else:
                total_time += req_time
                st.success(f"必要時間：{int(np.ceil(req_time))}時間")

    st.divider()

    if all_reached:
        st.metric("3教科合計の必要時間", f"{int(np.ceil(total_time))}時間")
    else:
        st.metric("3教科合計の必要時間", "一部未達成")

    st.subheader("📈 学習モデルの可視化")

    t_range = np.arange(0, MAX_TIME + 1, 1)
    rows = []

    for sub in SUBJECTS:
        config = user_profile[sub]["config"]

        for t in t_range:
            rows.append({
                "時間": t,
                "点数": calc_score(t, config),
                "科目": sub
            })

    df = pd.DataFrame(rows)

    line = alt.Chart(df).mark_line(size=3).encode(
        x=alt.X("時間", scale=alt.Scale(domain=[0, MAX_TIME])),
        y=alt.Y("点数", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "科目",
            scale=alt.Scale(domain=COLOR_DOMAIN, range=COLOR_RANGE)
        )
    )

    mark_rows = []

    for sub in SUBJECTS:
        config = user_profile[sub]["config"]
        target = user_profile[sub]["target"]
        t = calc_time(target, config)

        if t is not None:
            mark_rows.append({
                "時間": t,
                "点数": target,
                "科目": sub,
                "ラベル": f"{sub} 目標到達"
            })

    mark_df = pd.DataFrame(mark_rows)

    if len(mark_df) > 0:
        points = alt.Chart(mark_df).mark_point(
            size=180,
            filled=True,
            shape="diamond"
        ).encode(
            x="時間",
            y="点数",
            color=alt.Color(
                "科目",
                scale=alt.Scale(domain=COLOR_DOMAIN, range=COLOR_RANGE)
            ),
            tooltip=["科目", "時間", "点数"]
        )

        text = alt.Chart(mark_df).mark_text(dy=-15).encode(
            x="時間",
            y="点数",
            text="ラベル",
            color=alt.Color(
                "科目",
                scale=alt.Scale(domain=COLOR_DOMAIN, range=COLOR_RANGE)
            )
        )

        st.altair_chart(line + points + text, use_container_width=True)
    else:
        st.altair_chart(line, use_container_width=True)


with tab2:
    st.header("100名シミュレーション")
    st.caption("各生徒について、1教科あたり0〜40時間の学習を想定しています。")

    if st.button("シミュレーション実行"):
        sim = []

        for _ in range(100):
            student_type = np.random.choice(
                [
                    "数学得意・国語苦手",
                    "国語得意・数学苦手",
                    "英語得意",
                    "全体得意",
                    "全体苦手",
                    "平均",
                    "ランダム"
                ],
                p=[0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1]
            )

            if student_type == "数学得意・国語苦手":
                profs = {"数学": 5, "国語": 1, "英語": 3}
            elif student_type == "国語得意・数学苦手":
                profs = {"数学": 1, "国語": 5, "英語": 3}
            elif student_type == "英語得意":
                profs = {"数学": 3, "国語": 3, "英語": 5}
            elif student_type == "全体得意":
                profs = {"数学": 5, "国語": 5, "英語": 5}
            elif student_type == "全体苦手":
                profs = {"数学": 1, "国語": 1, "英語": 1}
            elif student_type == "平均":
                profs = {"数学": 3, "国語": 3, "英語": 3}
            else:
                profs = {
                    "数学": np.random.randint(1, 6),
                    "国語": np.random.randint(1, 6),
                    "英語": np.random.randint(1, 6)
                }

            for sub in SUBJECTS:
                target = np.random.randint(50, 96)
                t = np.random.uniform(0, MAX_TIME)
                config = get_model_config(sub, profs[sub], target)
                score = calc_score(t, config) + np.random.normal(0, 3)

                sim.append({
                    "時間": t,
                    "点数": np.clip(score, 0, 100),
                    "科目": sub,
                    "得意度": profs[sub],
                    "タイプ": student_type,
                    "目標点数": target
                })

        sim_df = pd.DataFrame(sim)

        chart = alt.Chart(sim_df).mark_circle(size=70, opacity=0.75).encode(
            x=alt.X("時間", scale=alt.Scale(domain=[0, MAX_TIME])),
            y=alt.Y("点数", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "科目",
                scale=alt.Scale(domain=COLOR_DOMAIN, range=COLOR_RANGE)
            ),
            tooltip=["タイプ", "科目", "得意度", "目標点数", "時間", "点数"]
        )

        st.altair_chart(chart, use_container_width=True)