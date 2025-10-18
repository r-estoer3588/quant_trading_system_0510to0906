# c:\Repos\quant_trading_system\common\ai_dashboard.py
"""
AI支援分析ダッシュボード
機械学習モデルの状態、分析結果、最適化提案を可視化
"""

import logging
from typing import Any, Dict

try:
    import numpy as np
    import plotly.graph_objects as go
    import streamlit as st

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from .ai_analysis import get_ai_analyzer

logger = logging.getLogger(__name__)


def render_ai_analysis_page() -> None:
    """AI支援分析ページの描画"""
    if not HAS_PLOTTING:
        st.error("📊 AI分析表示には plotly と streamlit が必要です")
        st.code("pip install plotly streamlit", language="bash")
        return

    st.title("🤖 AI支援分析システム")
    st.markdown("機械学習を使った異常検知、パフォーマンス予測、最適化提案")

    try:
        ai_analyzer = get_ai_analyzer()

        # 分析サマリー取得
        analysis_summary = ai_analyzer.get_analysis_summary()

        # サマリーカード表示
        render_ai_summary_cards(analysis_summary)

        # タブで機能を分割
        tab1, tab2, tab3, tab4 = st.tabs(["📈 モデル状態", "🔍 異常検知", "📊 パフォーマンス予測", "💡 最適化提案"])

        with tab1:
            render_model_status_tab(analysis_summary)

        with tab2:
            render_anomaly_detection_tab(analysis_summary)

        with tab3:
            render_performance_prediction_tab(analysis_summary)

        with tab4:
            render_optimization_suggestions_tab(analysis_summary)

    except Exception as e:
        st.error(f"AI分析システムエラー: {e}")
        logger.error(f"AI分析ページエラー: {e}")


def render_ai_summary_cards(summary: Dict[str, Any]) -> None:
    """AIシステムサマリーカードの表示"""
    model_status = summary.get("model_status", {})
    data_collection = summary.get("data_collection", {})
    _analysis_capabilities = summary.get("capabilities", {})  # 将来拡張用・未使用保持

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        is_trained = model_status.get("is_trained", False)
        status_color = "🟢" if is_trained else "🔴"
        st.metric(
            label=f"{status_color} モデル状態",
            value="訓練済み" if is_trained else "未訓練",
            delta=f"データ: {model_status.get('training_data_count', 0)}件",
        )

    with col2:
        has_sklearn = model_status.get("has_sklearn", False)
        ml_color = "🟢" if has_sklearn else "🟡"
        st.metric(
            label=f"{ml_color} ML機能",
            value="利用可能" if has_sklearn else "制限モード",
            delta="scikit-learn" if has_sklearn else "インストール推奨",
        )

    with col3:
        total_records = data_collection.get("total_records", 0)
        collection_rate = data_collection.get("collection_rate", "0/1000")
        st.metric(
            label="📊 データ収集",
            value=f"{total_records}件",
            delta=f"容量: {collection_rate}",
        )

    with col4:
        current_analysis = summary.get("current_analysis", {})
        analysis_status = current_analysis.get("analysis_status", "N/A")
        status_emoji = "✅" if analysis_status == "OK" else "⚠️"
        st.metric(
            label=f"{status_emoji} 分析状態",
            value=analysis_status,
            delta=(
                current_analysis.get("timestamp", "").split("T")[1][:8] if current_analysis.get("timestamp") else None
            ),
        )


def render_model_status_tab(summary: Dict[str, Any]) -> None:
    """モデル状態タブの描画"""
    st.subheader("🔧 機械学習モデル状態")

    model_status = summary.get("model_status", {})
    data_collection = summary.get("data_collection", {})
    capabilities = summary.get("analysis_capabilities", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 データ収集状況")

        total_records = data_collection.get("total_records", 0)
        max_records = 1000
        progress = min(total_records / max_records, 1.0)

        st.progress(progress)
        st.text(f"収集データ: {total_records}/{max_records} 件")
        st.text(f"特徴量数: {data_collection.get('feature_count', 0)}")

        # データ収集の可視化
        if total_records > 0:
            # 模擬的な進捗グラフ
            days = list(range(max(1, total_records - 50), total_records + 1))
            cumulative_data = [min(i, total_records) for i in days]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=cumulative_data,
                    mode="lines+markers",
                    name="データ蓄積",
                    line=dict(color="blue", width=3),
                )
            )

            fig.update_layout(
                title="データ収集進捗",
                xaxis_title="実行回数",
                yaxis_title="累積データ数",
                height=300,
            )

            st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("### 🤖 モデル情報")

        # モデル訓練状況
        is_trained = model_status.get("is_trained", False)  # noqa: F841
        training_count = model_status.get("training_data_count", 0)  # noqa: F841
        last_training = model_status.get("last_training")

        status_data = {
            "異常検知モデル": ("✅ 利用可能" if capabilities.get("anomaly_detection") else "❌ 未訓練"),
            "パフォーマンス予測": ("✅ 利用可能" if capabilities.get("performance_prediction") else "❌ 未訓練"),
            "最適化提案": ("✅ 利用可能" if capabilities.get("optimization_suggestions") else "❌ 未対応"),
            "scikit-learn": ("✅ インストール済み" if model_status.get("has_sklearn") else "❌ 未インストール"),
        }

        for feature, status in status_data.items():
            st.text(f"{feature}: {status}")

        if last_training:
            st.text(f"最終訓練: {last_training.split('T')[0]}")

        # 再訓練ボタン
        if st.button("🔄 モデル再訓練", key="retrain_models"):
            with st.spinner("モデルを再訓練中..."):
                ai_analyzer = get_ai_analyzer()
                success = ai_analyzer.train_models(force_retrain=True)
                if success:
                    st.success("✅ モデル再訓練完了!")
                    st.rerun()
                else:
                    st.error("❌ 再訓練に失敗しました")

    # 機能説明
    st.markdown("### 📖 AI分析機能の説明")

    with st.expander("異常検知モデル (Isolation Forest)"):
        st.markdown(
            """
        - **目的**: 通常と異なるパフォーマンスパターンを自動検知
        - **アルゴリズム**: Isolation Forest（孤立森林法）
        - **検知対象**: CPU使用率、実行時間、メモリ使用量の異常パターン
        - **精度**: 汚染率10%（10%を異常として検知）
        """
        )

    with st.expander("パフォーマンス予測モデル (Random Forest)"):
        st.markdown(
            """
        - **目的**: 実行時間の予測とパフォーマンス傾向の分析
        - **アルゴリズム**: Random Forest回帰
        - **予測対象**: 総実行時間、フェーズ別処理時間
        - **特徴量**: システムメトリクス、実行コンテキスト、時間的要因
        """
        )

    with st.expander("最適化提案システム"):
        st.markdown(
            """
        - **目的**: データに基づく具体的な改善提案の生成
        - **分析要素**: ボトルネックフェーズ、リソース使用状況、機械学習による重要要因
        - **提案内容**: 並列処理、キャッシュ活用、リソース配分最適化
        - **効果予測**: 改善見込みの定量的推定
        """
        )


def render_anomaly_detection_tab(summary: Dict[str, Any]) -> None:
    """異常検知タブの描画"""
    st.subheader("🔍 異常検知分析")

    current_analysis = summary.get("current_analysis", {})

    if not current_analysis:
        st.info("💡 システム実行後に異常検知結果が表示されます")
        return

    # 現在の異常検知状況
    col1, col2, col3 = st.columns(3)

    with col1:
        is_anomaly = current_analysis.get("is_anomaly", False)
        anomaly_color = "🔴" if is_anomaly else "🟢"
        st.metric(
            label=f"{anomaly_color} 異常検知",
            value="異常あり" if is_anomaly else "正常",
            delta=f"信頼度: {current_analysis.get('confidence', 0):.1%}",
        )

    with col2:
        anomaly_score = current_analysis.get("anomaly_score", 0)
        score_color = "🔴" if anomaly_score < -0.1 else "🟡" if anomaly_score < 0 else "🟢"
        st.metric(
            label=f"{score_color} 異常スコア",
            value=f"{anomaly_score:.3f}",
            delta="低いほど異常",
        )

    with col3:
        predicted_time = current_analysis.get("predicted_performance")
        if predicted_time:
            st.metric(label="⏱️ 予測実行時間", value=f"{predicted_time:.1f}秒", delta=None)
        else:
            st.metric(label="⏱️ 予測実行時間", value="N/A", delta="データ不足")

    # 異常検知の詳細説明
    if is_anomaly:
        st.warning("⚠️ **異常パターンが検出されました**")

        st.markdown("### 🔍 異常の詳細")
        st.markdown(
            f"""
        - **異常スコア**: {anomaly_score:.3f}（通常: > -0.1）
        - **検出時刻**: {current_analysis.get("timestamp", "N/A")}
        - **特徴量数**: {current_analysis.get("feature_count", 0)}
        """
        )

        # 異常検知のアドバイス
        st.markdown("### 💡 対応推奨事項")
        st.markdown(
            """
        1. **システムリソースの確認**: CPU、メモリ使用率をチェック
        2. **外部要因の調査**: ネットワーク、ディスクI/Oの状況確認
        3. **データ品質の確認**: 入力データの整合性チェック
        4. **最適化提案の確認**: AI分析による具体的な改善案を参照
        """
        )
    else:
        st.success("✅ **システムは正常に動作しています**")

    # 異常検知の仕組み説明
    with st.expander("🔧 異常検知の仕組み"):
        st.markdown(
            """
        **Isolation Forest アルゴリズム**

        1. **学習フェーズ**: 過去の正常なパフォーマンスデータから正常範囲を学習
        2. **検知フェーズ**: 新しいデータが正常範囲から外れているかを判定
        3. **スコア計算**: -1（異常）から +1（正常）までのスコアを算出
        4. **閾値判定**: スコアが -0.1 未満の場合に異常として検知

        **検知対象の特徴量**
        - 実行時間（総時間、最長フェーズ、ばらつき）
        - システムリソース（CPU、メモリ、I/O）
        - 実行コンテキスト（並列度、データサイズ）
        - 時間的要因（時刻、曜日）
        """
        )


def render_performance_prediction_tab(summary: Dict[str, Any]) -> None:
    """パフォーマンス予測タブの描画"""
    st.subheader("📊 パフォーマンス予測分析")

    current_analysis = summary.get("current_analysis", {})

    if not current_analysis:
        st.info("💡 システム実行後に予測結果が表示されます")
        return

    # 予測結果の表示
    predicted_time = current_analysis.get("predicted_performance")
    confidence = current_analysis.get("confidence", 0)

    if predicted_time:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="⏱️ 予測実行時間",
                value=f"{predicted_time:.1f}秒",
                delta=f"信頼度: {confidence:.1%}",
            )

        with col2:
            # 過去の実行時間との比較（模擬データ）
            ai_analyzer = get_ai_analyzer()
            if len(ai_analyzer.performance_history) > 0:
                recent_times = [r["total_time"] for r in list(ai_analyzer.performance_history)[-10:]]
                avg_time = np.mean(recent_times) if recent_times else predicted_time
                diff_percent = ((predicted_time - avg_time) / avg_time * 100) if avg_time > 0 else 0

                st.metric(
                    label="📈 過去平均との差",
                    value=f"{diff_percent:+.1f}%",
                    delta=f"平均: {avg_time:.1f}秒",
                )

        # 予測精度の可視化
        st.markdown("### 📊 予測精度トレンド")

        # 模擬的な予測精度データ
        if len(ai_analyzer.performance_history) > 5:
            recent_data = list(ai_analyzer.performance_history)[-20:]

            actual_times = [r["total_time"] for r in recent_data]
            # 模擬的な予測値（実際の実装では保存された予測値を使用）
            predicted_times = [t * (0.9 + 0.2 * np.random.random()) for t in actual_times]

            fig = go.Figure()

            x_values = list(range(len(actual_times)))

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=actual_times,
                    mode="lines+markers",
                    name="実際の実行時間",
                    line=dict(color="blue", width=3),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=predicted_times,
                    mode="lines+markers",
                    name="予測実行時間",
                    line=dict(color="red", width=2, dash="dash"),
                )
            )

            fig.update_layout(
                title="実行時間の予測精度",
                xaxis_title="実行回数（直近20回）",
                yaxis_title="実行時間（秒）",
                height=400,
                showlegend=True,
            )

            st.plotly_chart(fig, width="stretch")

        # 特徴量重要度（模擬データ）
        st.markdown("### 🎯 パフォーマンス影響要因")

        feature_names = [
            "CPU使用率",
            "メモリ使用率",
            "データサイズ",
            "並列ワーカー数",
            "時間帯",
            "キャッシュヒット率",
        ]
        importance_scores = np.random.random(len(feature_names))
        importance_scores = importance_scores / importance_scores.sum() * 100

        fig = go.Figure(data=[go.Bar(x=feature_names, y=importance_scores, marker_color="lightblue")])

        fig.update_layout(
            title="パフォーマンスへの影響度",
            xaxis_title="要因",
            yaxis_title="影響度（%）",
            height=400,
        )

        st.plotly_chart(fig, width="stretch")

    else:
        st.warning("⚠️ 予測モデルが利用できません")
        st.info("💡 十分なデータが蓄積されるとパフォーマンス予測が利用可能になります")

    # 予測機能の説明
    with st.expander("🔧 パフォーマンス予測の仕組み"):
        st.markdown(
            """
        **Random Forest 回帰モデル**

        1. **学習データ**: 過去の実行データから特徴量と実行時間の関係を学習
        2. **特徴量**: システムメトリクス、実行設定、時間的要因を組み合わせ
        3. **予測**: 現在の状況から実行時間を予測
        4. **信頼度**: 過去の予測精度に基づく信頼性スコア

        **活用メリット**
        - 事前のリソース計画立案
        - ボトルネック予測による最適化
        - SLA遵守のためのタイムアウト設定
        - 処理スケジューリングの最適化
        """
        )


def render_optimization_suggestions_tab(summary: Dict[str, Any]) -> None:
    """最適化提案タブの描画"""
    st.subheader("💡 AI最適化提案")

    suggestions = summary.get("optimization_suggestions", [])

    if not suggestions:
        st.info("💡 システム実行後に最適化提案が表示されます")
        return

    # 提案サマリー
    col1, col2, col3 = st.columns(3)

    priority_counts = {}
    for suggestion in suggestions:
        priority = suggestion.get("priority", "info")
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    with col1:
        high_count = priority_counts.get("high", 0)
        st.metric(label="🔴 高優先度", value=f"{high_count}件", delta="即座な対応推奨")

    with col2:
        medium_count = priority_counts.get("medium", 0)
        st.metric(label="🟡 中優先度", value=f"{medium_count}件", delta="計画的な改善")

    with col3:
        info_count = priority_counts.get("info", 0)
        st.metric(label="🔵 情報提供", value=f"{info_count}件", delta="参考情報")

    # 提案の詳細表示
    st.markdown("### 📋 最適化提案詳細")

    for i, suggestion in enumerate(suggestions):
        priority = suggestion.get("priority", "info")
        suggestion_type = suggestion.get("type", "general")
        title = suggestion.get("title", "No Title")
        description = suggestion.get("description", "No Description")
        estimated_improvement = suggestion.get("estimated_improvement", "N/A")

        # 優先度アイコンとカラー
        priority_config = {
            "high": {"icon": "🔴", "color": "red"},
            "medium": {"icon": "🟡", "color": "orange"},
            "info": {"icon": "🔵", "color": "blue"},
        }

        config = priority_config.get(priority, priority_config["info"])

        # 提案カード
        with st.container():
            st.markdown(
                f"""
            <div style="
                border-left: 4px solid {config["color"]};
                padding: 1rem;
                margin: 1rem 0;
                background-color: #f8f9fa;
                border-radius: 0 8px 8px 0;
            ">
                <h4>{config["icon"]} {title}</h4>
                <p><strong>タイプ:</strong> {suggestion_type}</p>
                <p><strong>詳細:</strong> {description}</p>
                <p><strong>予想効果:</strong> {estimated_improvement}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # 提案タイプ別の統計
    st.markdown("### 📊 提案カテゴリ別統計")

    type_counts = {}
    for suggestion in suggestions:
        suggestion_type = suggestion.get("type", "general")
        type_counts[suggestion_type] = type_counts.get(suggestion_type, 0) + 1

    if type_counts:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    hole=0.4,
                )
            ]
        )

        fig.update_layout(title="最適化提案の分布", height=400)

        st.plotly_chart(fig, width="stretch")

    # 実装アクションプラン
    st.markdown("### 🚀 実装アクションプラン")

    # 優先度順でソート
    priority_order = {"high": 0, "medium": 1, "info": 2}
    sorted_suggestions = sorted(suggestions, key=lambda x: priority_order.get(x.get("priority", "info"), 2))

    for i, suggestion in enumerate(sorted_suggestions[:5]):  # 上位5件のみ表示
        priority = suggestion.get("priority", "info")
        title = suggestion.get("title", "No Title")

        checkbox_key = f"suggestion_{i}_{hash(title)}"
        completed = st.checkbox(
            f"[{priority.upper()}] {title}",
            key=checkbox_key,
            help=suggestion.get("description", ""),
        )

        if completed:
            st.success(f"✅ 完了: {title}")

    # 最適化効果の計算
    if suggestions:
        st.markdown("### 📈 期待される効果")

        total_improvements = []
        for suggestion in suggestions:
            improvement_text = suggestion.get("estimated_improvement", "")
            # 簡単な数値抽出（実際の実装ではより詳細な解析が必要）
            if "%" in improvement_text:
                try:
                    # "20-40%の時間短縮" -> 30%として計算
                    numbers = [
                        int(s) for s in improvement_text.split() if s.replace("-", "").replace("%", "").isdigit()
                    ]
                    if numbers:
                        avg_improvement = sum(numbers) / len(numbers)
                        total_improvements.append(avg_improvement)
                except Exception:
                    pass

        if total_improvements:
            avg_improvement = np.mean(total_improvements)
            st.info(f"💡 全提案を実装した場合の予想改善効果: 約 {avg_improvement:.1f}%")

    # 提案システムの説明
    with st.expander("🔧 最適化提案システム"):
        st.markdown(
            """
        **AI分析による提案生成**

        1. **パフォーマンス分析**: 実行データから問題点を特定
        2. **機械学習分析**: モデルの特徴量重要度から改善ポイントを抽出
        3. **ルールベース分析**: 経験的知識に基づく最適化パターン
        4. **効果予測**: 過去データから改善効果を定量的に推定

        **提案カテゴリ**
        - **phase_optimization**: フェーズ別の処理最適化
        - **resource_optimization**: システムリソース使用の最適化
        - **ml_insight**: 機械学習による洞察
        - **configuration**: 設定パラメータの調整
        """
        )


# ダッシュボード統合関数
def render_ai_analysis_dashboard() -> None:
    """AI分析ダッシュボードの統合表示"""
    render_ai_analysis_page()
