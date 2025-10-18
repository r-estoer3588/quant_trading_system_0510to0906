"""リアルタイムメトリクス表示用ダッシュボードコンポーネント。

StreamlitとPlotlyを使用したリアルタイムメトリクス可視化機能。
CPU/メモリ/処理速度の推移をグラフ形式で表示。
"""

from datetime import datetime
import time
from typing import Any, Dict, List

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import streamlit as st

    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

from .structured_logging import get_trading_logger


class RealTimeDashboard:
    """リアルタイムメトリクスダッシュボード。"""

    def __init__(self):
        self.logger = get_trading_logger()
        self.last_update = 0
        self.update_interval = 2.0  # 2秒間隔で更新

    def render_dashboard(self, time_window_minutes: int = 10) -> None:
        """メインダッシュボードを描画。"""

        if not DASHBOARD_AVAILABLE:
            st.error("ダッシュボード機能には plotly と streamlit が必要です")
            return

        st.title("🔍 リアルタイムシステム監視")

        # 自動更新設定
        if st.checkbox("自動更新 (2秒間隔)", value=True):
            if time.time() - self.last_update > self.update_interval:
                st.rerun()
                self.last_update = time.time()

        # 時間窓設定
        col1, col2 = st.columns([3, 1])
        with col2:
            time_window = st.selectbox(
                "表示時間範囲",
                options=[5, 10, 15, 30, 60],
                index=1,  # 10分
                format_func=lambda x: f"{x}分",
            )

        # ダッシュボードデータ取得
        dashboard_data = self.logger.realtime_metrics.get_dashboard_data(time_window)

        # サマリー情報
        self._render_summary_cards(dashboard_data["summary"])

        # アラート情報
        if dashboard_data["alerts"]["count"] > 0:
            self._render_alerts_section(dashboard_data["alerts"])

        # システムメトリクスグラフ
        self._render_system_metrics_charts(dashboard_data["system_metrics"], time_window)

        # システム別パフォーマンス
        if dashboard_data["system_performance"]:
            self._render_system_performance_charts(dashboard_data["system_performance"], time_window)

        # 進捗情報
        self._render_progress_section()

        # ボトルネック分析
        self._render_bottleneck_analysis()

    def _render_summary_cards(self, summary: Dict[str, Any]) -> None:
        """サマリーカードを表示。"""

        st.subheader("📊 システム状況サマリー")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="平均CPU使用率",
                value=f"{summary['avg_cpu']:.1f}%",
                delta=f"最大: {summary['max_cpu']:.1f}%",
            )

        with col2:
            st.metric(
                label="平均メモリ使用率",
                value=f"{summary['avg_memory']:.1f}%",
                delta=f"最大: {summary['max_memory']:.1f}%",
            )

        with col3:
            st.metric(
                label="総スループット",
                value=f"{summary['total_throughput']:.1f}",
                delta="items/sec",
            )

        with col4:
            st.metric(
                label="アクティブシステム",
                value=summary["active_systems"],
                delta=f"データポイント: {summary['data_points']}",
            )

    def _render_alerts_section(self, alerts_data: Dict[str, Any]) -> None:
        """アラート情報を表示。"""

        st.subheader("🚨 アクティブアラート")

        severity_counts = alerts_data["by_severity"]

        col1, col2, col3 = st.columns(3)

        with col1:
            if severity_counts["critical"] > 0:
                st.error(f"🔴 重大: {severity_counts['critical']}件")
            else:
                st.success("🔴 重大: 0件")

        with col2:
            if severity_counts["warning"] > 0:
                st.warning(f"🟡 警告: {severity_counts['warning']}件")
            else:
                st.success("🟡 警告: 0件")

        with col3:
            if severity_counts["info"] > 0:
                st.info(f"🔵 情報: {severity_counts['info']}件")
            else:
                st.success("🔵 情報: 0件")

        # 最近のアラート詳細
        if alerts_data["recent"]:
            st.write("**最近のアラート:**")
            for alert in alerts_data["recent"]:
                icon = "🔴" if alert["severity"] == "critical" else "🟡" if alert["severity"] == "warning" else "🔵"
                st.write(f"{icon} {alert['message']}")

    def _render_system_metrics_charts(self, metrics_data: Dict[str, List], time_window: int) -> None:
        """システムメトリクスグラフを表示。"""

        st.subheader("📈 システムリソース推移")

        # CPU/メモリ/スループットの複合グラフ
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "CPU使用率",
                "メモリ使用率",
                "処理スループット",
                "リソース使用率比較",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
            vertical_spacing=0.12,
        )

        # CPU使用率
        if metrics_data["cpu"]:
            timestamps = [
                datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")) for item in metrics_data["cpu"]
            ]
            cpu_values = [item["value"] for item in metrics_data["cpu"]]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cpu_values,
                    name="CPU%",
                    line=dict(color="red"),
                    mode="lines+markers",
                ),
                row=1,
                col=1,
            )

        # メモリ使用率
        if metrics_data["memory"]:
            timestamps = [
                datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")) for item in metrics_data["memory"]
            ]
            memory_values = [item["value"] for item in metrics_data["memory"]]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=memory_values,
                    name="Memory%",
                    line=dict(color="blue"),
                    mode="lines+markers",
                ),
                row=1,
                col=2,
            )

        # スループット
        if metrics_data["throughput"]:
            timestamps = [
                datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")) for item in metrics_data["throughput"]
            ]
            throughput_values = [item["value"] for item in metrics_data["throughput"]]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=throughput_values,
                    name="Throughput",
                    line=dict(color="green"),
                    mode="lines+markers",
                ),
                row=2,
                col=1,
            )

        # リソース使用率比較（CPU vs Memory）
        if metrics_data["cpu"] and metrics_data["memory"]:
            cpu_values = [item["value"] for item in metrics_data["cpu"][-50:]]  # 最新50ポイント
            memory_values = [item["value"] for item in metrics_data["memory"][-50:]]

            fig.add_trace(
                go.Scatter(
                    x=cpu_values,
                    y=memory_values,
                    mode="markers",
                    name="CPU vs Memory",
                    marker=dict(color="purple"),
                ),
                row=2,
                col=2,
            )

        # レイアウト調整
        fig.update_layout(
            height=600,
            title_text=f"システムメトリクス推移 (過去{time_window}分)",
            showlegend=True,
        )

        # 軸ラベル
        fig.update_xaxes(title_text="時刻", row=2, col=1)
        fig.update_xaxes(title_text="時刻", row=2, col=2)
        fig.update_xaxes(title_text="CPU%", row=2, col=2)
        fig.update_yaxes(title_text="CPU%", row=1, col=1)
        fig.update_yaxes(title_text="Memory%", row=1, col=2)
        fig.update_yaxes(title_text="items/sec", row=2, col=1)
        fig.update_yaxes(title_text="Memory%", row=2, col=2)

        st.plotly_chart(fig, width="stretch")

    def _render_system_performance_charts(self, system_data: Dict[str, Dict], time_window: int) -> None:
        """システム別パフォーマンスチャートを表示。"""

        st.subheader("⚡ システム別パフォーマンス")

        # システム選択
        selected_systems = st.multiselect(
            "表示するシステム",
            options=list(system_data.keys()),
            default=list(system_data.keys())[:3],  # 最初の3システムをデフォルト選択
        )

        if not selected_systems:
            st.info("表示するシステムを選択してください")
            return

        # システム別実行時間
        fig_duration = go.Figure()

        for system_name in selected_systems:
            if "duration" in system_data[system_name]:
                duration_data = system_data[system_name]["duration"]
                if duration_data:
                    timestamps = [
                        datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")) for item in duration_data
                    ]
                    durations = [item["value"] for item in duration_data]

                    fig_duration.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=durations,
                            name=f"{system_name}",
                            mode="lines+markers",
                        )
                    )

        fig_duration.update_layout(
            title=f"システム別実行時間 (過去{time_window}分)",
            xaxis_title="時刻",
            yaxis_title="実行時間 (秒)",
            height=400,
        )

        st.plotly_chart(fig_duration, width="stretch")

        # システム別スループット
        fig_throughput = go.Figure()

        for system_name in selected_systems:
            if "throughput" in system_data[system_name]:
                throughput_data = system_data[system_name]["throughput"]
                if throughput_data:
                    timestamps = [
                        datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")) for item in throughput_data
                    ]
                    throughputs = [item["value"] for item in throughput_data]

                    fig_throughput.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=throughputs,
                            name=f"{system_name}",
                            mode="lines+markers",
                        )
                    )

        fig_throughput.update_layout(
            title=f"システム別スループット (過去{time_window}分)",
            xaxis_title="時刻",
            yaxis_title="処理速度 (items/sec)",
            height=400,
        )

        st.plotly_chart(fig_throughput, width="stretch")

    def _render_progress_section(self) -> None:
        """進捗情報セクションを表示。"""

        st.subheader("⏳ アクティブ進捗")

        active_progress = self.logger.progress_tracker.get_all_active_progress()

        if not active_progress:
            st.info("現在実行中のタスクはありません")
            return

        for progress in active_progress:
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**{progress['operation']}** ({progress['system']})")

                # プログレスバー
                progress_value = progress["progress_percentage"] / 100
                st.progress(progress_value)

                # 詳細情報
                details = f"進捗: {progress['processed_items']}/{progress['total_items']} "
                details += f"({progress['progress_percentage']:.1f}%) "
                details += f"処理速度: {progress['current_rate']:.2f} items/sec"

                if progress["is_prediction_available"]:
                    details += f" | 残り時間: {progress['remaining_display']}"
                    if progress["completion_time_display"]:
                        details += f" | 完了予定: {progress['completion_time_display']}"
                    details += f" (信頼度: {progress['confidence_level']:.1f}%)"

                st.caption(details)

            with col2:
                if progress["is_prediction_available"]:
                    st.metric(
                        "残り時間",
                        progress["remaining_display"] or "計算中",
                        delta=f"信頼度 {progress['confidence_level']:.0f}%",
                    )

    def _render_bottleneck_analysis(self) -> None:
        """ボトルネック分析セクションを表示。"""

        st.subheader("🔍 ボトルネック分析")

        # 分析対象システム選択
        analysis_system = st.selectbox(
            "分析対象システム",
            options=[
                "all",
                "system1",
                "system2",
                "system3",
                "system4",
                "system5",
                "system6",
                "system7",
            ],
            index=0,
        )

        # ボトルネック分析実行
        bottleneck_data = self.logger.bottleneck_analyzer.get_ui_data(analysis_system)

        if not bottleneck_data["phases"]:
            st.info("分析データがありません")
            return

        # フェーズ別時間分析
        col1, col2 = st.columns(2)

        with col1:
            # フェーズ別時間割合（円グラフ）
            phase_names = [phase["name"] for phase in bottleneck_data["phases"]]
            time_percentages = [phase["time_percentage"] for phase in bottleneck_data["phases"]]

            fig_pie = px.pie(
                values=time_percentages,
                names=phase_names,
                title="フェーズ別処理時間割合",
            )

            st.plotly_chart(fig_pie, width="stretch")

        with col2:
            # フェーズ別平均実行時間（棒グラフ）
            avg_durations = [phase["avg_duration"] for phase in bottleneck_data["phases"]]

            fig_bar = px.bar(
                x=phase_names,
                y=avg_durations,
                title="フェーズ別平均実行時間",
                labels={"x": "フェーズ", "y": "平均実行時間 (秒)"},
            )

            # ボトルネックのフェーズをハイライト
            colors = ["red" if phase["is_bottleneck"] else "blue" for phase in bottleneck_data["phases"]]
            fig_bar.update_traces(marker_color=colors)

            st.plotly_chart(fig_bar, width="stretch")

        # ボトルネック詳細
        if bottleneck_data["bottlenecks"]:
            st.write("**🚨 検出されたボトルネック:**")
            for bottleneck in bottleneck_data["bottlenecks"]:
                severity_icon = "🔴" if bottleneck["severity"] == "high" else "🟡"
                st.write(
                    f"{severity_icon} **{bottleneck['phase']}**: "
                    f"{bottleneck['time_percentage']:.1f}% "
                    f"(平均{bottleneck['avg_duration']:.2f}秒) "
                    f"傾向: {bottleneck['trend']}"
                )

        # 推奨事項
        if bottleneck_data["recommendations"]:
            st.write("**💡 最適化推奨事項:**")
            for recommendation in bottleneck_data["recommendations"]:
                st.write(f"• {recommendation}")


def render_realtime_metrics_page():
    """リアルタイムメトリクスページを描画。"""

    if not DASHBOARD_AVAILABLE:
        st.error("📊 リアルタイムメトリクス表示には plotly と streamlit が必要です")
        st.code("pip install plotly streamlit", language="bash")
        return

    dashboard = RealTimeDashboard()

    # メトリクス収集開始
    logger = get_trading_logger()
    if not logger.realtime_metrics.collection_active:
        logger.realtime_metrics.start_collection()
        st.success("✅ リアルタイムメトリクス収集を開始しました")

    dashboard.render_dashboard()


if __name__ == "__main__":
    # Streamlitアプリとして直接実行する場合
    render_realtime_metrics_page()
