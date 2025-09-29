# c:\Repos\quant_trading_system\common\ai_integration.py
"""
AI分析機能統合ヘルパー
既存システムとAI分析の統合を簡素化
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AIIntegrationHelper:
    """AI分析機能の統合ヘルパークラス"""

    def __init__(self):
        self.ai_analyzer = None
        self.is_available = False
        self._initialize_ai()

    def _initialize_ai(self) -> None:
        """AI分析システムの初期化"""
        try:
            from .ai_analysis import get_ai_analyzer

            self.ai_analyzer = get_ai_analyzer()
            self.is_available = True
            logger.info("AI分析システム初期化完了")
        except ImportError:
            logger.info("AI分析機能は利用できません (scikit-learn 未インストール)")
            self.is_available = False
        except Exception as e:
            logger.error(f"AI分析システム初期化エラー: {e}")
            self.is_available = False

    def collect_and_analyze(
        self,
        phase_data: Dict[str, float],
        system_metrics: Optional[Dict[str, float]] = None,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """パフォーマンスデータの収集と分析の実行"""
        if not self.is_available or not self.ai_analyzer:
            return None

        try:
            # デフォルト値の設定
            if system_metrics is None:
                system_metrics = {}

            if execution_context is None:
                execution_context = {}

            # データ収集
            self.ai_analyzer.collect_performance_data(
                phase_data=phase_data,
                system_metrics=system_metrics,
                execution_context=execution_context,
            )

            # 分析実行
            analysis_result = self.ai_analyzer.analyze_current_performance(
                phase_data=phase_data,
                system_metrics=system_metrics,
                execution_context=execution_context,
            )

            return analysis_result

        except Exception as e:
            logger.error(f"AI分析エラー: {e}")
            return None

    def get_optimization_suggestions(self) -> list:
        """最適化提案の取得"""
        if not self.is_available or not self.ai_analyzer:
            return []

        try:
            return self.ai_analyzer.get_optimization_suggestions()
        except Exception as e:
            logger.error(f"最適化提案取得エラー: {e}")
            return []

    def get_analysis_summary(self) -> Dict[str, Any]:
        """分析サマリーの取得"""
        if not self.is_available or not self.ai_analyzer:
            return {
                "model_status": {"is_trained": False, "has_sklearn": False},
                "data_collection": {"total_records": 0},
                "current_analysis": {},
                "optimization_suggestions": [],
                "analysis_capabilities": {
                    "anomaly_detection": False,
                    "performance_prediction": False,
                    "optimization_suggestions": False,
                },
            }

        try:
            return self.ai_analyzer.get_analysis_summary()
        except Exception as e:
            logger.error(f"分析サマリー取得エラー: {e}")
            return {}

    def train_models(self, force_retrain: bool = False) -> bool:
        """モデルの訓練"""
        if not self.is_available or not self.ai_analyzer:
            return False

        try:
            return self.ai_analyzer.train_models(force_retrain=force_retrain)
        except Exception as e:
            logger.error(f"モデル訓練エラー: {e}")
            return False

    def log_analysis_results(
        self, analysis_result: Optional[Dict[str, Any]], logger_func=None
    ) -> None:
        """分析結果をログに出力"""
        if not analysis_result or not logger_func:
            return

        try:
            # 異常検知結果
            if analysis_result.get("is_anomaly"):
                anomaly_score = analysis_result.get("anomaly_score", 0)
                logger_func(f"AI異常検知: スコア={anomaly_score:.3f}")

            # パフォーマンス予測
            predicted_time = analysis_result.get("predicted_performance")
            if predicted_time:
                confidence = analysis_result.get("confidence", 0)
                logger_func(
                    f"AI予測実行時間: {predicted_time:.1f}秒 (信頼度: {confidence:.1%})"
                )

            # 最適化提案
            suggestions = self.get_optimization_suggestions()
            high_priority_suggestions = [
                s for s in suggestions if s.get("priority") == "high"
            ]

            for suggestion in high_priority_suggestions[:2]:  # 上位2件のみ
                title = suggestion.get("title", "No Title")
                logger_func(f"AI最適化提案: {title}")

        except Exception as e:
            if logger_func:
                logger_func(f"AI分析結果ログ出力エラー: {e}")


# グローバルインスタンス
_ai_integration_helper = None


def get_ai_integration_helper() -> AIIntegrationHelper:
    """AI統合ヘルパーのシングルトンインスタンス取得"""
    global _ai_integration_helper
    if _ai_integration_helper is None:
        _ai_integration_helper = AIIntegrationHelper()
    return _ai_integration_helper


# 便利関数
def ai_collect_and_analyze(
    phase_data: Dict[str, float],
    system_metrics: Optional[Dict[str, float]] = None,
    execution_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """AI分析のワンライナー実行"""
    helper = get_ai_integration_helper()
    return helper.collect_and_analyze(phase_data, system_metrics, execution_context)


def ai_log_results(analysis_result: Optional[Dict[str, Any]], logger_func=None) -> None:
    """AI分析結果のログ出力"""
    helper = get_ai_integration_helper()
    helper.log_analysis_results(analysis_result, logger_func)
