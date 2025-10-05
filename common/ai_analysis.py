# c:\Repos\quant_trading_system\common\ai_analysis.py
"""
AI支援分析機能
機械学習を使った異常検知、パフォーマンス予測、最適化提案の自動生成
"""

from collections import deque
from datetime import datetime
import json
import logging
from pathlib import Path
import pickle
import threading
import time
from typing import Any, Dict, List, Optional
import warnings

import numpy as np

# 機械学習ライブラリを条件付きインポート
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn が見つかりません。AI分析機能は制限されます。")

logger = logging.getLogger(__name__)


class AIPerformanceAnalyzer:
    """
    AI支援パフォーマンス分析システム
    機械学習を使った異常検知、予測分析、最適化提案の生成
    """

    def __init__(self, model_save_dir: str = "data_cache/ai_models"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # データ収集用
        self.performance_history = deque(maxlen=1000)  # 過去1000回分の実行データ
        self.feature_history = deque(maxlen=1000)  # 特徴量履歴

        # モデル
        self.anomaly_detector = None
        self.performance_predictor = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None

        # 学習状態
        self.is_trained = False
        self.last_training_time = None
        self.training_data_count = 0

        # 分析結果
        self.current_analysis = {}
        self.optimization_suggestions = []

        # スレッドセーフティ
        self.lock = threading.Lock()

        logger.info("AI分析システム初期化完了")
        self._load_models()

    def collect_performance_data(
        self,
        phase_data: Dict[str, float],
        system_metrics: Dict[str, float],
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """パフォーマンスデータの収集"""
        try:
            with self.lock:
                timestamp = datetime.now()

                # 特徴量の構築
                features = self._extract_features(phase_data, system_metrics, execution_context)

                # データ保存
                performance_record = {
                    "timestamp": timestamp,
                    "phase_data": phase_data.copy(),
                    "system_metrics": system_metrics.copy(),
                    "features": features,
                    "total_time": sum(phase_data.values()),
                    "execution_context": execution_context or {},
                }

                self.performance_history.append(performance_record)
                self.feature_history.append(features)

                # 定期的な再学習
                if len(self.performance_history) % 50 == 0:
                    self._schedule_retraining()

                logger.debug(f"パフォーマンスデータ収集: {len(features)}特徴量")

        except Exception as e:
            logger.error(f"パフォーマンスデータ収集エラー: {e}")

    def _extract_features(
        self,
        phase_data: Dict[str, float],
        system_metrics: Dict[str, float],
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """特徴量の抽出"""
        if not HAS_SKLEARN:
            return np.array([])

        features = []

        # フェーズデータから特徴量
        phase_times = list(phase_data.values())
        features.extend(
            [
                sum(phase_times),  # 総実行時間
                max(phase_times) if phase_times else 0,  # 最長フェーズ時間
                (np.std(phase_times) if len(phase_times) > 1 else 0),  # フェーズ時間のばらつき
                len([t for t in phase_times if t > 5.0]),  # 5秒超過フェーズ数
            ]
        )

        # システムメトリクスから特徴量
        features.extend(
            [
                system_metrics.get("cpu_percent", 0),
                system_metrics.get("memory_percent", 0),
                system_metrics.get("disk_io_read", 0),
                system_metrics.get("disk_io_write", 0),
                system_metrics.get("network_bytes", 0),
            ]
        )

        # 実行コンテキストから特徴量
        if execution_context:
            features.extend(
                [
                    execution_context.get("parallel_workers", 1),
                    execution_context.get("data_size_mb", 0),
                    execution_context.get("cache_hit_rate", 0),
                    1 if execution_context.get("test_mode", False) else 0,
                ]
            )
        else:
            features.extend([1, 0, 0, 0])

        # 時間的特徴量
        now = datetime.now()
        features.extend(
            [
                now.hour,  # 時間帯
                now.weekday(),  # 曜日
                (now - datetime(2024, 1, 1)).days,  # 年初からの日数
            ]
        )

        return np.array(features, dtype=np.float32)

    def train_models(self, force_retrain: bool = False) -> bool:
        """機械学習モデルの訓練"""
        if not HAS_SKLEARN:
            logger.warning("scikit-learn がインストールされていません")
            return False

        try:
            with self.lock:
                if len(self.performance_history) < 20:
                    logger.info("訓練データが不足しています（最低20件必要）")
                    return False

                if self.is_trained and not force_retrain:
                    logger.debug("モデルは既に訓練済みです")
                    return True

                logger.info("AI分析モデルの訓練を開始...")

                # データ準備
                X = np.array(list(self.feature_history))
                y_times = np.array([record["total_time"] for record in self.performance_history])

                # データの正規化
                X_scaled = self.scaler.fit_transform(X)

                # 異常検知モデルの訓練
                self.anomaly_detector = IsolationForest(
                    contamination=0.1,  # 10%を異常として検知
                    random_state=42,
                    n_estimators=100,
                )
                self.anomaly_detector.fit(X_scaled)

                # パフォーマンス予測モデルの訓練
                if len(X) > 10:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y_times, test_size=0.2, random_state=42
                    )

                    self.performance_predictor = RandomForestRegressor(
                        n_estimators=100, random_state=42, max_depth=10
                    )
                    self.performance_predictor.fit(X_train, y_train)

                    # モデル性能評価
                    y_pred = self.performance_predictor.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    logger.info(f"予測モデル性能 (MSE): {mse:.2f}")

                self.is_trained = True
                self.last_training_time = datetime.now()
                self.training_data_count = len(X)

                # モデル保存
                self._save_models()

                logger.info(f"AI分析モデル訓練完了 ({len(X)}件のデータで訓練)")
                return True

        except Exception as e:
            logger.error(f"モデル訓練エラー: {e}")
            return False

    def analyze_current_performance(
        self,
        phase_data: Dict[str, float],
        system_metrics: Dict[str, float],
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """現在のパフォーマンス分析"""
        if not HAS_SKLEARN or not self.is_trained:
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "predicted_performance": None,
                "confidence": 0.0,
                "analysis_status": "モデル未訓練",
            }

        try:
            with self.lock:
                # 特徴量抽出
                features = self._extract_features(phase_data, system_metrics, execution_context)
                features_scaled = self.scaler.transform([features])

                # 異常検知
                anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1

                # パフォーマンス予測
                predicted_time = None
                confidence = 0.0

                if self.performance_predictor:
                    predicted_time = self.performance_predictor.predict(features_scaled)[0]

                    # 信頼度計算（過去の予測精度に基づく）
                    if len(self.performance_history) > 10:
                        recent_actuals = [
                            r["total_time"] for r in list(self.performance_history)[-10:]
                        ]
                        recent_features = list(self.feature_history)[-10:]
                        if recent_features:
                            recent_scaled = self.scaler.transform(recent_features)
                            recent_preds = self.performance_predictor.predict(recent_scaled)
                            mse = mean_squared_error(recent_actuals, recent_preds)
                            confidence = max(0, 1 - (mse / np.mean(recent_actuals)))

                analysis_result = {
                    "anomaly_score": float(anomaly_score),
                    "is_anomaly": bool(is_anomaly),
                    "predicted_performance": (float(predicted_time) if predicted_time else None),
                    "confidence": float(confidence),
                    "analysis_status": "OK",
                    "feature_count": len(features),
                    "timestamp": datetime.now().isoformat(),
                }

                self.current_analysis = analysis_result

                # 最適化提案の生成
                if is_anomaly or (
                    predicted_time
                    and predicted_time
                    > np.mean([r["total_time"] for r in list(self.performance_history)[-20:]])
                ):
                    self._generate_optimization_suggestions(phase_data, system_metrics, features)

                return analysis_result

        except Exception as e:
            logger.error(f"パフォーマンス分析エラー: {e}")
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "predicted_performance": None,
                "confidence": 0.0,
                "analysis_status": f"エラー: {str(e)}",
            }

    def _generate_optimization_suggestions(
        self,
        phase_data: Dict[str, float],
        system_metrics: Dict[str, float],
        features: np.ndarray,
    ) -> None:
        """最適化提案の生成"""
        try:
            suggestions = []

            # フェーズ分析に基づく提案
            if phase_data:
                slowest_phase = max(phase_data.items(), key=lambda x: x[1])
                if slowest_phase[1] > 10.0:  # 10秒超過
                    suggestions.append(
                        {
                            "type": "phase_optimization",
                            "priority": "high",
                            "title": f"{slowest_phase[0]} フェーズの最適化",
                            "description": f"{slowest_phase[0]}が{slowest_phase[1]:.1f}秒と長時間です。並列処理やキャッシュ活用を検討してください。",
                            "estimated_improvement": "20-40%の時間短縮",
                        }
                    )

            # システムリソースに基づく提案
            cpu_usage = system_metrics.get("cpu_percent", 0)
            memory_usage = system_metrics.get("memory_percent", 0)

            if cpu_usage > 80:
                suggestions.append(
                    {
                        "type": "resource_optimization",
                        "priority": "medium",
                        "title": "CPU使用率最適化",
                        "description": f"CPU使用率が{cpu_usage:.1f}%と高いです。処理の分散化やアルゴリズム改善を検討してください。",
                        "estimated_improvement": "10-30%の負荷軽減",
                    }
                )

            if memory_usage > 70:
                suggestions.append(
                    {
                        "type": "resource_optimization",
                        "priority": "medium",
                        "title": "メモリ使用量最適化",
                        "description": f"メモリ使用率が{memory_usage:.1f}%です。データ分割やガベージコレクション最適化を検討してください。",
                        "estimated_improvement": "15-25%のメモリ効率化",
                    }
                )

            # 機械学習ベースの提案
            if self.performance_predictor and len(self.performance_history) > 50:
                feature_importance = self.performance_predictor.feature_importances_
                most_important_idx = np.argmax(feature_importance)

                feature_names = [
                    "総実行時間",
                    "最長フェーズ時間",
                    "フェーズ時間ばらつき",
                    "長時間フェーズ数",
                    "CPU使用率",
                    "メモリ使用率",
                    "ディスクI/O読込",
                    "ディスクI/O書込",
                    "ネットワーク",
                    "並列ワーカー数",
                    "データサイズ",
                    "キャッシュヒット率",
                    "テストモード",
                    "時間帯",
                    "曜日",
                    "日数",
                ]

                if most_important_idx < len(feature_names):
                    suggestions.append(
                        {
                            "type": "ml_insight",
                            "priority": "info",
                            "title": "AI分析による重要要素",
                            "description": f"パフォーマンスに最も影響するのは「{feature_names[most_important_idx]}」です。この要素の最適化を優先してください。",
                            "estimated_improvement": "分析に基づく最適化",
                        }
                    )

            self.optimization_suggestions = suggestions

            if suggestions:
                logger.info(f"{len(suggestions)}個の最適化提案を生成しました")

        except Exception as e:
            logger.error(f"最適化提案生成エラー: {e}")

    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """最適化提案の取得"""
        with self.lock:
            return self.optimization_suggestions.copy()

    def get_analysis_summary(self) -> Dict[str, Any]:
        """分析サマリーの取得"""
        with self.lock:
            summary = {
                "model_status": {
                    "is_trained": self.is_trained,
                    "training_data_count": self.training_data_count,
                    "last_training": (
                        self.last_training_time.isoformat() if self.last_training_time else None
                    ),
                    "has_sklearn": HAS_SKLEARN,
                },
                "data_collection": {
                    "total_records": len(self.performance_history),
                    "feature_count": (len(self.feature_history[0]) if self.feature_history else 0),
                    "collection_rate": f"{len(self.performance_history)}/1000",
                },
                "current_analysis": (self.current_analysis.copy() if self.current_analysis else {}),
                "optimization_suggestions": self.get_optimization_suggestions(),
                "analysis_capabilities": {
                    "anomaly_detection": HAS_SKLEARN and self.is_trained,
                    "performance_prediction": HAS_SKLEARN
                    and self.is_trained
                    and self.performance_predictor is not None,
                    "optimization_suggestions": True,
                },
            }

            return summary

    def _schedule_retraining(self) -> None:
        """再訓練のスケジューリング"""

        def retrain():
            try:
                time.sleep(1)  # 少し待ってから実行
                self.train_models(force_retrain=True)
            except Exception as e:
                logger.error(f"再訓練エラー: {e}")

        threading.Thread(target=retrain, daemon=True).start()

    def _save_models(self) -> None:
        """モデルの保存"""
        if not HAS_SKLEARN or not self.is_trained:
            return

        try:
            # 異常検知モデル
            if self.anomaly_detector:
                with open(self.model_save_dir / "anomaly_detector.pkl", "wb") as f:
                    pickle.dump(self.anomaly_detector, f)

            # 予測モデル
            if self.performance_predictor:
                with open(self.model_save_dir / "performance_predictor.pkl", "wb") as f:
                    pickle.dump(self.performance_predictor, f)

            # スケーラー
            if self.scaler:
                with open(self.model_save_dir / "scaler.pkl", "wb") as f:
                    pickle.dump(self.scaler, f)

            # メタデータ
            metadata = {
                "last_training": (
                    self.last_training_time.isoformat() if self.last_training_time else None
                ),
                "training_data_count": self.training_data_count,
                "is_trained": self.is_trained,
            }

            with open(self.model_save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.debug("AI分析モデルを保存しました")

        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")

    def _load_models(self) -> None:
        """保存されたモデルの読み込み"""
        if not HAS_SKLEARN:
            return

        try:
            # メタデータ確認
            metadata_path = self.model_save_dir / "metadata.json"
            if not metadata_path.exists():
                logger.debug("保存されたモデルが見つかりません")
                return

            with open(metadata_path) as f:
                metadata = json.load(f)

            # 異常検知モデル
            anomaly_path = self.model_save_dir / "anomaly_detector.pkl"
            if anomaly_path.exists():
                with open(anomaly_path, "rb") as f:
                    self.anomaly_detector = pickle.load(f)

            # 予測モデル
            predictor_path = self.model_save_dir / "performance_predictor.pkl"
            if predictor_path.exists():
                with open(predictor_path, "rb") as f:
                    self.performance_predictor = pickle.load(f)

            # スケーラー
            scaler_path = self.model_save_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            # 状態復元
            self.is_trained = metadata.get("is_trained", False)
            self.training_data_count = metadata.get("training_data_count", 0)
            if metadata.get("last_training"):
                self.last_training_time = datetime.fromisoformat(metadata["last_training"])

            logger.info(
                f"保存されたAI分析モデルを読み込みました (訓練データ: {self.training_data_count}件)"
            )

        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            # エラー時は初期状態にリセット
            self.anomaly_detector = None
            self.performance_predictor = None
            self.is_trained = False


# グローバルインスタンス
_ai_analyzer = None


def get_ai_analyzer() -> AIPerformanceAnalyzer:
    """AI分析システムのシングルトンインスタンス取得"""
    global _ai_analyzer
    if _ai_analyzer is None:
        _ai_analyzer = AIPerformanceAnalyzer()
    return _ai_analyzer
