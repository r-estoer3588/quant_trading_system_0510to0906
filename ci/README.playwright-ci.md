Playwright CI helper files

このディレクトリには、Playwright を CI 環境で安定して動作させるための補助ファイルが入っています。

目的:

- Docker イメージをビルドして、CI 上でもローカルでも同一環境で Playwright テストを実行できるようにする。

主なファイル:

- `Dockerfile.playwright-ci`: Playwright 公式イメージをベースに、Python / Node の依存、フォント、ロケール、タイムゾーンを固定した CI イメージ
- `run_in_docker.sh`: ローカルでイメージをビルドし、テストを実行するためのシンプルなラッパー

ローカル実行手順:

1. イメージをビルド:

```bash
./ci/run_in_docker.sh --namespace local123
```

2. パイプライン実行をスキップして Playwright テストのみ実行したい場合:

```bash
./ci/run_in_docker.sh --namespace local123 --skip-pipeline
```

（`run_in_docker.sh` は内部で `docker build` を行い、コンテナ内で `python scripts/run_all_systems_today.py --test-mode mini --save-csv` と `npx playwright test` を順に実行します）

GitHub Actions での利用:

- ワークフロー `./.github/workflows/playwright-docker.yml` が追加済みです。
- このワークフローは Buildx を使ってイメージをビルドし、コンテナを実行してから `playwright-report` と `results_csv` をアーティファクトにアップロードします。

Tips:

- CI 上では `RUN_NAMESPACE` に `github.run_id` を利用して出力を分離することで、並列のワークフロー実行によるファイル衝突を避けられます。
- `PIPELINE_USE_RUN_LOCK` と `PIPELINE_USE_RUN_SUBDIR` を有効にすると、保存フェーズでロック取得を行い、出力をランごとに分離して衝突を回避します。
