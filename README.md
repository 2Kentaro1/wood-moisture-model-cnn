# Wood NIR CNN Embedding Experiment

木材NIRスペクトルから含水率を予測するための、CNN embedding + 手作り特徴量 + tabular model比較プロジェクトです。

重要な方針:

- hard routing / oracle regime / target由来cluster は使いません。
- CVは `species number` による `GroupKFold` です。
- `species_index_balanced` sample weight をCNN lossと後段モデルの両方で使います。
- CNNは含水率回帰器そのものではなく、transport latent / spectral embedding抽出器として扱います。

## Folder

```text
data/
  train.csv
  test.csv
notebooks/
  00_colab_setup.ipynb
  10_run_cnn_embedding_experiment.ipynb
  20_analyze_cnn_results.ipynb
src/nir_mc/
scripts/
outputs/nir_cnn_embedding/
```

## Colab Setup

1. Google Colab Webで `notebooks/00_colab_setup.ipynb` を開き、GitHubからpublic repoをcloneまたはpullします。
2. `pip install -r requirements.txt` を実行します。
3. cloneされたrepo直下の `data/` に `train.csv` と `test.csv` を配置します。
4. `notebooks/10_run_cnn_embedding_experiment.ipynb` からscriptを実行します。

GitHub repoは `https://github.com/2Kentaro1/wood-moisture-model-cnn.git` を使います。public repoなのでtokenなしでclone/pullできます。

## Run

```bash
python scripts/run_cnn_embedding_experiment.py \
  --train-path data/train.csv \
  --test-path data/test.csv \
  --output-dir outputs/nir_cnn_embedding \
  --band full \
  --embedding-dim 16 \
  --target-transform none \
  --epochs 100
```

## Outputs

- `features/handcrafted_features_train.csv`
- `features/handcrafted_features_test.csv`
- `features/cnn_embedding_train_<setting>.csv`
- `features/cnn_embedding_test_<setting>.csv`
- `models/cnn_fold*_<setting>.pt`
- `models/downstream_*_<setting>.pkl`
- `results/cnn_oof_predictions_<setting>.csv`
- `results/downstream_compare_<setting>.csv`
- `results/final_compare_df.csv`
- `results/config_<setting>.json`
- `figures/*.png`

## Submissions

`sample number,pred` の2列、headerなし、`utf-8-sig` で保存します。

- `submission_CNN_only.csv`
- `submission_phase7_extra_species_index.csv`
- `submission_phase7_cnn_embedding_extra.csv`
- `submission_phase7_stable4_cnn_embedding_extra.csv`
- `submission_ensemble_cnn_phase7.csv`
