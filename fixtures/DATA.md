# Digits dataset provenance

`digits.csv.gz` is the unchanged 1,797-sample, 64-feature digits file distributed
with scikit-learn at commit `25dee604bae18205b01548348388baf7a1cdfe0e` (release 1.7.2).
It contains integer pixels in [0,16] and ten digit labels. The stored SHA-256 is
`09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22`.

- [Pinned file](https://github.com/scikit-learn/scikit-learn/blob/25dee604bae18205b01548348388baf7a1cdfe0e/sklearn/datasets/data/digits.csv.gz)
- [scikit-learn dataset description](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- Alpaydin, E. and Kaynak, C. (1998), *Optical Recognition of Handwritten Digits*,
  UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49).
- The UCI dataset is provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

scikit-learn distributes this subset from the original UCI test data. Our
experiments make new stratified train/validation/test splits within that subset;
they do not claim comparability with the original UCI train/test benchmark.
The file is unmodified. Normalization, splits, trained models and activations
are experimental derivatives whose commands and hashes are retained separately.
