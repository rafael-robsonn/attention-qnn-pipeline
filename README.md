# Pipeline de classificação binária com validação cruzada, atenção espacial e interpretabilidade

Pipeline de ponta a ponta que gera um dataset sintético, compara modelos clássicos com duas redes neurais e explica as previsões da CNN com Gradient SHAP.

## O que o projeto faz

1. **Dados sintéticos.** Gera 30.000 amostras com 64 features ("pixels" de uma imagem 8x8). O rótulo depende só de `pixel_0`, `pixel_15`, `pixel_32` e `pixel_45`, mais ruído. Cerca de 15% das amostras são positivas.
2. **Engenharia de features com Polars.** Cria `pixel_energy_sum` e `interaction_core` e atribui folds estratificados por classe.
3. **Validação cruzada de 10 folds** para KNN e Árvore de Decisão (acurácia média e desvio).
4. **Split sem vazamento.** Fold 0 é teste, fold 1 é validação (early stopping) e os 8 restantes são treino.
5. **Modelos comparados no fold de teste:**
   - KNN (hiperparâmetros por GridSearch)
   - Árvore de Decisão (critério entropia, poda por `ccp_alpha`)
   - CNN com atenção espacial: bloco residual com self-attention sobre o grid 8x8
   - QNN: MLP com ativações seno e cosseno em espaço de 32 dimensões. É uma simulação clássica, não roda circuito quântico.
6. **Treino das redes:** AdamW, cosine annealing, `BCEWithLogitsLoss` e early stopping independente por modelo, com restauração dos melhores pesos.
7. **Métricas:** acurácia, F1, ROC AUC e log loss. Como a classe positiva é minoritária, compare por F1, AUC e log loss, não pela acurácia.
8. **Interpretabilidade:** Gradient SHAP (expected gradients) na CNN. Os pixels dominantes devem ser os quatro que geram o rótulo, o que serve de checagem de sanidade.
9. **Saídas** na pasta `out/`:
   - `pipeline_output.parquet`: fold de teste com as previsões dos 4 modelos
   - `curvas.html`: curvas de treino e validação (Altair)
   - `qnn_norms.html`: evolução da norma dos parâmetros da QNN (Plotly)
   - `densidades.png`: distribuição das probabilidades por modelo e classe (plotnine)
   - `shap_top10.png`: top 10 pixels por importância SHAP (seaborn objects)

## Como rodar

```
pip install -r requirements.txt
python pipeline.py
```

No Colab, `!pip install -q polars pyarrow plotnine` basta. Ative a GPU para acelerar o treino das redes. O script usa CUDA automaticamente quando disponível.

## Resultados de referência (fold de teste, n = 3001)

| Modelo | F1 | ROC AUC | Log loss |
|---|---|---|---|
| KNN | 0.10 | 0.877 | 0.310 |
| Decision Tree | 0.90 | 0.984 | 0.132 |
| CNN + Attention | 0.91 | 0.995 | 0.070 |
| QNN | 0.92 | 0.997 | 0.054 |

Os valores variam um pouco entre execuções e entre CPU e GPU.
