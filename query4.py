import pandas as pd
import numpy as np

ARQUIVO_ENTRADA = 'relatorio_consolidado_corrigido.csv'
ARQUIVO_SAIDA = 'analise_quantitativa.csv' 

try:
    df = pd.read_csv(ARQUIVO_ENTRADA, sep=';')
    print(f"Arquivo '{ARQUIVO_ENTRADA}' carregado com sucesso.")
except FileNotFoundError:
    print(f"O arquivo de entrada '{ARQUIVO_ENTRADA}' nao foi encontrado.")
    exit()

modelos = sorted(df['identificador_modelo'].unique())
print(f"\nModelos encontrados: {', '.join(modelos)}")

lista_resultados = []

for modelo in modelos:
    df_modelo = df[df['identificador_modelo'] == modelo]
    
    tp = (df_modelo['status_validacao'] == 'VP').sum()
    fp = (df_modelo['status_validacao'] == 'FP').sum()
    fn = (df_modelo['status_validacao'] == 'FN').sum()
    tn = (df_modelo['status_validacao'] == 'VN').sum()
    
    epsilon = 1e-9

    precisao = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    acuracia = (tp + tn) / (tp + tn + fp + fn + epsilon)
    f1_score = 2 * (precisao * recall) / (precisao + recall + epsilon)

    lista_resultados.append({
        'Modelo': modelo,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Acuracia': acuracia,
        'Precisao': precisao,
        'Recall': recall,
        'F1-Score': f1_score
    })

df_resultados = pd.DataFrame(lista_resultados)

pd.options.display.float_format = '{:,.4f}'.format

try:
    df_resultados.to_csv(ARQUIVO_SAIDA, index=False, sep=';', decimal=',')
except Exception as e:
    print(f"\nErro ao salvar o arquivo de resultados: {e}")

print("\n--- Relatório de Performance Geral por Modelo ---")
print(df_resultados.to_string(index=False))