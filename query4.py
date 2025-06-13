import pandas as pd
import re
import numpy as np

ARQUIVO_ENTRADA = 'relatorio_consolidado.csv'
ARQUIVO_SAIDA = 'analise_quantitativa_pt.csv'

try:
    df = pd.read_csv(ARQUIVO_ENTRADA, sep=';')
    print(f"Arquivo '{ARQUIVO_ENTRADA}' carregado com sucesso.")
except FileNotFoundError:
    print(f"ERRO: O arquivo '{ARQUIVO_ENTRADA}' não foi encontrado.")
    print("Por favor, certifique-se de que o script está na mesma pasta que o seu relatório consolidado.")
    exit()

modelos = sorted(df['identificador_modelo'].unique())
colunas_reais = [col for col in df.columns if col.startswith('real_tem_')]
vulnerabilidades = sorted([re.sub(r'^real_tem_', '', col) for col in colunas_reais])

print(f"\nModelos encontrados: {', '.join(modelos)}")
print(f"Vulnerabilidades encontradas: {', '.join(vulnerabilidades)}\n")

lista_resultados = []

for modelo in modelos:
    df_modelo = df[df['identificador_modelo'] == modelo].copy()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for vuln in vulnerabilidades:
        coluna_real = f'real_tem_{vuln}'
        coluna_predita = f'pred_tem_{vuln}'

        if coluna_real not in df_modelo.columns or coluna_predita not in df_modelo.columns:
            continue

        y_real = df_modelo[coluna_real].astype(str).str.lower() == 'true'
        y_predito = df_modelo[coluna_predita].astype(str).str.lower() == 'true'

        tp = (y_real & y_predito).sum()
        
        fp = (~y_real & y_predito).sum()
        
        fn = (y_real & ~y_predito).sum()

        tn = (~y_real & ~y_predito).sum()
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        epsilon = 1e-9

        precisao = tp / (tp + fp + epsilon)
        
        recall = tp / (tp + fn + epsilon)

        f1_score = 2 * (precisao * recall) / (precisao + recall + epsilon)

        populacao_total = tp + tn + fp + fn
        acuracia = (tp + tn) / (populacao_total + epsilon)
        
        lista_resultados.append({
            'Modelo': modelo,
            'Vulnerabilidade': vuln,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Acuracia': acuracia,
            'Precisao': precisao,
            'Recall': recall,
            'F1-Score': f1_score
        })
    
    populacao_total_micro = total_tp + total_tn + total_fp + total_fn
    precisao_micro = total_tp / (total_tp + total_fp + epsilon)
    recall_micro = total_tp / (total_tp + total_fn + epsilon)
    f1_micro = 2 * (precisao_micro * recall_micro) / (precisao_micro + recall_micro + epsilon)
    acuracia_micro = (total_tp + total_tn) / (populacao_total_micro + epsilon)

    lista_resultados.append({
        'Modelo': modelo,
        'Vulnerabilidade': '== GERAL (Micro Media) ==',
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'TN': total_tn,
        'Acuracia': acuracia_micro,
        'Precisao': precisao_micro,
        'Recall': recall_micro,
        'F1-Score': f1_micro
    })

df_resultados = pd.DataFrame(lista_resultados)

pd.options.display.float_format = '{:,.4f}'.format

try:
    df_resultados.to_csv(ARQUIVO_SAIDA, index=False, sep=';', decimal=',')
    print(f"\nAnálise concluida. Resultados salvos em '{ARQUIVO_SAIDA}'")
except Exception as e:
    print(f"\nErro ao salvar o arquivo de resultados: {e}")

print("\n--- Amostra do Relatorio de Analise Quantitativa ---")
print(df_resultados.to_string())