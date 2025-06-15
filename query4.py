import pandas as pd
import re
import numpy as np

ARQUIVO_ENTRADA = 'relatorio_consolidado.csv'
ARQUIVO_SAIDA = 'analise_quantitativa.csv' 

try:
    df = pd.read_csv(ARQUIVO_ENTRADA, sep=';')
    print(f"Arquivo '{ARQUIVO_ENTRADA}' carregado com sucesso.")
except FileNotFoundError:
    print(f"ERRO: O arquivo '{ARQUIVO_ENTRADA}' não foi encontrado.")
    exit()

modelos = sorted(df['identificador_modelo'].unique())
colunas_preditas = [col for col in df.columns if col.startswith('pred_tem_')]
vulnerabilidades = sorted([re.sub(r'^pred_tem_', '', col) for col in colunas_preditas])


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
        coluna_predita = f'pred_tem_{vuln}'

        if coluna_predita not in df_modelo.columns:
            continue
        
        y_real = pd.Series([True] * len(df_modelo), index=df_modelo.index)
        
        y_predito = df_modelo[coluna_predita].astype(str).str.lower() == 'true'

        tp = (y_real & y_predito).sum()      # Acertos: O modelo previu 'True' e o real era 'True'.
        fp = (~y_real & y_predito).sum()     # Erros (Tipo I): Sempre será 0, pois ~y_real é sempre 'False'.
        fn = (y_real & ~y_predito).sum()     # Erros (Tipo II): O modelo previu 'False' quando o real era 'True'.
        tn = (~y_real & ~y_predito).sum()     # Acertos Negativos: Sempre será 0, pois ~y_real é sempre 'False'.
        
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
    print(f"\nAnálise corrigida concluída. Resultados salvos em '{ARQUIVO_SAIDA}'")
except Exception as e:
    print(f"\nErro ao salvar o arquivo de resultados: {e}")

print("\n--- Amostra do Relatório de Análise Corrigida ---")
print(df_resultados[df_resultados['Vulnerabilidade'] == '== GERAL (Micro Media) =='].to_string())