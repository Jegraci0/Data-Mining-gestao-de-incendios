import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ============ CONFIG ============ #
st.set_page_config(page_title="App de Modelos", layout="centered")

st.title("Deteção de Incêndios 🔥")
modelo_tipo = st.sidebar.selectbox("Escolha o tipo de modelo:", 
                                   ['Classificação', 'Regressão', 'Clustering', 'Associação'])

# ============ CLASSIFICAÇÃO ============ #
if modelo_tipo == 'Classificação':
    st.subheader("📌 Previsão de Classe com 3 variáveis manuais")

    import numpy as np

    # Inputs simples
    duracao = st.number_input("Duração (horas)", 0.0, 100.0, step=0.5)
    temperatura = st.number_input("Temperatura", 0.0, 50.0)
    fwi = st.number_input("FWI", 0.0, 100.0)

    modelo = joblib.load('Modelo_Classificacao_Regressao_Logistica.pkl')
    
    # Se tens scaler compatível, descomenta esta linha e carrega
    # scaler = joblib.load('Scaler_75features.pkl')

    n_features = modelo.n_features_in_
    entrada = np.zeros((1, n_features))

    # Vamos assumir que as 3 variáveis estão nos 3 primeiros índices
    entrada[0, 0] = duracao
    entrada[0, 1] = temperatura
    entrada[0, 2] = fwi

    # Se usares scaler, aplica ele aqui
    # entrada = scaler.transform(entrada)

    if st.button("Prever"):
        try:
            pred = modelo.predict(entrada)
            if hasattr(modelo, "predict_proba"):
                pred_prob = modelo.predict_proba(entrada)
                classe = 'Sem reacendimento' if pred[0] == 0 else 'Com reacendimento'
                prob = pred_prob[0][pred[0]]
                st.success(f"Classe prevista: {classe} (Probabilidade: {prob:.2f})")
            else:
                st.success(f"Classe prevista: {'Sem reacendimento' if pred[0] == 0 else 'Com reacendimento'}")
        except Exception as e:
            st.error(f"Erro na previsão: {e}")



# ============ REGRESSÃO ============ #
elif modelo_tipo == 'Regressão':
    st.subheader("📈 Regressão — Exploração com modelo treinado")
 
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
 
    # Carrega modelo e scaler
    modelo = joblib.load('modelo_regressao_RFR.pkl')
    scaler = joblib.load('modelo_regressao_scaler.pkl') 
 
    n_features = modelo.n_features_in_
 
     # Tenta obter os nomes das features (caso tenha vindo de um DataFrame)
    try:
        feature_names = modelo.feature_names_in_
    except AttributeError:
        feature_names = [f'feature_{i}' for i in range(n_features)]


 # Pega as 5 features mais importantes
    importances = modelo.feature_importances_
    df_importancia = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    top5 = df_importancia.head(5)

    st.markdown("### ✏️ Preencha os valores abaixo para prever:")

    # Inicializa vetor de entrada com zeros
    entrada = np.zeros((1, n_features))

    # Interface para entrada manual das 5 variáveis mais importantes
    for i, row in top5.iterrows():
        val = st.number_input(f"{row['feature']}", value=0.0, step=0.1, format="%.2f")
        idx = list(feature_names).index(row['feature'])
        entrada[0, idx] = val

    if st.button("🔍 Prever Tempo de Chegada"):
        try:
            pred = modelo.predict(entrada)
            st.success(f"🕒 Tempo estimado: **{pred[0]:.2f} minutos**")
        except Exception as e:
            st.error(f"Erro ao fazer previsão: {e}")

    st.markdown("### 📊 Importância das Features (por índice)")
    importances = modelo.feature_importances_
 
    fig, ax = plt.subplots()
    ax.bar(range(len(importances)), importances)
    ax.set_xlabel("Índice da feature")
    ax.set_ylabel("Importância")
    ax.set_title("Importância das Features no Modelo")
    st.pyplot(fig)
 


# ============ CLUSTERING ============ #
elif modelo_tipo == 'Clustering':
    st.subheader("🔗 Atribuição de Cluster com PCA")

    # Carregar modelos
    scaler = joblib.load('Modelo_Clustering_scaler.pkl')
    pca = joblib.load('Modelo_Clustering_pca.pkl')
    modelo = joblib.load('Modelo_Clustering_kmeans.pkl')

    # Variáveis originais com que o scaler foi treinado
    variaveis_originais = scaler.feature_names_in_

    # Gerar dados fictícios com as mesmas colunas
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    df = pd.DataFrame()

    for var in variaveis_originais:
        if "Area" in var:
            df[var] = np.random.uniform(0, 500, 100)
        elif "Duracao" in var:
            df[var] = np.random.uniform(0, 24, 100)
        elif "Densidade" in var:
            df[var] = np.random.uniform(0, 1000, 100)
        elif "FWI" in var or "fwi" in var:
            df[var] = np.random.uniform(0, 100, 100)
        elif "Vento" in var:
            df[var] = np.random.uniform(0, 360, 100)
        elif "Precipitacao" in var:
            df[var] = np.random.uniform(0, 10, 100)
        elif "Reacendimentos" in var or "Fogacho" in var:
            df[var] = np.random.randint(0, 2, 100)
        else:
            df[var] = np.random.uniform(0, 100, 100)

    try:
        # Aplicar pipeline
        entrada = scaler.transform(df[variaveis_originais])
        entrada_pca = pca.transform(entrada)
        clusters = modelo.predict(entrada_pca)
        df['Cluster'] = clusters

        st.success("Clusters atribuídos com sucesso!")

        # Interpretação interativa
        st.markdown("### 📘 Interpretação dos Clusters")

        interpretacoes = {
            0: "**Cluster 0 - Risco Muito Elevado**: zonas florestais de alto risco, com elevada perigosidade (3.40), fogacho (0.84) e condições críticas (altitude, declive, vento, fwi).",
            1: "**Cluster 1 - Risco Elevado**: regiões com condições meteorológicas severas, destacando-se pela temperatura (25.16 °C), fwi (36.58) e vento intenso (12.70).",
            2: "**Cluster 2 - Baixo Risco**: áreas de baixo impacto, com os menores valores nas variáveis analisadas, indicando incêndios moderados ou controlados.",
            3: "**Cluster 3 - Risco Urbano de longa duração**: zonas urbanas com elevada densidade populacional (2.82), grande área ardida (3.63) e longa duração dos incêndios (3.76 h)."
        }

        # Mostrar apenas os clusters que existem nos dados
        clusters_existentes = sorted(df['Cluster'].unique())
        interpretacoes_validas = {k: v for k, v in interpretacoes.items() if k in clusters_existentes}

        cluster_id = st.selectbox("🔍 Escolhe um cluster para ver a interpretação:", options=clusters_existentes)
        st.markdown(interpretacoes_validas[cluster_id])

        # Mostrar amostras do cluster selecionado
        st.markdown(f"### 🧪 Amostras do Cluster {cluster_id}")
        st.dataframe(df[df['Cluster'] == cluster_id].head(10))

        # Estatísticas e gráficos
        st.markdown("### 📊 Estatísticas e Gráficos por Cluster")
        stats = df.groupby('Cluster')[['Reacendimentos', 'AreaTotalIncSimul5000']].describe().transpose()
        st.write("📋 Estatísticas descritivas por cluster:")
        st.dataframe(stats)

        medias = df.groupby('Cluster')[['Reacendimentos', 'AreaTotalIncSimul5000']].mean()

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Gráfico de barras das médias
        fig, ax = plt.subplots()
        medias.plot(kind='bar', ax=ax)
        ax.set_title("Média de Reacendimentos e Área Total Incidente por Cluster")
        ax.set_ylabel("Média")
        st.pyplot(fig)

        # Histograma da variável “Reacendimentos” por cluster
        fig2, ax2 = plt.subplots()
        for cluster_id in clusters_existentes:
            subset = df[df['Cluster'] == cluster_id]
            sns.histplot(subset['Reacendimentos'], label=f'Cluster {cluster_id}', kde=False, ax=ax2, alpha=0.5)

        ax2.legend()
        ax2.set_title("Distribuição de Reacendimentos por Cluster")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Erro ao aplicar clustering com PCA: {e}")

# ============ ASSOCIAÇÃO ============ #
elif modelo_tipo == 'Associação':
    st.subheader("📋 Regras de Associação")

    rules = joblib.load('Modelo_Associacao_Apriori.pkl')

    min_conf = st.slider("Filtrar por confiança mínima", 0.0, 1.0, 0.5)
    min_lift = st.slider("Filtrar por lift mínimo", 0.0, 10.0, 1.0)
    top_n = st.slider("Mostrar até quantas regras?", 1, 50, 10)

    regras_filtradas = []

    for rule in rules:
        stat = rule.ordered_statistics[0]
        conf = stat.confidence
        lift = stat.lift

        if conf >= min_conf and lift >= min_lift:
            regras_filtradas.append({
                'antecedente': ', '.join(stat.items_base),
                'consequente': ', '.join(stat.items_add),
                'confiança': round(conf, 2),
                'lift': round(lift, 2),
                'suporte': round(rule.support, 3)
            })

    if regras_filtradas:
        df_regras = pd.DataFrame(regras_filtradas).sort_values(by='confiança', ascending=False).head(top_n)
        st.dataframe(df_regras)
    else:
        st.warning("⚠️ Nenhuma regra encontrada com esses filtros.")

