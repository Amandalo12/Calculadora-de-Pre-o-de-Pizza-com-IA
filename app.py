import numpy as np
import pandas as pd
import streamlit as st

st.title("Calculadora de Preço de Pizza com IA")

st.write(
    "Bem-vindo! Este aplicativo estima o preço de uma pizza "
    "com base no seu tamanho."
)

# Dados de exemplo
pizza = pd.DataFrame({
    "tamanho": [20, 25, 30, 35, 40],
    "preco": [20, 30, 40, 50, 60]
})

st.subheader("Nossos Dados de Exemplo")
st.dataframe(pizza, hide_index=True, use_container_width=True)

# Treinamento por regressão linear
X = pizza["tamanho"].to_numpy(dtype=float)
y = pizza["preco"].to_numpy(dtype=float)

# Calcula a reta que corresponde aos dados
coeficientes = np.polyfit(X, y, 1)
inclinacao, intercepto = coeficientes

st.subheader("Faça uma Previsão")

tamanho_usuario = st.slider(
    "Selecione o tamanho da pizza (em cm):",
    min_value=15,
    max_value=50,
    value=40,
    step=1
)

if st.button("Calcular Preço Estimado"):
    preco_previsto = inclinacao * tamanho_usuario + intercepto

    st.success(
        f"O preço estimado para uma pizza de "
        f"{tamanho_usuario} cm é de R$ {preco_previsto:.2f}"
    )
