import pandas as pd
import streamlit as st
import tensorflow as tf

# Configuração da página
st.title("Calculadora de Preço de Pizza com IA")
st.write(
    "Bem-vindo! Este aplicativo utiliza uma rede neural simples para prever o"
    " preço de uma pizza com base no seu tamanho."
)

# Dados de treino fornecidos
pizza = pd.DataFrame(
    {"tamanho": [20, 25, 30, 35, 40], "preco": [20, 30, 40, 50, 60]}
)

# Mostrando os dados na tela de forma amigável
st.subheader("Nossos Dados de Exemplo")
st.dataframe(pizza)

# Preparando os dados para a Inteligência Artificial
X = pizza["tamanho"].astype(float)
y = pizza["preco"].astype(float)


# Criando e treinando o modelo de Rede Neural
@st.cache_resource
def treinar_modelo():
  # Criando uma rede neural com 1 neurônio
  model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

  # Usando o otimizador Adam para evitar instabilidade e o erro 'nan'
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.5),
      loss="mean_squared_error",
  )

  # Treinando o modelo (500 ciclos de aprendizado)
  model.fit(X, y, epochs=500, verbose=0)
  return model


# Treina o modelo ao carregar a página
model = treinar_modelo()

# Seção interativa para o cliente
st.subheader("Faça uma Previsão")
tamanho_usuario = st.slider(
    "Selecione o tamanho da pizza (em cm):",
    min_value=15,
    max_value=50,
    value=30,
)

# Botão para calcular
if st.button("Calcular Preço Estimado"):
  # Fazendo a previsão com o modelo treinado
  preco_previsto = model.predict(pd.Series([float(tamanho_usuario)]))[0][0]

  # Exibindo o resultado para o cliente
  st.success(
      f"O preço estimado para uma pizza de {tamanho_usuario} cm é de R$"
      f" {preco_previsto:.2f}"
  )