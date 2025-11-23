# HealthHelp IA

Aplicação web baseada em Deep Learning e Visão Computacional para análise emocional e avaliação de rotina, com foco em promover bem-estar mental e equilíbrio diário.
O projeto utiliza uma rede neural convolucional (CNN) pré-treinada para classificar emoções faciais e integrar esses resultados a um sistema de recomendações personalizadas.

## Objetivo do Projeto

O HealthHelp IA foi desenvolvido para:
* Analisar a emoção do usuário a partir de uma imagem facial.
* Avaliar a rotina diária (sono, trabalho, lazer e exercícios).
* Oferecer sugestões personalizadas de bem-estar.
* Integrar Deep Learning, visão computacional e análise de hábitos em uma aplicação moderna e intuitiva.

## Tecnologias Utilizadas

### Back-end / IA

* Python 3.10
* TensorFlow 2.x / Keras
* NumPy
* OpenCV (para detecção facial)
* Pillow

### Aplicação Web

* Streamlit Cloud

### Inteligência Artificial

* CNN (Convolutional Neural Network)
* Detecção facial via Haar Cascades
* Entradas processadas para 48×48 grayscale
* Classes emocionais: Raiva, Nojo, Medo, Feliz, Triste, Surpreso, Neutro

## Arquitetura do Projeto

```
Bem-Estar-IA/
│── app.py               # Aplicação principal (Streamlit)
│── requirements.txt     # Dependências
│── emotion_model.h5     # Modelo CNN pré-treinado
│── haarcascade_...xml   # (Carregado via OpenCV interno)
│── README.md            # Documentação
```

## Como o Projeto Funciona

### 1. Processamento da Imagem (Visão Computacional)

A aplicação utiliza o OpenCV para detectar rostos na imagem enviada. Se um rosto for encontrado, ele é recortado (face crop), convertido para escala de cinza e redimensionado para 48×48 pixels antes de entrar na rede neural.

### 2. Predição da Emoção

O modelo (`emotion_model.h5`) processa a imagem recortada e retorna as probabilidades para as 7 emoções possíveis.

### 3. Avaliação de Rotina

O usuário informa:
* Horas de sono
* Trabalho/estudo
* Lazer
* Exercícios

### 4. Recomendações

O sistema combina a emoção detectada + dados da rotina e gera dicas personalizadas de saúde e bem-estar.

## Como Executar Localmente

### Clone o repositório

```bash
git clone https://github.com/Ramalho044/IoT_GS
cd IoT_GS
```

(Nota: Verifique se a pasta criada foi `IoT_GS` ou `Bem-Estar-IA`)

### Instale as dependências

```bash
pip install -r requirements.txt
```

### Execute o app

```bash
streamlit run app.py
```

## Streamlit Cloud

1. Envie o projeto (incluindo o arquivo `emotion_model.h5`) ao GitHub.
2. Vá até https://iotgshealthhel.streamlit.app/.
3. A aplicação carregará o modelo treinado e estará pronta para uso.

## Integração com Disciplinas

* Visão Computacional: Detecção de faces e pré-processamento de imagem.
* Deep Learning: Uso de redes neurais convolucionais (CNN).
* Desenvolvimento Web: Interface interativa com Streamlit.
* Saúde e Bem-estar: Lógica de recomendações de saúde mental.
* HCI/UX: Feedback visual imediato e interatividade.

## Critérios de Avaliação — Atendidos

| Critério | Status | Detalhes |
|----------|--------|----------|
| Deep Learning (60 pts) | ✔ | Modelo CNN funcional e carregamento otimizado |
| Integração (20 pts) | ✔ | IA + Saúde + Análise de Rotina |
| Boas práticas (10 pts) | ✔ | Código modular, tratamento de erros e Clean Code |
| Apresentação (10 pts) | ✔ | Interface intuitiva e feedback visual claro |

## Autores

* Gabriel Lima Silva - RM 556773
* Cauã Marcelo Da Silva Machado - RM 558024
* Marcos Ramalho - RM 554611

