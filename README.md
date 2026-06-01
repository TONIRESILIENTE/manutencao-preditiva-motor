🛠️ Manutenção Preditiva de Motores Elétricos

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-ff69b4)
![Status](https://img.shields.io/badge/status-concluído-success)
![Licença](https://img.shields.io/badge/licença-MIT-yellow)

> **Sistema completo de diagnóstico de falhas em rolamentos utilizando análise de vibração e machine learning, com dashboard interativo para uso em campo.**



## 🎯 Objetivo

Desenvolver um sistema inteligente capaz de **identificar automaticamente** diferentes tipos e severidades de falhas em rolamentos de motores elétricos, utilizando apenas sinais de vibração. O projeto simula um cenário real de manutenção preditiva, onde a detecção precoce de defeitos evita paradas não programadas, reduz custos e aumenta a segurança operacional.



## 🧠 Contexto do autor

Este projeto integra meu **portfólio de transição de carreira** da manutenção industrial para a área de dados e inteligência artificial. A ideia nasceu da vivência prática com máquinas rotativas e da vontade de unir conhecimento de chão de fábrica com técnicas modernas de machine learning.

Cada etapa — da extração de características à construção do dashboard — foi pensada para demonstrar não apenas habilidades técnicas, mas também a capacidade de **resolver problemas reais de engenharia** com ferramentas gratuitas e código aberto.



## 📊 Dataset

| Característica | Descrição |
|----------------|-----------|
| **Nome** | CWRU Bearing Dataset |
| **Origem** | Case Western Reserve University |
| **Equipamento** | Motor elétrico de 2 HP (Reliance Electric) |
| **Sensores** | Acelerômetros no drive end (DE) e fan end (FE) |
| **Falhas** | Pista interna (IR), pista externa (OR), esfera (B) |
| **Severidades** | 0.007", 0.014", 0.021" |
| **Frequência de amostragem** | 12 kHz |
| **Condição normal** | Motor sem defeitos |

O dataset é público e amplamente utilizado como benchmark em trabalhos acadêmicos de manutenção preditiva.



## 🏗️ Arquitetura do projeto

```
manutencao-preditiva-motor/
├── data/
│ ├── 1730 RPM/ ... 1797 RPM/ # Dados brutos (.npz)
│ └── processed/
│ └── features_cwru.csv # Características extraídas
├── notebooks/
│ ├── 01_analise_exploratoria.ipynb # EDA e extração de features
│ └── 02_modelagem.ipynb # Treinamento e avaliação
├── models/
│ ├── xgboost_cwru.pkl # Modelo treinado
│ ├── scaler.pkl # Normalizador
│ └── label_encoder.pkl # Codificador de classes
├── app/
│ └── dashboard.py # Dashboard interativo (Streamlit)
├── src/ # Scripts auxiliares
├── requirements.txt
├── .gitignore
└── README.md

```



## ⚙️ Como reproduzir

### 1. Clone o repositório
```bash
git clone https://github.com/TONIRESILIENTE/manutencao-preditiva-motor.git
cd manutencao-preditiva-motor

2. Crie e ative o ambiente virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

3. Instale as dependências
pip install -r requirements.txt

4. Execute os notebooks (opcional)
Abra a pasta no VS Code e execute os notebooks em ordem:

notebooks/01_analise_exploratoria.ipynb → extrai características e gera data/processed/features_cwru.csv

notebooks/02_modelagem.ipynb → treina modelos e salva os arquivos em models/

5. Execute o dashboard
streamlit run app/dashboard.py

Acesse http://localhost:8501 no navegador.

🧪 Metodologia
Coleta de dados: utilizados 16 sinais de vibração do drive end a 12 kHz, cobrindo 1 condição normal e 12 tipos de falhas.

Extração de características: cada sinal foi dividido em janelas de 1024 pontos. De cada janela, extraímos 9 características:

RMS, curtose, skewness, pico, fator de crista

Frequência dominante (FFT)

Energia em bandas de frequência (baixa, média, alta)

Modelagem: 4 classificadores foram treinados com validação hold-out (80% treino, 20% teste):

Random Forest, XGBoost, SVM, KNN

Avaliação: acurácia, precision, recall, f1-score e matriz de confusão.

📈 Resultados
Modelo	Acurácia
Random Forest	97.77%
XGBoost	98.26%
SVM	90.82%
KNN	94.29%
100% de acerto na classe Normal (sem falha) — zero falsos negativos.

Erros residuais ocorreram apenas entre severidades do mesmo tipo de falha (ex: OR007 confundido com OR021), nunca entre uma falha e um motor saudável.

Em um teste com um sinal novo de IR007, o modelo previu corretamente com 99,75% de confiança.

Matriz de confusão (XGBoost)
A matriz de confusão (gerada no notebook 02) mostra a diagonal principal quase perfeita, com raros desvios em classes vizinhas de severidade.

🖥️ Dashboard Interativo
O projeto inclui um dashboard web construído com Streamlit para diagnóstico instantâneo de falhas

Como executar
streamlit run app/dashboard.py

Funcionalidades
📤 Upload de arquivos .npz com sinais de vibração.

📋 Seleção de exemplos pré‑carregados (Normal, IR007, B014, OR021).

🎚️ Ajuste da janela de análise (início e tamanho).

🔍 Exibição do diagnóstico completo:

Classe predita

Confiança da predição

Status colorido (verde = saudável, amarelo = falha)
📊 Tabela com as 9 características extraídas.

📈 Gráficos interativos do sinal no tempo e no espectro de frequência.

🚧 Dificuldades enfrentadas
Dificuldade	Solução
Estrutura de nomes de arquivos inconsistente (OR@12, OR@6)	Uso de expressões regulares para padronizar os rótulos
Desbalanceamento entre classes (Normal tem mais janelas)	O XGBoost lidou bem naturalmente; espaço para ponderação de classes no futuro
Tamanho dos vetores da FFT não coincidia com frequências	Uso de np.fft.rfft e rfftfreq com truncamento para garantir compatibilidade
Caminhos de arquivos quebrando entre notebooks e dashboard	Uso de pathlib.Path com caminhos relativos robustos e os.path.dirname(__file__)
💡 Principais aprendizados
A extração de características no domínio do tempo e da frequência é a chave para transformar sinais brutos em dados modeláveis.

XGBoost superou os demais modelos, confirmando sua robustez para problemas de classificação industriais.

A normalização (StandardScaler) foi essencial para o desempenho de SVM e KNN, mas não afetou negativamente os modelos baseados em árvores.

A matriz de confusão revelou que o modelo nunca falhou em identificar um motor saudável — o requisito mais crítico em manutenção.

Um dashboard interativo agrega valor imenso ao projeto, tornando-o demonstrável para qualquer pessoa.

🔮 Próximos passos
Implementar data augmentation para balancear as classes e aumentar a robustez.

Adicionar explicabilidade com SHAP para entender quais features mais contribuem para cada classe.

Criar uma API REST (FastAPI) para servir o modelo em produção.

Integrar com um banco de dados de histórico para armazenar diagnósticos ao longo do tempo.

Testar o modelo com dados de outros motores e condições de operação (validação cruzada externa).

🤝 Agradecimentos
Este projeto foi construído integralmente com ferramentas gratuitas e open source. Agradecimento especial ao laboratório CWRU por disponibilizar o dataset e à comunidade de machine learning pelo conhecimento compartilhado.

Desenvolvido por Toni Almeida Muniz como parte de uma jornada de transição de carreira. Feedbacks são muito bem-vindos!




