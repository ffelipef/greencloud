## ⚡ GreenCloud — Sistema Evolutivo-Fuzzy para Precificação Dinâmica de Servidores

O **GreenCloud** é um protótipo funcional de Inteligência Computacional que une um **Motor de Inferência Fuzzy Mamdani** (para tomada de decisão e tarifação dinâmica) a um **Motor Evolutivo (Algoritmo Genético)** encarregado de sintonizar as funções de pertinência para maximizar a lucratividade sob restrições severas de infraestrutura.

---

## 🚀 Guia de Instalação e Execução Local

Siga os passos descritos abaixo para preparar o ambiente virtual e inicializar a aplicação Streamlit no seu computador.

### 1. Preparar o Ambiente Virtual
Crie um ambiente isolado para evitar conflitos de dependências globais na sua máquina:

```bash
# No Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# No Windows:
python -m venv venv
    call venv/Scripts/activate
```

### 2. Instalar as Dependências Obrigatórias
Certifique-se de que o arquivo requirements.txt fornecido está no mesmo diretório e execute:
```bash
    pip install -r requirements.txt 
```
As s dependências incluem: streamlit, scikit-fuzzy, numpy, matplotlib e pandas. 

### 3. Iniciar a Aplicação
Execute o servidor web do Streamlit para abrir o painel de controlo técnico:
```bash
    streamlit run app.py
```

Após o comando, o sistema abrirá automaticamente uma aba no seu navegador web padrão através do endereço: http://localhost:8501. 

---

## 🛠️ Arquitetura e Engenharia do Sistema

### 1. Motor Fuzzy (Mamdani)
O motor monitoriza continuamente o estado de telemetria dos servidores utilizando duas entradas:
Carga do Servidor (0 a 100%): Mapeado em Baixa, Média e Alta.
Demanda de Requisições (0 a 1000 req/s): Mapeado em Baixa, Média e Alta.
Variável de Saída (Multiplicador de Preço): Fator dinâmico de 1.0x a 5.0x.
O mapeamento baseia-se numa matriz estrita de 12 regras lógicas estruturadas que tratam desde estados ociosos a anomalias de rede (como ataques DDoS).

### 2. Otimização por Algoritmo Genético (AG)
Mapeamento Genético: O cromossomo é composto por uma cadeia contínua de 7 números reais (float) entre 1.0 e 5.0. No momento da execução do simulador, os genes são ordenados e mapeados nos vértices das curvas triangulares da variável de preço.
Mecanismos de Busca: Seleção por Torneio ($k=3$), Crossover Blend (BLX-$\alpha$) para espaço contínuo e Mutação Gaussiana com decaimento.
Garantia de Variabilidade: Conforme os requisitos estritos, o AG roda em 5 execuções independentes utilizando sementes aleatórias (seeds) distintas (42, 101, 777, 2026, 99) para gerar métricas estáveis de desvio-padrão.
### 3. Função de Aptidão (Fitness Financeiro)

A avaliação quantitativa simula o comportamento financeiro ao longo de 10 cenários históricos e pune com rigor desvios de mercado:
Penalidade por Abuso de Mercado: Preços excessivos quando a procura/demanda está excessivamente baixa.
Penalidade por Risco de Quebra de SLA: Sub-precificação em momentos de sobrecarga crítica de hardware.

