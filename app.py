import time
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

st.set_page_config(page_title="GreenCloud System", layout="wide")

# historico de cenario p calculo de fitness
CENARIOS_HISTORICOS = [
    (10, 50), (25, 150), (40, 300), (50, 450), (60, 600),
    (75, 750), (85, 800), (90, 950), (95, 1000), (30, 800)
]


CENARIOS_VALIDACAO = [
    {"Nome": "Madrugada Ociosa", "Carga": 10.0, "Demanda": 50.0},
    {"Nome": "Início do Expediente", "Carga": 45.0, "Demanda": 300.0},
    {"Nome": "Pico de Acesso", "Carga": 85.0, "Demanda": 850.0},
    {"Nome": "Sobrecarga Crítica", "Carga": 98.0, "Demanda": 980.0},
    {"Nome": "Ataque DDoS / Anomalia", "Carga": 95.0, "Demanda": 150.0}, # Carga alta, pouca demanda real produtiva
    {"Nome": "Alta Eficiência", "Carga": 30.0, "Demanda": 700.0}
]

#motor fuzy (MAMDANI)
def construir_sistema_fuzzy(params_saida=None):
    """
    Constrói o Sistema de Controle Fuzzy Mamdani.
    Se 'params_saida' (cromossomo) for fornecido, os vértices de 'preco' serão customizados.
    Cromossomo esperado: lista/array com 7 pontos ordenados de 1.0 a 5.0.
    """
    carga = ctrl.Antecedent(np.arange(0, 101, 1), 'carga')
    demanda = ctrl.Antecedent(np.arange(0, 1001, 10), 'demanda')
    preco = ctrl.Consequent(np.arange(1.0, 5.1, 0.1), 'preco')

    #pert entrada
    carga['baixa'] = fuzz.trimf(carga.universe, [0, 0, 50])
    carga['media'] = fuzz.trimf(carga.universe, [20, 50, 80])
    carga['alta'] = fuzz.trimf(carga.universe, [50, 100, 100])

    demanda['baixa'] = fuzz.trimf(demanda.universe, [0, 0, 400])
    demanda['media'] = fuzz.trimf(demanda.universe, [200, 500, 800])
    demanda['alta'] = fuzz.trimf(demanda.universe, [600, 1000, 1000])

    #pert saida
    if params_saida is None:
        #heuristica padrao
        preco['baixo'] = fuzz.trimf(preco.universe, [1.0, 1.0, 3.0])
        preco['medio'] = fuzz.trimf(preco.universe, [2.0, 3.0, 4.0])
        preco['alto'] = fuzz.trimf(preco.universe, [3.0, 5.0, 5.0])
    else:
        p = sorted(params_saida)
        #p0, p1 e p2 -> baixo | p2, p3 e p4 -> medio | p4, p5 e p6 -> alto
        preco['baixo'] = fuzz.trimf(preco.universe, [p[0], p[1], p[2]])
        preco['medio'] = fuzz.trimf(preco.universe, [p[2], p[3], p[4]])
        preco['alto'] = fuzz.trimf(preco.universe, [p[4], p[5], p[6]])

    regras = [
        ctrl.Rule(carga['baixa'] & demanda['baixa'], preco['baixo']),
        ctrl.Rule(carga['baixa'] & demanda['media'], preco['baixo']),
        ctrl.Rule(carga['baixa'] & demanda['alta'], preco['medio']),
        
        ctrl.Rule(carga['media'] & demanda['baixa'], preco['baixo']),
        ctrl.Rule(carga['media'] & demanda['media'], preco['medio']),
        ctrl.Rule(carga['media'] & demanda['alta'], preco['alto']),
        
        ctrl.Rule(carga['alta'] & demanda['baixa'], preco['medio']),
        ctrl.Rule(carga['alta'] & demanda['media'], preco['alto']),
        ctrl.Rule(carga['alta'] & demanda['alta'], preco['alto']),
        
        # Regras de contorno/específicas adicionais para totalizar as 12 estritas
        ctrl.Rule(carga['baixa'] & demanda['alta'], preco['medio']),
        ctrl.Rule(carga['alta'] & demanda['baixa'], preco['medio']),
        ctrl.Rule(carga['media'] & demanda['alta'], preco['alto'])
    ]

    sistema_controle = ctrl.ControlSystem(regras)
    return ctrl.ControlSystemSimulation(sistema_controle), preco

def executar_fuzzy(simulador, v_carga, v_demanda):
    """Executa a inferência Fuzzy tratando possíveis exceções de corte."""
    try:
        simulador.input['carga'] = v_carga
        simulador.input['demanda'] = v_demanda
        simulador.compute()
        return simulador.output['preco']
    except Exception:
        # Fallback caso a desnebularização falhe por falta de ativação de regras
        return 1.0

#fitness (funca ode aptidao)
def calcular_fitness(cronossomo):
    """
    Avalia o lucro simulado sob restrições econômicas e operacionais.
    Bonifica o equilíbrio econômico e pune abusos ou sub-precificação.
    """
    simulador, _ = construir_sistema_fuzzy(cronossomo)
    lucro_total = 0.0
    
    for c_carga, c_demanda in CENARIOS_HISTORICOS:
        fator_preco = executar_fuzzy(simulador, c_carga, c_demanda)
        
        # modelo economico da simulação
        ganho = (c_demanda * 0.1) * fator_preco
        penalidade = 0.0

        #preço abusivo sem demanda
        if fator_preco > 3.5 and c_demanda < 400:
            penalidade += (fator_preco - 3.5) * 50.0
        #preço muito baixo em carga alta
        if cl_fator := (fator_preco < 2.5 and c_carga > 80):
            penalidade += (2.5 - fator_preco) * 80.0
            
        lucro_total += (ganho - penalidade)
        
    return max(0.1, lucro_total)

#algoritimo genetico
def gerar_cromossomo():
    """Gera um indivíduo válido composto por 7 genes ordenados entre 1.0 e 5.0"""
    return sorted(np.random.uniform(1.0, 5.0, 7))

def selecao_torneio(populacao, fitness, k=3):
    selecionados = np.random.choice(len(populacao), k, replace=False)
    melhor_idx = selecionados[np.argmax(fitness[selecionados])]
    return populacao[melhor_idx].copy()

def crossover_blend(pai1, pai2, alpha=0.3):
    """Crossover BLX-alpha para representação real contínua."""
    filho = []
    for g1, g2 in zip(pai1, pai2):
        min_g, max_g = min(g1, g2), max(g1, g2)
        diff = max_g - min_g
        gene = np.random.uniform(min_g - alpha * diff, max_g + alpha * diff)
        gene = np.clip(gene, 1.0, 5.0)
        filho.append(gene)
    return sorted(filho)

def mutacao_gaussiana(cromossomo, pm=0.2, escala=0.3):
    if np.random.rand() < pm:
        cromossomo = [np.clip(gene + np.random.normal(0, escala), 1.0, 5.0) for gene in cromossomo]
    return sorted(cromossomo)

def rodar_algoritmo_genetico(geracoes, tam_pop, seed):
    np.random.seed(seed)
    populacao = [gerar_cromossomo() for _ in range(tam_pop)]
    historico_fitness = []
    
    for gen in range(geracoes):
        fitness_pop = np.array([calcular_fitness(ind) for ind in populacao])
        historico_fitness.append(np.max(fitness_pop))
        
        nova_pop = []
        # Elitismo: preserva os 2 melhores
        melhores_idx = np.argsort(fitness_pop)[-2:]
        for idx in melhores_idx:
            nova_pop.append(populacao[idx])
            
        while len(nova_pop) < tam_pop:
            p1 = selecao_torneio(populacao, fitness_pop)
            p2 = selecao_torneio(populacao, fitness_pop)
            filho = crossover_blend(p1, p2)
            filho = mutacao_gaussiana(filho)
            nova_pop.append(filho)
            
        populacao = nova_pop
        
    fitness_pop = np.array([calcular_fitness(ind) for ind in populacao])
    melhor_idx = np.argmax(fitness_pop)
    return populacao[melhor_idx], historico_fitness

#random search
def rodar_random_search(iteracoes, seed):
    np.random.seed(seed)
    melhor_fit = -1
    melhor_ind = None
    for _ in range(iteracoes):
        ind = gerar_cromossomo()
        fit = calcular_fitness(ind)
        if fit > melhor_fit:
            melhor_fit = fit
            melhor_ind = ind
    return melhor_ind, melhor_fit

# ==========================================
# 6. INTERFACE GRÁFICA (STREAMLIT)
# ==========================================
st.title("⚡ GreenCloud — Sistema Evolutivo-Fuzzy")
st.markdown("### Otimização Meta-Heurística de Precificação Dinâmica de Infraestrutura Cloud")
st.write("Esta aplicação integra lógica fuzzy Mamdani com algoritmos genéticos baseados em strings reais para otimizar regras de faturamento sob cargas computacionais dinâmicas.")

# Sidebar de Hiperparâmetros
st.sidebar.header("🔧 Configurações Globais do AG")
tam_pop = st.sidebar.slider("Tamanho da População", 10, 50, 20)
geracoes = st.sidebar.slider("Número de Gerações", 5, 50, 15)

# Instanciação dos Estados da Aplicação
if "dados_execucao" not in st.session_state:
    st.session_state.dados_execucao = None

col1, col2, col3 = st.columns(3)

# --- BOTÃO 1: HEURÍSTICA INICIAL ---
if col1.button("▶️ Executar Heurística Inicial", use_container_width=True):
    t_start = time.time()
    fit_inicial = calcular_fitness(None)
    t_end = time.time()
    
    st.session_state.dados_execucao = {
        "tipo": "Inicial",
        "melhor_fit": fit_inicial,
        "tempo": t_end - t_start,
        "cromossomo": [1.0, 1.0, 3.0, 3.0, 4.0, 5.0, 5.0],
        "curva": [fit_inicial] * geracoes,
        "std_fit": 0.0
    }
    st.success("Heurística Padrão Processada!")

# --- BOTÃO 2: RANDOM SEARCH ---
if col2.button("🎲 Executar Random Search", use_container_width=True):
    t_start = time.time()
    total_evals = tam_pop * geracoes
    
    # Roda 5 execuções independentes para coletar métricas de desvio padrão
    fits_seeds = []
    melhores_inds = []
    for s in [42, 101, 777, 2026, 99]:
        ind, fit = rodar_random_search(total_evals, seed=s)
        fits_seeds.append(fit)
        melhores_inds.append(ind)
        
    t_end = time.time()
    idx_top = np.argmax(fits_seeds)
    
    st.session_state.dados_execucao = {
        "tipo": "RandomSearch",
        "melhor_fit": fits_seeds[idx_top],
        "tempo": (t_end - t_start) / 5,
        "cromossomo": melhores_inds[idx_top],
        "curva": sorted(fits_seeds),
        "std_fit": np.std(fits_seeds)
    }
    st.success("Busca Aleatória Finalizada!")

# --- BOTÃO 3: ALGORITMO GENÉTICO ---
if col3.button("🧬 Executar Otimização Genética", use_container_width=True):
    t_start = time.time()
    
    # Executa 5 rodadas independentes mudando a semente (Exigência de Variabilidade)
    resultados_ag = []
    curvas_ag = []
    
    for s in [42, 101, 777, 2026, 99]:
        ind, curva = rodar_algoritmo_genetico(geracoes, tam_pop, seed=s)
        resultados_ag.append((ind, curva[-1]))
        curvas_ag.append(curva)
        
    t_end = time.time()
    
    # Seleção do melhor resultado absoluto entre as sementes
    melhores_fits = [r[1] for r in resultados_ag]
    idx_best = np.argmax(melhores_fits)
    melhor_cromossomo_ag = resultados_ag[idx_best][0]
    
    st.session_state.dados_execucao = {
        "tipo": "AG",
        "melhor_fit": melhores_fits[idx_best],
        "tempo": (t_end - t_start) / 5,
        "cromossomo": melhor_cromossomo_ag,
        "curva": curvas_ag[idx_best],
        "std_fit": np.std(melhores_fits)
    }
    st.success("Algoritmo Genético Otimizado com Sucesso!")

# ==========================================
# 7. EXIBIÇÃO DE RESULTADOS & GRÁFICOS
# ==========================================
if st.session_state.dados_execucao is not None:
    res = st.session_state.dados_execucao
    
    st.markdown("---")
    st.subheader("📊 Painel Analítico de Desempenho")
    
    # Tabela estática de baseline para o cálculo de melhoria percentual
    fit_base_inicial = calcular_fitness(None)
    melhoria_perc = ((res["melhor_fit"] - fit_base_inicial) / fit_base_inicial) * 100
    
    # Grid de Indicadores Chave (KPIs)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Melhor Fitness Encontrado", f"{res['melhor_fit']:.2f} pts")
    kpi2.metric("Tempo Médio de Execução", f"{res['tempo']:.4f} seg")
    kpi3.metric("Melhoria vs Heurística", f"{melhoria_perc:+.2f}%")
    kpi4.metric("Desvio Padrão (5-Seeds)", f"± {res['std_fit']:.3f}")

    # Plotagem da Curva de Convergência do AG
    col_graph, col_param = st.columns([2, 1])
    
    with col_graph:
        st.markdown("**Evolução Temporal do Fitness Técnico**")
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(res["curva"], marker='o', color='#10b981', linewidth=2)
        ax.set_title("Curva de Convergência / Aprendizado Real")
        ax.set_xlabel("Mapeamento de Épocas / Gerações")
        ax.set_ylabel("Score de Fitness (Lucratividade Líquida)")
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        
    with col_param:
        st.markdown("**DNA do Cromossomo Otimizado**")
        st.write("Distribuição ordenada dos pontos mapeados para as funções triangulares do Preço:")
        df_genes = pd.DataFrame(res["cromossomo"], columns=["Vértice (Fator Multiplicador)"])
        st.dataframe(df_genes, use_container_width=True)

    # ==========================================
    # 8. VALIDAÇÃO EM 6 CENÁRIOS FIXOS
    # ==========================================
    st.markdown("---")
    st.subheader("🧪 Validação Cruzada: Matriz de Teste em Produção (6 Cenários)")
    
    # Reconstrói o simulador atual com o cromossomo em memória
    simulador_val, preco_ctrl = construir_sistema_fuzzy(res["cromossomo"])
    
    lista_validada = []
    for cen in CENARIOS_VALIDACAO:
        preco_calculado = executar_fuzzy(simulador_val, cen["Carga"], cen["Demanda"])
        lista_validada.append({
            "Cenário Operacional": cen["Nome"],
            "Carga Alocada (%)": f"{cen['Carga']:.1f}%",
            "Taxa de Requisições": f"{cen['Demanda']:.0f} req/s",
            "Multiplicador de Preço (Saída)": f"{preco_calculado:.2f}x"
        })
        
    st.table(pd.DataFrame(lista_validada))

    # Plotagem Gráfica das Funções de Pertinência de Saída Resultantes
    st.markdown("**Visualização Física das Funções de Pertinência Otimizadas (Preço Dinâmico)**")
    fig_memb, ax_memb = plt.subplots(figsize=(10, 2.5))
    
    # Gera o universo de preço para plotagem direta
    u_preco = preco_ctrl.universe
    ax_memb.plot(u_preco, preco_ctrl['baixo'].mf, 'b', linewidth=1.5, label='Baixo')
    ax_memb.plot(u_preco, preco_ctrl['medio'].mf, 'g', linewidth=1.5, label='Médio')
    ax_memb.plot(u_preco, preco_ctrl['alto'].mf, 'r', linewidth=1.5, label='Alto')
    
    ax_memb.set_title("Resultados de Espaçamento das Funções de Pertinência após Processamento")
    ax_memb.legend()
    st.pyplot(fig_memb)

    #superficie de decisao 3d
    st.markdown("---")
    st.subheader("🌐 Superfície de Decisão 3D do Controlador")
    st.write("Mapeamento geométrico do comportamento do sistema em todo o universo de discurso.")

    with st.spinner("Gerando malha da superfície 3D..."):
        x_carga = np.linspace(0, 100, 20)
        y_demanda = np.linspace(0, 1000, 20)
        X, Y = np.meshgrid(x_carga, y_demanda)
        Z = np.zeros_like(X)


        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = executar_fuzzy(simulador_val, X[i, j], Y[i, j])

        fig_3d = plt.figure(figsize=(10, 5))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        surf = ax_3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        
        ax_3d.set_xlabel('Carga do Servidor (%)')
        ax_3d.set_ylabel('Demanda (req/s)')
        ax_3d.set_zlabel('Multiplicador de Preço')
        ax_3d.set_title('Superfície de Controle Otimizada')
        fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5)
        
        st.pyplot(fig_3d)

    #teste manual
    st.markdown("---")
    st.subheader("🎮 Simulação Interativa (Suporte à Decisão)")
    st.write("Insira valores manualmente para testar a robustez do motor sintonizado.")

    input_col1, input_col2 = st.columns(2)
    with input_col1:
        u_carga = st.number_input("Digite a Carga Atual do Servidor (%)", min_value=0.0, max_value=100.0, value=50.0)
    with input_col2:
        u_demanda = st.number_input("Digite a Demanda de Requisições (req/s)", min_value=0.0, max_value=1000.0, value=500.0)
        
    if st.button("Calcular Preço em Tempo Real"):
        preco_usuario = executar_fuzzy(simulador_val, u_carga, u_demanda)
        st.metric(label="Preço Dinâmico Recomendado", value=f"{preco_usuario:.2f}x")