import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class AlgoritmoGenetico:
    def __init__(self, D, S, H, C, Sseg, tam_populacao=100, num_geracoes=100):
        self.D = D  
        self.S = S  
        self.H = H  
        self.C = C  
        self.Sseg = Sseg  
        self.tam_populacao = tam_populacao
        self.num_geracoes = num_geracoes
        self.melhor_historico = []
        self.media_historico = []
        
    #Faz o calculo do custo total e realiza a penalização para soluções inviáveis
    def custo_total(self, Q):
        if Q < self.Sseg or Q > self.C: 
            return float('inf')
        return (self.D / Q) * self.S + (Q / 2) * self.H
    
    def criar_populacao_inicial(self):
        return np.random.uniform(self.Sseg, self.C, self.tam_populacao)
    
    #faz uma avaliação para a população gerada
    def avaliar_populacao(self, populacao):
        return np.array([self.custo_total(Q) for Q in populacao])
    
    #faz a seleção dos pais
    def selecionar_pais(self, populacao, fitness, num_pais):
        pais = []
        for _ in range(num_pais):
            indices_torneio = np.random.randint(0, len(populacao), 3)
            fitness_torneio = fitness[indices_torneio]
            pais.append(populacao[indices_torneio[np.argmin(fitness_torneio)]])
        return np.array(pais)
    
    #realiza o cruzamento dos pais
    def cruzamento(self, pai1, pai2):
        alpha = np.random.random()
        filho = alpha * pai1 + (1 - alpha) * pai2
        return filho
    
    #aplica a mutação
    def mutacao(self, individuo, taxa_mutacao=0.1):
        if np.random.random() < taxa_mutacao:
            delta = np.random.uniform(-10, 10)  # Alteração aleatória
            novo_valor = individuo + delta
            # Garante que o valor está dentro dos limites
            return np.clip(novo_valor, self.Sseg, self.C)
        return individuo
    
    #Executa o algoritmo genético
    def executar(self):
        # População inicial
        populacao = self.criar_populacao_inicial()
        
        for geracao in range(self.num_geracoes):
            fitness = self.avaliar_populacao(populacao)
            
            # Armazena o melhor e a média para histórico
            self.melhor_historico.append(np.min(fitness))
            self.media_historico.append(np.mean(fitness))
            
            # Seleção
            pais = self.selecionar_pais(populacao, fitness, self.tam_populacao)
            
            # Nova população
            nova_populacao = []
            
            # Elitismo - mantém o melhor indivíduo
            melhor_idx = np.argmin(fitness)
            nova_populacao.append(populacao[melhor_idx])
            
            # Gera nova população
            while len(nova_populacao) < self.tam_populacao:
                # Seleciona pais aleatoriamente
                idx1, idx2 = np.random.randint(0, len(pais), 2)
                # Cruzamento
                filho = self.cruzamento(pais[idx1], pais[idx2])
                # Mutação
                filho = self.mutacao(filho)
                nova_populacao.append(filho)
            
            populacao = np.array(nova_populacao)
        
        # Encontra a melhor solução
        fitness_final = self.avaliar_populacao(populacao)
        melhor_idx = np.argmin(fitness_final)
        melhor_q = populacao[melhor_idx]
        melhor_custo = fitness_final[melhor_idx]
        
        return melhor_q, melhor_custo, self.melhor_historico, self.media_historico


#função que cria as áreas para a colocação dos dados a apresentação dos resultados e gráficos 
class OtimizacaoEstoqueGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Otimização de Estoque - Algoritmo Genético")
        self.root.geometry("1200x800")
        
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(expand=True, fill='both')
        
        self.left_frame = ttk.LabelFrame(self.main_frame, text="Parâmetros", padding="10")
        self.left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        self.criar_widgets()
        self.configurar_graficos()
        
    def criar_widgets(self):
        # Parâmetros do problema
        self.params = {
            'D': ('Demanda anual (D):', 1000),
            'S': ('Custo de fazer um pedido (S):', 50),
            'H': ('Custo de manutenção (H):', 2),
            'C': ('Capacidade máxima (C):', 200),
            'Sseg': ('Estoque segurança (Sseg):', 10),
            'pop': ('Tamanho da população:', 100),
            'ger': ('Número de gerações:', 100)
        }
        
        self.entries = {}
        for i, (key, (label, default)) in enumerate(self.params.items()):
            ttk.Label(self.left_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            entry = ttk.Entry(self.left_frame, width=20)
            entry.insert(0, str(default))
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.entries[key] = entry
        
        ttk.Button(self.left_frame, text="Otimizar", command=self.otimizar).grid(row=len(self.params), column=0, columnspan=2, pady=10)
        
        # Área de resultado
        self.resultado_text = tk.Text(self.left_frame, height=5, width=40)
        self.resultado_text.grid(row=len(self.params)+1, column=0, columnspan=2, pady=5)
        
    def configurar_graficos(self):
        # Frame para os gráficos
        self.fig = Figure(figsize=(12, 5))
        
        # Gráfico de convergência
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_title('Convergência do AG')
        self.ax1.set_xlabel('Geração')
        self.ax1.set_ylabel('Custo')
        
        # Gráfico de custo total
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_title('Curva de Custo Total')
        self.ax2.set_xlabel('Quantidade de Pedido (Q)')
        self.ax2.set_ylabel('Custo Total')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.right_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def atualizar_graficos(self, melhor_q, melhor_historico, media_historico, D, S, H, C, Sseg):
        # Limpa os gráficos
        self.ax1.clear()
        self.ax2.clear()
        
        # Gráfico de convergência
        self.ax1.plot(melhor_historico, label='Melhor fitness')
        self.ax1.plot(media_historico, label='Média fitness')
        self.ax1.set_title('Convergência do AG')
        self.ax1.set_xlabel('Geração')
        self.ax1.set_ylabel('Custo')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Gráfico de custo total
        Q_range = np.linspace(max(1, Sseg), C, 1000)
        custos = [(D/q)*S + (q/2)*H for q in Q_range]
        self.ax2.plot(Q_range, custos, label='Custo Total')
        self.ax2.scatter([melhor_q], [(D/melhor_q)*S + (melhor_q/2)*H], 
                        color='red', marker='o', s=100, label='Solução AG')
        self.ax2.set_title('Curva de Custo Total')
        self.ax2.set_xlabel('Quantidade de Pedido (Q)')
        self.ax2.set_ylabel('Custo Total')
        self.ax2.legend()
        self.ax2.grid(True)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def otimizar(self):
        try:
            # Coleta valores dos campos
            D = float(self.entries['D'].get())
            S = float(self.entries['S'].get())
            H = float(self.entries['H'].get())
            C = float(self.entries['C'].get())
            Sseg = float(self.entries['Sseg'].get())
            tam_pop = int(self.entries['pop'].get())
            num_ger = int(self.entries['ger'].get())
            
            # Cria e executa o algoritmo genético
            ag = AlgoritmoGenetico(D, S, H, C, Sseg, tam_pop, num_ger)
            melhor_q, melhor_custo, melhor_historico, media_historico = ag.executar()
            
            # Atualiza resultado
            resultado = (f"Melhor solução encontrada:\n"
                       f"Q* = {melhor_q:.2f}\n"
                       f"Custo total = R$ {melhor_custo:.2f}")
            
            self.resultado_text.delete(1.0, tk.END)
            self.resultado_text.insert(tk.END, resultado)
            
            # Atualiza gráficos
            self.atualizar_graficos(melhor_q, melhor_historico, media_historico,
                                  D, S, H, C, Sseg)
            
        except ValueError as e:
            messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OtimizacaoEstoqueGUI(root)
    root.mainloop()