import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#função que cria as áreas para a colocação dos dados a apresentação dos resultados e gráficos 
class OtimizacaoEstoqueGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora de Quantidade de Pedido Ótima")
        self.root.geometry("900x800")
        
        style = ttk.Style()
        style.configure('TLabel', padding=5)
        style.configure('TButton', padding=5)
        style.configure('TEntry', padding=5)
        
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side='left', fill='both', expand=True)
        
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side='right', fill='both', expand=True)
        
        self.campos = {
            'D': ('Demanda anual (D):', 1000),
            'S': ('Custo de fazer um pedido (S):', 50),
            'H': ('Custo de manutenção por unidade (H):', 2),
            'C': ('Capacidade de armazenamento (C):', 200),
            'Sseg': ('Estoque de segurança (Sseg):', 10)
        }
        
        self.entries = {}
        for i, (key, (label, default)) in enumerate(self.campos.items()):
            ttk.Label(self.left_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            entry = ttk.Entry(self.left_frame, width=30)
            entry.insert(0, str(default))
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.entries[key] = entry
        
        button_frame = ttk.Frame(self.left_frame)
        button_frame.grid(row=len(self.campos), column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Calcular", command=self.calcular_q_otimo).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Limpar", command=self.limpar_campos).pack(side='left', padx=5)
        
        self.resultado_text = tk.Text(self.left_frame, height=5, width=40)
        self.resultado_text.grid(row=len(self.campos)+1, column=0, columnspan=2, pady=10)
        self.resultado_text.config(state='disabled')
        
        self.setup_grafico()

    #realiza a configuração dos gráficos
    def setup_grafico(self):
        """Configura a área do gráfico"""
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax.set_xlabel('Quantidade de Pedido (Q)')
        self.ax.set_ylabel('Custo Total')
        self.ax.set_title('Relação entre Quantidade de Pedido e Custo Total')
        self.ax.grid(True)
        self.canvas.draw()

    def atualizar_grafico(self, Q_otimo, D, S, H, C, Sseg):
        """Atualiza o gráfico com os novos valores"""
        self.ax.clear()
        
        # Gera pontos para o gráfico
        Q_range = np.linspace(max(1, Sseg), C, 1000)
        custos = [self.custo_total(q, D, S, H) for q in Q_range]
        
        # Plota a curva de custo
        self.ax.plot(Q_range, custos, 'b-', label='Custo Total')
        
        # Marca o ponto ótimo
        custo_otimo = self.custo_total(Q_otimo, D, S, H)
        self.ax.plot(Q_otimo, custo_otimo, 'ro', label='Ponto Ótimo')
        
        # Adiciona linhas tracejadas para o ponto ótimo
        self.ax.axvline(x=Q_otimo, color='r', linestyle='--', alpha=0.3)
        self.ax.axhline(y=custo_otimo, color='r', linestyle='--', alpha=0.3)
        
        # Configurações do gráfico
        self.ax.set_xlabel('Quantidade de Pedido (Q)')
        self.ax.set_ylabel('Custo Total')
        self.ax.set_title('Relação entre Quantidade de Pedido e Custo Total')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend()
        
        # Adiciona anotação para o ponto ótimo
        self.ax.annotate(f'Q* = {Q_otimo:.2f}\nCusto = {custo_otimo:.2f}',
                        xy=(Q_otimo, custo_otimo),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->'))
        
        # Atualiza o canvas
        self.canvas.draw()

    #Valida os valores para serem maiores que zero
    def validar_entrada(self, valor):
        try:
            num = float(valor)
            return num > 0
        except ValueError:
            return False

    #Faz o calculo do custo total
    def custo_total(self, Q, D, S, H):
        return (D / Q) * S + (Q / 2) * H

    #verifica o valor otimo para ser utilizado
    def calcular_q_otimo(self):
        self.resultado_text.config(state='normal')
        self.resultado_text.delete(1.0, tk.END)
        
        try:
            valores = {}
            for key in self.campos:
                valor = self.entries[key].get()
                if not self.validar_entrada(valor):
                    raise ValueError(f"Valor inválido para {key}")
                valores[key] = float(valor)
            
            #restrições
            restricoes = [
                {'type': 'ineq', 'fun': lambda Q: valores['C'] - Q[0]},    # Q <= C
                {'type': 'ineq', 'fun': lambda Q: Q[0] - valores['Sseg']}  # Q >= Sseg
            ]
            
            # Realiza a otimização com a função minimize
            resultado = minimize(
                lambda Q: self.custo_total(Q[0], valores['D'], valores['S'], valores['H']),
                x0=[valores['Sseg']],
                constraints=restricoes,
                bounds=[(0, None)]
            )
            
            if resultado.success:
                q_otimo = resultado.x[0]
                custo_minimo = resultado.fun
                resultado_texto = (
                    f"Quantidade ótima de pedido (Q): {q_otimo:.2f}\n"
                    f"Custo total mínimo: R$ {custo_minimo:.2f}\n\n"
                    f"Status da otimização: Sucesso!"
                )
                
                # Atualiza o gráfico
                self.atualizar_grafico(
                    q_otimo,
                    valores['D'],
                    valores['S'],
                    valores['H'],
                    valores['C'],
                    valores['Sseg']
                )
            else:
                resultado_texto = "A otimização não convergiu.\nVerifique os valores inseridos."
            
            self.resultado_text.insert(tk.END, resultado_texto)
            
        except ValueError as e:
            messagebox.showerror("Erro", str(e))
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro inesperado: {str(e)}")
        finally:
            self.resultado_text.config(state='disabled')

    #utilizado para limpar os campos de entrada
    def limpar_campos(self):
        for key, (_, default) in self.campos.items():
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, str(default))
        self.resultado_text.config(state='normal')
        self.resultado_text.delete(1.0, tk.END)
        self.resultado_text.config(state='disabled')
        
        # Limpa o gráfico
        self.ax.clear()
        self.ax.set_xlabel('Quantidade de Pedido (Q)')
        self.ax.set_ylabel('Custo Total')
        self.ax.set_title('Relação entre Quantidade de Pedido e Custo Total')
        self.ax.grid(True)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = OtimizacaoEstoqueGUI(root)
    root.mainloop()