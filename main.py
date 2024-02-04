import os
from tkinter import *
from tkinter import filedialog
from openpyxl import Workbook
from numba import cuda
import numpy as np

def listar_arquivos_diretorio(diretorio):
    arquivos = []
    for pasta_atual, sub_pastas, arquivos_na_pasta in os.walk(diretorio):
        for nome_arquivo in arquivos_na_pasta:
            caminho_arquivo = os.path.join(pasta_atual, nome_arquivo)
            if os.path.isfile(caminho_arquivo):
                tamanho_arquivo = os.path.getsize(caminho_arquivo)
                arquivos.append((nome_arquivo, caminho_arquivo, tamanho_arquivo))
    return arquivos

@cuda.jit
def vetor_addition_gpu(a, b, result):
    i = cuda.grid(1)
    if i < len(result):
        result[i] = a[i] + b[i]

def criar_planilha():
    diretorio = diretorio_entry.get()
    arquivos = listar_arquivos_diretorio(diretorio)
    planilha = Workbook()
    planilha_ativa = planilha.active
    planilha_ativa.append(["Nome do Arquivo", "Caminho", "Tamanho (bytes)"])
    for arquivo in arquivos:
        planilha_ativa.append(arquivo)

    # Example GPU computation
    a = np.ones(10000000)
    b = np.ones(10000000)
    result = np.zeros_like(a)

    # Offload computation to GPU
    threadsperblock = 256
    blockspergrid = (len(result) + (threadsperblock - 1)) // threadsperblock
    vetor_addition_gpu[blockspergrid, threadsperblock](a, b, result)

    nome_arquivo_excel = filedialog.asksaveasfilename(defaultextension=".xlsx")
    if nome_arquivo_excel:
        planilha.save(nome_arquivo_excel)
        status_label.config(text="Planilha criada com sucesso!")




# Configurações da janela principal
root = Tk()
root.title("Criador de Planilha de Arquivos")
root.geometry("400x200")

# Campo de entrada para o diretório
diretorio_label = Label(root, text="Diretório:")
diretorio_label.pack()
diretorio_entry = Entry(root, width=50)
diretorio_entry.pack()

# Botão para selecionar o diretório
def selecionar_diretorio():
    diretorio = filedialog.askdirectory()
    diretorio_entry.delete(0, END)
    diretorio_entry.insert(0, diretorio)
selecionar_diretorio_button = Button(root, text="Selecionar Diretório", command=selecionar_diretorio)
selecionar_diretorio_button.pack()

# Botão para criar a planilha
criar_planilha_button = Button(root, text="Criar Planilha", command=criar_planilha)
criar_planilha_button.pack()

# Label para exibir status
status_label = Label(root, text="")
status_label.pack()

root.mainloop()
