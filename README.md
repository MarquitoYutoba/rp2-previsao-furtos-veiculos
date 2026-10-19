# rp2-previsao-furtos-veiculos

# 🐍 Configuração do Ambiente Python

Este guia explica como configurar o ambiente de desenvolvimento do projeto Python, criar e ativar um ambiente virtual, instalar as dependências e desativar o ambiente quando necessário.

## 1. Pré-requisitos

Antes de começar, verifique se o Python e o Git estão instalados na sua máquina.

- **Python:** https://www.python.org/downloads/
- **Git:** https://git-scm.com/downloads/

Para verificar a instalação do Python, execute no terminal:

**Windows:**
```bash
python --version
```

**Linux:**
```bash
python3 --version
```

## 2. Clone o repositório

Caso ainda não tenha o projeto na sua máquina, clone o repositório:

```bash
git clone URL_DO_REPOSITORIO
```

Entre na pasta do projeto:

```bash
cd NOME_DO_PROJETO
```

Substitua `URL_DO_REPOSITORIO` e `NOME_DO_PROJETO` pelas informações do seu repositório.

## 3. Crie o ambiente virtual

O ambiente virtual (`venv`) permite instalar as dependências do projeto de forma isolada, sem interferir nas outras instalações do Python da máquina.

### Windows

```bash
python -m venv .venv
```

### Linux

```bash
python3 -m venv .venv
```

Esse comando cria uma pasta chamada `.venv` na raiz do projeto.

## 4. Ative o ambiente virtual

Antes de instalar as dependências, ative o ambiente virtual correspondente ao seu sistema operacional.

### Windows — Prompt de Comando (CMD)

```cmd
.venv\Scripts\activate.bat
```

### Windows — PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

> **Observação:** caso o PowerShell bloqueie a execução do script, pode ser necessário ajustar a política de execução para o usuário atual. Consulte a documentação oficial do PowerShell antes de alterar essa configuração.

### Linux

```bash
source .venv/bin/activate
```

Após a ativação, normalmente aparecerá `(.venv)` no início da linha do terminal, indicando que o ambiente virtual está ativo.

## 5. Instale as dependências

Com o ambiente virtual ativado, instale todas as dependências listadas no arquivo `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Esse comando instala as bibliotecas necessárias para executar o projeto.

**Importante:** execute o comando na raiz do projeto, onde está localizado o arquivo `requirements.txt`.

Se o arquivo não existir, verifique com a equipe responsável pelo projeto como obter a lista correta de dependências antes de continuar.

## 6. Execute o projeto

Após instalar as dependências, execute o projeto conforme as instruções específicas da aplicação.

Por exemplo, se o arquivo principal for `main.py`:

```bash
python main.py
```

Substitua `main.py` pelo arquivo de entrada correto do projeto, se necessário.

## 7. Desative o ambiente virtual

Quando terminar de trabalhar no projeto, você pode desativar o ambiente virtual com o comando:

```bash
deactivate
```

Esse comando funciona tanto no Windows quanto no Linux.

A desativação encerra o uso do ambiente virtual no terminal atual. Os arquivos e as dependências instaladas permanecem salvos na pasta `.venv`.

## 8. Boas práticas

- Não envie a pasta `.venv` para o GitHub.
- Mantenha o arquivo `requirements.txt` atualizado.
- Ative o ambiente virtual sempre que for trabalhar no projeto.
- Instale novas dependências com o ambiente virtual ativado.
- Não compartilhe senhas, tokens ou credenciais no repositório.

Para adicionar uma nova dependência ao projeto:

```bash
python -m pip install nome-da-biblioteca
```

Para atualizar o arquivo `requirements.txt` com as dependências instaladas:

```bash
python -m pip freeze > requirements.txt
```

---

**Pronto!** Seguindo essas etapas, você terá o ambiente Python configurado e as dependências instaladas para trabalhar no projeto.