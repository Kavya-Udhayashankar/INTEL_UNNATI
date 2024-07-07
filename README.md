# INTEL_UNNATI
PS4 - Introduction to GenAI and Simple LLM Inference on CPU and Finetuning of LLM Model to Create a Custom Chatbot.

This project explores the capabilities of GenAI and the process of implementing LLM inference on CPU, with a focus on developing a customized chatbot.

## Overview

The chosen LLM model for this project is Mistral, which has been specifically fine-tuned and quantized to optimize its performance for mathematical problem-solving tasks. The chatbot is designed to run efficiently on CPU, providing accurate and reliable assistance for math-related queries through an intuitive user interface created with Streamlit.

## Features
- Efficient CPU-Based Inference
- Custom Fine-Tuned LLM for Math Problems
- Model Quantization for Enhanced Performance
- User-Friendly Streamlit Interface

## Details
- DATASET: [AnonY0324/orca-math-word-problems-200k](https://huggingface.co/datasets/AnonY0324/orca-math-word-problems-200k)
- MODEL: [UKV/mistral_q5_k_m_maths_dataset](https://huggingface.co/UKV/mistral_q5_k_m_maths_dataset_akh)


## Implementation Details

The above project was created in  [Intel® Tiber™ Developer Cloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/services.html)

clone the repository 
```bash
  git clone https://github.com/Kavya-Udhayashankar/INTEL_UNNATI.git

```
Setting up the environment

```bash
  git clone https://github.com/intel/intel-extension-for-transformers.git
  
```
Create Conda environment

```bash
  conda create -n itrex-1 python=3.10 -y
  conda activate itrex-1
  
```
Install required packages 

```bash
  pip install intel-extension-for-transformers
  pip install -r requirements_cpu.txt
  pip install -r requirements.txt
  
```
Create a huggingface access token and login

```bash
  huggingface-cli login
```
Run the below files 

```bash
  python build_chatbot_on_spr.py
  python single_node_finetuning_on_spr.py
```
## Sample queries in build_chatbot_on_spr
Example,
- "Tell me about Intel Xeon Scalable Processors."
- "What is AMD Ryzen?"
- "What is the difference between ARM and x86 architectures?"
- "Can you explain what a GPU is?"
- "What is the role of a motherboard in a computer?"

## Timeline in single_node_finetuning_on_spr
![Screenshot from 2024-07-07 21-31-41](https://github.com/Kavya-Udhayashankar/INTEL_UNNATI/assets/115878369/dc842c20-42b8-4352-b987-1e01b41f966f)


## Math Assistant Chatbot
```bash
  Run Custom_chatbot_CPU_Inference.py
```
## Streamlit User Interface

Install Streamlit and Localtunnel

```bash
pip install -q streamlit
npm install localtunnel
```

Run the Streamlit app

```bash
streamlit run app.py &>/content/logs.txt &
npx localtunnel --port 8501
```
