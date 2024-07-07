{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "NeuralChat is a customizable chat framework designed to create user own chatbot within few minutes on multiple architectures. This notebook is used to demonstrate how to build a talking chatbot on 4th Generation of Intel® Xeon® Scalable Processors Sapphire Rapids.\n",
    "\n",
    "The 4th Generation of Intel® Xeon® Scalable processor provides two instruction sets viz. AMX_BF16 and AMX_INT8 which provides acceleration for bfloat16 and int8 operations respectively."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Prepare Environment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Install intel extension for transformers:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install intel-extension-for-transformers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Install Requirements:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!git clone https://github.com/intel/intel-extension-for-transformers.git"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/\n",
    "!pip install -r requirements_cpu.txt\n",
    "%cd ../../../"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Build your chatbot 💻"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Text Chat"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Giving NeuralChat the textual instruction, it will respond with the textual response."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/ucf2ac712c468911283ee2151bd7861b/.conda/envs/itrex-1/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations\n",
      "  warnings.warn(\n",
      "/home/ucf2ac712c468911283ee2151bd7861b/.conda/envs/itrex-1/lib/python3.10/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading model Intel/neural-chat-7b-v3-1\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5f14fec44b804336b5300af814ba0f1a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Intel Xeon Scalable Processors represent a family of high-performance central processing units (CPUs) designed for data centers, cloud computing, and other demanding workloads. These processors offer significant improvements in performance, efficiency, and scalability compared to their predecessors. They feature advanced technologies such as Intel Advanced Vector Extensions 512 (AVX-512), Intel Turbo Boost Technology 2.0, and Intel Hyper-Threading Technology, which contribute to increased throughput and reduced latency. Additionally, they support various memory configurations, including DDR4, DDR3L, and Optane DC Persistent Memory, allowing for flexible system designs tailored to specific needs. Overall, the Intel Xeon Scalable Processors aim to deliver exceptional performance and reliability for mission-critical applications and large-scale deployments. интелект интелл процессор скалируемый эксон интелл скалируемый процессор эксон интелл скалируемый процессор эксен интелл скалируемый процессор эксен интелл скали\n"
     ]
    }
   ],
   "source": [
    "# BF16 Optimization\n",
    "from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig\n",
    "from intel_extension_for_transformers.transformers import MixedPrecisionConfig\n",
    "config = PipelineConfig(optimization_config=MixedPrecisionConfig())\n",
    "chatbot = build_chatbot(config)\n",
    "response = chatbot.predict(query=\"Tell me about Intel Xeon Scalable Processors.\")\n",
    "print(response)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AMD Ryzen is a series of high-performance central processing units (CPUs) developed by Advanced Micro Devices (AMD). It was first introduced in 2017 as a competitor to Intel's Core processors. The Ryzen lineup offers various models with different core counts, clock speeds, and features, catering to diverse needs such as gaming, content creation, and general computing tasks. These CPUs are known for their impressive performance, power efficiency, and affordability, making them popular among PC enthusiasts and gamers alike. інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде\n"
     ]
    }
   ],
   "source": [
    "response1 = chatbot.predict(query=\"What is AMD Ryzen?\")\n",
    "print(response1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The main difference between ARM and x86 architectures lies in their design and usage. ARM (Advanced RISC Machine) architecture is a reduced instruction set computing (RISC) design, which focuses on efficiency and low power consumption. It's commonly used in mobile devices, embedded systems, and Internet of Things (IoT) applications. ARM processors are generally smaller and consume less power compared to x86 processors.\n",
      "\n",
      "On the other hand, x86 (short for Intel 8086) architecture is a complex instruction set computing (CISC) design, originally developed by Intel. This architecture is known for its flexibility and compatibility with various operating systems. It has been widely adopted in personal computers, servers, and workstations. X86 processors offer better performance and support for multitasking compared to ARM processors. However, they tend to consume more power and generate more heat.\n",
      "\n",
      "In summary, while both architectures have their unique strengths and weaknesses, ARM is more suitable for energy-efficient, portable devices, whereas x86 caters to high-performance desktop and server applications.\n"
     ]
    }
   ],
   "source": [
    "response2 = chatbot.predict(query=\"What is the difference between ARM and x86 architectures?\")\n",
    "print(response2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A GPU (Graphics Processing Unit) is a specialized electronic circuit designed for accelerating the rendering of graphics and images on various devices like computers, smartphones, and gaming consoles. It works in parallel with the CPU (Central Processing Unit), which handles general computing tasks. GPUs are optimized for handling complex mathematical calculations required for processing visual data, making them particularly efficient at rendering high-quality 2D and 3D graphics, video playback, and other graphical applications. In summary, a GPU is like a superhero for visuals, helping our devices display stunning images and animations. інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде інде\n"
     ]
    }
   ],
   "source": [
    "response3 = chatbot.predict(query=\"Can you explain what a GPU is?\")\n",
    "print(response3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The role of a motherboard in a computer can be compared to the central nervous system of a human body. It serves as the main hub where all essential components connect and communicate with each other. A motherboard houses various crucial elements such as CPU (Central Processing Unit), RAM (Random Access Memory), storage devices like hard drives or SSDs, and expansion slots for additional hardware like graphics cards or network adapters. It also provides power supply and distributes it among different parts of the computer. In summary, the motherboard acts as the backbone of a computer, enabling seamless communication between its vital organs and ensuring optimal performance. интелектуальный бот может быть полезным помощником в жизни человека, предоставляя разнообразные услуги и возможности для обучения и развития. интеллект может быть применен не только к технологическим областям, но и к другим сферам, таким как искусство, наука или образование.\n"
     ]
    }
   ],
   "source": [
    "response4 = chatbot.predict(query=\"What is the role of a motherboard in a computer?\")\n",
    "print(response4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Text Chat With Retrieval Plugin"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "User could also leverage NeuralChat Retrieval plugin to do domain specific chat by feding with some documents like below:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval/\n",
    "!pip install -r requirements.txt\n",
    "%cd ../../../../../../"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!mkdir docs\n",
    "%cd docs\n",
    "!curl -OL https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/docs/sample.jsonl\n",
    "!curl -OL https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/docs/sample.txt\n",
    "!curl -OL https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/docs/sample.xlsx\n",
    "%cd .."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from intel_extension_for_transformers.neural_chat import PipelineConfig\n",
    "from intel_extension_for_transformers.neural_chat import build_chatbot\n",
    "from intel_extension_for_transformers.neural_chat import plugins\n",
    "plugins.retrieval.enable=True\n",
    "plugins.retrieval.args[\"input_path\"]=\"./docs/\"\n",
    "config = PipelineConfig(plugins=plugins)\n",
    "chatbot = build_chatbot(config)\n",
    "response = chatbot.predict(\"How many cores does the Intel® Xeon® Platinum 8480+ Processor have in total?\")\n",
    "print(response)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Voice Chat with ASR & TTS Plugin"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the context of voice chat, users have the option to engage in various modes: utilizing input audio and receiving output audio, employing input audio and receiving textual output, or providing input in textual form and receiving audio output.\n",
    "\n",
    "For the Python API code, users have the option to enable different voice chat modes by setting ASR and TTS plugins enable or disable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/pipeline/plugins/audio/\n",
    "!pip install -r requirements.txt\n",
    "%cd ../../../../../../"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!curl -OL https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/speaker_embeddings/spk_embed_default.pt\n",
    "!curl -OL https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from intel_extension_for_transformers.neural_chat import PipelineConfig\n",
    "from intel_extension_for_transformers.neural_chat import build_chatbot\n",
    "from intel_extension_for_transformers.neural_chat import plugins\n",
    "plugins.tts.enable = True\n",
    "plugins.tts.args[\"output_audio_path\"] = \"./response.wav\"\n",
    "plugins.asr.enable = True\n",
    "\n",
    "config = PipelineConfig(plugins=plugins)\n",
    "chatbot = build_chatbot(config)\n",
    "result = chatbot.predict(query=\"./sample.wav\")\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Low Precision Optimization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## BF16"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# BF16 Optimization\n",
    "from intel_extension_for_transformers.neural_chat.config import PipelineConfig\n",
    "from intel_extension_for_transformers.transformers import MixedPrecisionConfig\n",
    "config = PipelineConfig(optimization_config=MixedPrecisionConfig())\n",
    "chatbot = build_chatbot(config)\n",
    "response = chatbot.predict(query=\"Tell me about Intel Xeon Scalable Processors.\")\n",
    "print(response)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "neural-chat",
   "language": "python",
   "name": "neural-chat"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
