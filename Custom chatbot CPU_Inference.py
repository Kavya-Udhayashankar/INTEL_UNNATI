{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d2cfdcbf-ff32-4b1a-95b5-a096b55b1de1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer, TextStreamer\n",
    "from intel_extension_for_transformers.transformers import AutoModelForCausalLM\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8012416b-3849-4b91-911d-d84b2adc4bef",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1f12adc225104da3ad340c6d28d33734",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Fetching 1 files:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "da112d9a1346438c929efabf342af11f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Fetching 1 files:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "To solve this equation, we need to isolate the variable on one side of the equation and set up an algebraic equation that can be solved for x. Let's perform these steps.\n",
      "\n",
      "First, subtract 5 from both sides of the equation:\n",
      "2x = 13 - 5\n",
      "\n",
      "Now, divide both sides by 2:\n",
      "2x / 2 = (13 - 5) / 2\n",
      "x = 8 / 2\n",
      "\n",
      "Finally, simplify the fraction to find the value of x:\n",
      "x = 4\n",
      "\n",
      "So the solution is x = 4.\n"
     ]
    }
   ],
   "source": [
    "from ctransformers import AutoModelForCausalLM\n",
    "\n",
    "# Initialize the model\n",
    "llm = AutoModelForCausalLM.from_pretrained(\n",
    "    \"UKV/mistral_q5_k_m_maths_dataset_akh\",\n",
    "    model_file=\"mistral_q5_k_m_maths_dataset_akh-unsloth.Q5_K_M.gguf\",\n",
    "    model_type=\"mistral\",\n",
    "    gpu_layers=0\n",
    ")\n",
    "\n",
    "# Define the system prompt\n",
    "system_prompt = \"\"\"You are an advanced math assistant. Your primary task is to read each math question carefully, paying close attention to all values and details provided. You should process the values accurately and provide the correct answer with clear explanations when necessary. Always double-check your calculations to ensure accuracy.\n",
    "\n",
    "When a question involves multiple steps or operations, break down your response step-by-step to show your work and reasoning.\n",
    "\n",
    "If a user provides an unclear or ambiguous question, ask for clarification before proceeding with the answer.\n",
    "\n",
    "Example Format:\n",
    "User: What is 123.45 divided by 6.7?\n",
    "Assistant:\n",
    "1. Read the question carefully.\n",
    "2. Identify the operation: division.\n",
    "3. Perform the calculation: 123.45 / 6.7.\n",
    "4. Provide the answer with a brief explanation: The result of dividing 123.45 by 6.7 is approximately 18.42.\n",
    "\n",
    "User: Calculate the sum of 2/3 and 4/5.\n",
    "Assistant:\n",
    "1. Read the question carefully.\n",
    "2. Identify the operation: addition of fractions.\n",
    "3. Perform the calculation: 2/3 + 4/5.\n",
    "4. Provide the answer with a brief explanation: The sum of 2/3 and 4/5 is 22/15 or approximately 1.47.\n",
    "\n",
    "Begin.\"\"\"\n",
    "\n",
    "def generate_response(llm, system_prompt, user_input, max_length=512):\n",
    "    # Combine the system prompt with the user input\n",
    "    full_prompt = f\"{system_prompt}\\nUser: {user_input}\\nAssistant:\"\n",
    "\n",
    "    # Ensure the prompt length does not exceed the maximum context length\n",
    "    if len(full_prompt.split()) > max_length:\n",
    "        full_prompt = \" \".join(full_prompt.split()[:max_length])\n",
    "\n",
    "    # Generate the response\n",
    "    response = llm(full_prompt)\n",
    "    return response\n",
    "\n",
    "# Example usage\n",
    "user_input = \"Solve for x: 2x + 5 = 13\"\n",
    "response = generate_response(llm, system_prompt, user_input)\n",
    "print(response)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b3bb60a-d505-4cfb-baf1-3e062b81d8c5",
   "metadata": {},
   "outputs": [],
   "source": []
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
 "nbformat_minor": 5
}
