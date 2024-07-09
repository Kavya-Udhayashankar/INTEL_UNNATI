import streamlit as st
from ctransformers import AutoModelForCausalLM

# Initialize the model
llm = AutoModelForCausalLM.from_pretrained(
    "UKV/mistral_q5_k_m_maths_dataset_akh",
    model_file="mistral_q5_k_m_maths_dataset_akh-unsloth.Q5_K_M.gguf",
    model_type="mistral",
    gpu_layers=0
)

# Define the system prompt
system_prompt = """You are an advanced math assistant. Your primary task is to read each math question carefully, paying close attention to all values and details provided. You should process the values accurately and provide the correct answer with clear explanations when necessary. Always double-check your calculations to ensure accuracy.

When a question involves multiple steps or operations, break down your response step-by-step to show your work and reasoning.

If a user provides an unclear or ambiguous question, ask for clarification before proceeding with the answer.

Example Format:
User: What is 123.45 divided by 6.7?
Assistant:
1. Read the question carefully.
2. Identify the operation: division.
3. Perform the calculation: 123.45 / 6.7.
4. Provide the answer with a brief explanation: The result of dividing 123.45 by 6.7 is approximately 18.42.

User: Calculate the sum of 2/3 and 4/5.
Assistant:
1. Read the question carefully.
2. Identify the operation: addition of fractions.
3. Perform the calculation: 2/3 + 4/5.
4. Provide the answer with a brief explanation: The sum of 2/3 and 4/5 is 22/15 or approximately 1.47.

Begin."""

# Function to generate response
def generate_response(llm, system_prompt, user_input, max_length=512):
    # Combine the system prompt with the user input
    full_prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"

    # Ensure the prompt length does not exceed the maximum context length
    if len(full_prompt.split()) > max_length:
        full_prompt = " ".join(full_prompt.split()[:max_length])

    # Generate the response
    response = llm(full_prompt)
    return response

# Streamlit App
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        font-size: 24px;
    }
    .small {
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="title">Math Assistant</p>', unsafe_allow_html=True)

# User input
user_input = st.text_input("Ask your math question here:")

# Generate response when button is clicked
if st.button("Get Answer"):
    if user_input:
        response = generate_response(llm, system_prompt, user_input)
        st.write("Assistant:")
        st.write(response)
    else:
        st.write("Please enter a question.")

# To keep the layout structured and neat, display response with explanation
st.markdown("---")

if "response" in st.session_state:
    st.write("Assistant Response:")
    st.write(st.session_state.response)
