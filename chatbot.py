from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
import streamlit as st

# Your Hugging Face API token
huggingfacehub_api_token = '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'

# Model and tokenizer
model = "HuggingFaceH4/starchat-beta"
llm = HuggingFaceHub(repo_id=model,
                     huggingfacehub_api_token=huggingfacehub_api_token,  # Pass the token here
                     model_kwargs={"min_length": 30, "max_new_tokens": 256,
                                   "do_sample": True, "temperature": 0.2,
                                   "top_k": 50, "top_p": 0.95,
                                   "eos_token_id": 49155})

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="You are a helpful assistant. {user_input}"
)

# LLMChain setup
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

# Streamlit UI
st.title("AI Assistant")
user_input = st.text_input("You:", "")
if st.button("Send"):
    llm_reply = llm_chain.run({"user_input": user_input})
    reply = llm_reply.partition('<|end|>')[0]  # Parse response
    st.text_area("Chatbot:", reply, height=100)
