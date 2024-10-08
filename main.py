
import streamlit as st
import helper_functions.llm as llm

# Set up and run this Streamlit
#testing
# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("Vance Streamlit App")

form = st.form(key="form")
form.subheader("What do you want to ask")

user_prompt = form.text_area("Enter your prompt here", height=200)

if form.form_submit_button("Submit"):
    st.toast(f"User Input Submitted - {user_prompt}")
    response = llm.get_completion(user_prompt)
    st.write(response) # <--- This displays the response generated by the LLM onto the frontend 🆕
    print(f"User Input is {user_prompt}")