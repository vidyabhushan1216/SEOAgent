import streamlit as st
from agents import run_crew

# Streamlit UI setup
st.title("SEO Article Generator with CrewAI and Groq")
st.markdown("### Enter a topic to generate an SEO-optimized article.")

# Input for the topic
topic = st.text_input("Topic:", "")

# Button to trigger the Crew execution
if st.button("Generate Article"):
    if topic:
        with st.spinner("Generating SEO-optimized content..."):
            try:
                # Run the Crew with the provided topic
                result = run_crew(topic)
                st.markdown("### Generated Article")
                st.markdown(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a topic to generate an article.")
