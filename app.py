import streamlit as st
from agents import run_crew

# Streamlit UI setup
st.set_page_config(page_title="SEO Article Generator", page_icon="📝", layout="wide")

st.title("📝 SEO Article Generator with CrewAI and Groq")
st.markdown("""
Welcome to the **SEO Article Generator**! Enter a topic below, and watch as our AI agents collaborate to produce a high-quality, SEO-optimized article.
""")

# Input for the topic
st.sidebar.header("User Input")
topic = st.sidebar.text_input("Enter the topic you want to generate an article about:")

# Button to trigger the Crew execution
if st.sidebar.button("Generate Article"):
    if topic:
        st.markdown("## Process Overview")
        with st.spinner("Generating SEO-optimized content..."):
            try:
                # Run the crew and get the result
                result = run_crew(topic)

                # Extract the process logs
                process_logs = result.get("process_logs", "")
                final_output = result.get("final_output", "No final output available")

                # Display the final generated article content
                st.write("---")
                st.markdown("## 📄 Final Generated Article")
                st.success("Your SEO-optimized article is ready!")
                
                # Display the actual article content here
                st.write(final_output)

                st.balloons()  # Celebration animation

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a topic to generate an article.")
