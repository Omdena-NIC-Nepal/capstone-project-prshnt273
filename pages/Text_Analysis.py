import streamlit as st
from utils.nlp_utils import analyze_text
import spacy
from spacy import displacy

def show():
    st.title(" Climate Text Analysis")
    st.markdown("""
    Analyze climate-related text using NLP techniques with spaCy.
    Extract entities, relationships, and key terms from climate reports.
    """)
    
    # Text input
    text = st.text_area("Enter climate-related text to analyze:", """
    Climate change is causing rising global temperatures and more extreme weather events. 
    The IPCC reports that human activities are the dominant cause of observed warming since the mid-20th century.
    Mitigation strategies include reducing greenhouse gas emissions and transitioning to renewable energy.
    """)
    
    if st.button("Analyze Text"):
        if text.strip():
            results = analyze_text(text)
            
            st.subheader("Named Entities")
            st.markdown(displacy.render(results['doc'], style="ent", jupyter=False), unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Most Frequent Words")
                st.table(results['word_freq'])
                
            with col2:
                st.subheader("Nouns and Verbs")
                st.write("**Nouns:**", ", ".join(results['nouns'][:10]))
                st.write("**Verbs:**", ", ".join(results['verbs'][:10]))
            
            st.subheader("Dependency Parse")
            st.markdown(displacy.render(results['doc'], style="dep", jupyter=False, options={'distance': 100}), unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to analyze.")