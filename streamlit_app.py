import os
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

def search_and_calculate_tfidf(search_query):
    hasil_directory = '/Users/farrelmanazilin/Document/kuliah/data/Text Mining/Hasil'
    folders_to_check = ['Kesehatan', 'Teknologi', 'Politik', 'Olahraga', 'Kriminal']

    corpus = []

    for folder_name in folders_to_check:
        folder_path = os.path.join(hasil_directory, folder_name)
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    word_freq_dict = {}
                    for line in lines[1:]:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            word = parts[0]
                            freq = int(parts[1])
                            word_freq_dict[word] = freq
                    corpus.append(' '.join(word_freq_dict.keys()))
 
    tfidf_scores = calculate_tfidf(corpus, search_query)

    return tfidf_scores
 
def calculate_tfidf(corpus, search_query):
 
    vocabulary = set()
    for doc in corpus:
        vocabulary.update(doc.split()) 
    df = {}
    for term in vocabulary:
        df[term] = sum(1 for doc in corpus if term in doc)
 
    query_terms = search_query.split()
    tfidf_scores = {}
    for term in query_terms:
        if term in vocabulary:
            tf = query_terms.count(term) / len(query_terms)
            idf = np.log(len(corpus) / (df[term] + 1))
            tfidf_scores[term] = tf * idf

    return tfidf_scores
d_tfidf = {}
        total_tfidf = 0.0
        for word in query_words:
            word_idx = tfidf_vectorizer.vocabulary_.get(word)
            if word_idx is not None:
                word_tfidf[word] = tfidf_matrix[doc_index, word_idx]
                total_tfidf += tfidf_matrix[doc_index, word_idx]

        result = {
            "Nama File": file_name,
            "Word TF-IDF": word_tfidf,
            "Total TF-IDF": total_tfidf
        }
        results.append(result)

    return results

# Aplikasi Streamlit
st.title('Aplikasi Pencarian TF-IDF')

search_query = st.text_input("Masukkan kata kunci pencarian:")
if search_query:
    results = search_and_display_results(search_query)
    st.header('Hasil Pencarian:')
    for i, result in enumerate(results):
        st.subheader(f'Urutan {i + 1}')
        st.write(f'Nama File: {result["Nama File"]}')
        st.write('Word TF-IDF:')
        for word, tfidf in result['Word TF-IDF'].items():
            st.write(f'{word}: {tfidf}')
        st.write(f'Total TF-IDF: {result["Total TF-IDF"]}')
