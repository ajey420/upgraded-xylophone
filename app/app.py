import streamlit as st
import tempfile
from dataloading import *
from langchain_core.messages import HumanMessage
#from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Chroma, OpenAIEmbeddings
#from langchain_core import multi_modal_rag_chain

os.environ['OPENAI_API_KEY'] = 'sk-proj-DE1WYZ1gF6bqOph4foxlT3BlbkFJzDyQVa8DcKaI9PM1yDwL'


openai_api_key = 'sk-proj-DE1WYZ1gF6bqOph4foxlT3BlbkFJzDyQVa8DcKaI9PM1yDwL'

# Ensure initialization of key components in session state

if 'chat_history' not in st.session_state : st.session_state.chat_history = []

def main():
    st.set_page_config(page_title='Unstructured Data Interaction Chatbot', page_icon=':books:')
    st.header("Unstructured Data Interaction Chatbot :books:")
    
    # Ensure initialization of key components in session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'chain_multimodal_rag' not in st.session_state:
        st.session_state.chain_multimodal_rag = None
    if 'retriever_multi_vector_img' not in st.session_state:
         st.session_state.retriever_multi_vector_img = None

    with st.sidebar:
        st.subheader("Upload Your PDF")
        uploaded_file = st.file_uploader("Upload pdf")
        if st.button("Process") and uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            with st.spinner("Processing..."):
                try : 
                    raw_pdf_elements = process_pdf(tmp_file_path)
                    texts, tables = categorize_elements(raw_pdf_elements)
                    joined_texts = " ".join(texts)

                except : 
                    texts = 'This is some text'
                    tables = []
                    joined_texts = 'This is some text'
                    tables = []

                texts_4k_token = text_splitter.split_text(joined_texts)
                text_summaries, table_summaries = generate_text_summaries(texts_4k_token, tables, summarize_texts=True)
                img_base64_list, image_summaries = generate_img_summaries("./figures")

                st.session_state.vectorstore = Chroma(
                    collection_name="mm_rag_pdf",
                    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
                )

                st.session_state.retriever_multi_vector_img = create_multi_vector_retriever(
                    st.session_state.vectorstore,
                    text_summaries,
                    texts,
                    table_summaries,
                    tables,
                    image_summaries,
                    img_base64_list,
                )

                st.session_state.chain_multimodal_rag = multi_modal_rag_chain(st.session_state.retriever_multi_vector_img)
                # st.write(type(session_state.chain_multimodal_rag))
                st.success("Processing complete! Now ask your question.")

    question = st.chat_input("Your question")
    # st.session_state.chat_history = []
    if question and st.session_state.chain_multimodal_rag:
        response = st.session_state.chain_multimodal_rag.invoke({
            'input' : question , 
            'chat_history' : st.session_state.chat_history
        })  # Properly passing a string as the question
        if response:
            st.write("Response:", response)
            st.session_state.chat_history.extend([
                HumanMessage(
                    content = question
                ) , 
                response['answer']
            ])

        st.write(st.session_state.chat_history)

        

if __name__ == '__main__':
    main()
