import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import config
import boto3
import time
import uuid

s3 = boto3.resource(
    service_name='s3',
    region_name=config.AWS_DEFAULT_REGION,
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
)

dynamodb = boto3.resource('dynamodb',
                          aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                          region_name=config.AWS_DEFAULT_REGION)


os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY


def upload_file_to_s3(s3, file):
    """
    Upload a file to an S3 bucket

    :param file: File to upload
    :param bucket_name: Bucket to upload to
    :param object_name: S3 object name. If not specified, file_name is used
    :return: True if file was uploaded, else False
    """

    object_name = file.name

    try:
        s3.Bucket("chatbot-filestorage").put_object(Key=object_name, Body=file)
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True


def save_chat_to_dynamodb(dynamodb, user_id,  chat_history):
    table = dynamodb.Table('chatbot-chat-history')

    for message in chat_history:
        timestamp = int(time.time() * 1000)
        # Generate a unique key using a UUID
        history_key = user_id

        try:
            table.put_item(
                Item={
                    'history': history_key,  # Unique identifier for each item
                    'timestamp': timestamp,
                    'user_id': user_id,  # Replace with actual user ID if available
                    'role': message['role'],
                    'content': message['content']
                }
            )
        except Exception as e:
            print(f"Error saving to DynamoDB: {e}")


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    st.write('Made with ❤️ by Team-10')
    st.markdown("***")
    st.write("***Team-10*** ")
    st.write("Nischay Sai Cherukuri")
    st.write("Pranav Sai Putta")
    st.write("Rajeev Koneru")


def main():
    st.header("Chat with PDF 💬")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(
            uuid.uuid4()) + "_" + str(int(time.time()))

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:

        # Replace with your actual bucket name
        uploaded_to_s3 = upload_file_to_s3(s3, pdf)
        if uploaded_to_s3:
            st.success("File uploaded to S3")
        else:
            st.error("Failed to upload file to S3")

        pdfReader = PdfReader(pdf)
        raw_text = ''
        for page in pdfReader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Load or create FAISS index
        embeddings = OpenAIEmbeddings()
        # st.write(texts)
        docsearch = FAISS.from_texts(texts, embeddings)

    # Chat input
    query = st.chat_input("Ask questions about your PDF file:")
    if query:
        # Process the query
        docs = docsearch.similarity_search(query=query, k=3)
        llm = OpenAI(model="text-davinci-003")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append(
            {"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            with st.chat_message("User", avatar="👩‍💻"):
                st.write(content)
        else:
            with st.chat_message("Assistant"):
                st.write(content)

    if st.button('End Session'):
        # Save the current chat history
        save_chat_to_dynamodb(
            dynamodb, st.session_state.user_id, st.session_state.messages)

        # Clear the session state
        st.session_state.messages = []

        # Reload the page to start a new session
        st.rerun()


if __name__ == '__main__':
    main()
