import transformers
import torch
import streamlit as st
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def capitalize_sentences(text):
    # Split the text into sentences
    sentences = re.split('(?<=[.!?]) +', text)

    # Capitalize the first letter of each sentence and join them back together
    corrected_text = ' '.join(sentence.capitalize() for sentence in sentences)

    return corrected_text


def main():
    model_id = "Bhotuya/TextSummarizerAI_Basic_v1"  # custom finetuned model made 4 summarizing

    pipeline = transformers.pipeline("summarization", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
                                     device_map="auto"
                                     )

    st.title('AI summarizer ')

    # Initialize session state for user input if it doesn't exist
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ''

    # Create a layout with columns
    col1, col2 = st.columns([5, 1])  # adjust the numbers to change the relative sizes of the columns

    # Place the text input field in the first column and the button in the second column
    user_input = col1.text_area('Enter text (anything above 1024 tokens will be truncated)',
                                value=st.session_state['user_input'], height=200)

    # Add a button for adding custom text
    if col2.button('Show example'):
        custom_text = ("The Israel-Palestine conflict is a long-standing political and territorial dispute between "
                       "Israelis and Palestinians, which began in the early 20th century. The heart of the conflict "
                       "is a dispute over land and borders. The conflict began with the establishment of Israel in "
                       "1948, which led to the displacement of many Palestinians. Since then, there have been several "
                       "wars and uprisings, known as intifadas. The issues at the core of the conflict include the "
                       "status of Jerusalem, the borders of Israel, the right of return for Palestinian refugees, "
                       "and the establishment of a Palestinian state alongside Israel. Peace efforts, including the "
                       "Oslo Accords and the Camp David Summit, have attempted to resolve these issues, but have not "
                       "resulted in a final peace agreement. The conflict has resulted in a tragic loss of life and "
                       "has had a major impact on the lives of all involved. It remains one of the world’s most "
                       "difficult and enduring conflicts, with both sides suffering from periodic bouts of violence "
                       "and ongoing political instability.")
        st.session_state['user_input'] = custom_text  # Update session state
        st.experimental_rerun()  # Rerun the script to update the text input field

    if st.button('Summarize '):
        if user_input:
            with st.spinner('Summarizing...'):

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1024,
                    chunk_overlap=30,
                    length_function=len,
                    is_separator_regex=False,
                )
                # Split the input into chunks of 1024 tokens
                chunks = text_splitter.create_documents([user_input])

                ans = ' '.join([pipeline("summarize: " + str(chunk))[0]['summary_text'] for chunk in chunks])

                ans = capitalize_sentences(ans)

                st.write('\n\n', ans)

        else:
            st.write("Nothing's there")


if __name__ == '__main__':
    main()
