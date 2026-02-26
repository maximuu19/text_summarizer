import transformers
import torch
import streamlit as st
import re
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from compression import compress, decompress


def capitalize_sentences(text):
    # Split the text into sentences
    sentences = re.split('(?<=[.!?]) +', text)

    # Capitalize the first letter of each sentence and join them back together
    corrected_text = ' '.join(sentence.capitalize() for sentence in sentences)

    return corrected_text


@st.cache_resource
def load_summarizer():
    model_id = "Bhotuya/TextSummarizerAI_Basic_v1"  # custom finetuned model made 4 summarizing
    pipeline = transformers.pipeline(
        "summarization",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline


def summarizer_tab():
    pipeline = load_summarizer()

    # Initialize session state for user input if it doesn't exist
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ''

    # Create a layout with columns
    col1, col2 = st.columns([5, 1])  # adjust the numbers to change the relative sizes of the columns

    # Place the text input field in the first column and the button in the second column
    user_input = col1.text_area('Enter long text (anything above 1024 tokens will be chunked)',
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
                       "has had a major impact on the lives of all involved. It remains one of the world's most "
                       "difficult and enduring conflicts, with both sides suffering from periodic bouts of violence "
                       "and ongoing political instability.")
        st.session_state['user_input'] = custom_text  # Update session state
        st.rerun()  # Rerun the script to update the text input field

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


# ---------------------------------------------------------------------------
# Compression tab – Adaptive Predictive Modeling
# ---------------------------------------------------------------------------

_SAMPLE_TEXT = (
    "The Israel-Palestine conflict is a long-standing political and territorial dispute between "
    "Israelis and Palestinians, which began in the early 20th century. The heart of the conflict "
    "is a dispute over land and borders. The conflict began with the establishment of Israel in "
    "1948, which led to the displacement of many Palestinians. Since then, there have been several "
    "wars and uprisings, known as intifadas. The issues at the core of the conflict include the "
    "status of Jerusalem, the borders of Israel, the right of return for Palestinian refugees, "
    "and the establishment of a Palestinian state alongside Israel. Peace efforts, including the "
    "Oslo Accords and the Camp David Summit, have attempted to resolve these issues, but have not "
    "resulted in a final peace agreement. The conflict has resulted in a tragic loss of life and "
    "has had a major impact on the lives of all involved. It remains one of the world's most "
    "difficult and enduring conflicts, with both sides suffering from periodic bouts of violence "
    "and ongoing political instability."
)


def _render_metrics(result: dict) -> None:
    """Display original size, compressed size, and compression ratio."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Original Size", f"{result['original_size']:,} bytes")
    c2.metric("Compressed Size", f"{result['compressed_size']:,} bytes")
    ratio = result['compression_ratio']
    delta_label = f"{ratio:+.1f}%"
    c3.metric("Compression Ratio", delta_label,
              delta=delta_label,
              delta_color="normal" if ratio >= 0 else "inverse")


def _render_dictionary(result: dict) -> None:
    """Visualise the dynamically generated context-specific dictionary."""
    st.subheader("Dynamic Dictionary – Word/Value Mapping")
    st.caption(
        "Words are ranked by adaptive frequency. The most frequent tokens receive "
        "the **shortest** variable-length integer codes (Zipf-inspired optimisation)."
    )

    sorted_words = result['sorted_words']
    freq = result['frequencies']
    code_sizes = result['code_sizes']
    total_tokens = sum(freq.values())

    top_n = min(30, len(sorted_words))
    rows = []
    for rank, word in enumerate(sorted_words[:top_n], start=1):
        count = freq[word]
        rows.append({
            "Rank": rank,
            "Token": word,
            "Frequency (count)": count,
            "Frequency (%)": f"{count / total_tokens * 100:.2f}%",
            "Code (index)": result['dictionary'][word],
            "Code Size (bytes)": code_sizes[word],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart of top-15 token frequencies
    st.subheader("Token Frequency Distribution (Top 15)")
    chart_df = df.head(15).set_index("Token")[["Frequency (count)"]]
    st.bar_chart(chart_df)


def _render_bigrams(result: dict) -> None:
    """Visualise the predictive bigram model."""
    bigrams = result['bigrams']
    if not bigrams:
        return

    st.subheader("Predictive Model – Top Bigrams (N-gram Analysis)")
    st.caption(
        "The predictive model tracks which tokens follow each other most often. "
        "This context-aware probability guides future encoding decisions."
    )

    rows = [{"Bigram": f'"{w1} \u2192 {w2}"', "Co-occurrence Count": cnt}
            for (w1, w2), cnt in sorted(bigrams.items(), key=lambda x: -x[1])[:15]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def compression_tab() -> None:
    """Render the Adaptive Predictive Compression tab."""
    st.markdown(
        "### Adaptive, Predictive Modeling \u2013 Text Compression\n"
        "This tool compresses text using a **dynamic dictionary** built by real-time "
        "statistical analysis. Tokens with the highest frequency receive the shortest "
        "variable-length codes, following principles similar to **Zipf\u2019s Law**."
    )

    # Session-state keys used by this tab
    for key in ('comp_input', 'comp_result', 'comp_decompressed'):
        if key not in st.session_state:
            st.session_state[key] = '' if key != 'comp_result' else None

    # --- Input area --------------------------------------------------------
    col_in, col_btn = st.columns([5, 1])
    user_text = col_in.text_area(
        "Input Text",
        value=st.session_state['comp_input'],
        height=200,
        placeholder="Paste or type text here\u2026",
        key="comp_input_area",
    )

    if col_btn.button("Sample Text", key="comp_sample"):
        st.session_state['comp_input'] = _SAMPLE_TEXT
        st.session_state['comp_result'] = None
        st.session_state['comp_decompressed'] = ''
        st.rerun()

    # --- Control panel -----------------------------------------------------
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    do_compress = btn_col1.button("\U0001f5dc Compress", use_container_width=True)
    do_decompress = btn_col2.button("\U0001f4c2 Decompress", use_container_width=True)
    do_reset = btn_col3.button("\U0001f504 Reset", use_container_width=True)

    active_text = user_text or st.session_state.get('comp_input', '')

    if do_reset:
        st.session_state['comp_input'] = ''
        st.session_state['comp_result'] = None
        st.session_state['comp_decompressed'] = ''
        st.rerun()

    if do_compress:
        if active_text.strip():
            with st.spinner("Analysing and compressing\u2026"):
                result = compress(active_text)
            if result:
                st.session_state['comp_result'] = result
                st.session_state['comp_input'] = active_text
                st.session_state['comp_decompressed'] = ''
            else:
                st.warning("Could not compress the provided text.")
        else:
            st.warning("Please enter some text before compressing.")

    if do_decompress:
        result = st.session_state.get('comp_result')
        if result and result.get('compressed_data'):
            with st.spinner("Decompressing\u2026"):
                recovered = decompress(result['compressed_data'])
            st.session_state['comp_decompressed'] = recovered or "(decompression failed)"
        else:
            st.warning("No compressed data found. Please compress some text first.")

    # --- Output display ----------------------------------------------------
    result = st.session_state.get('comp_result')
    if result:
        st.divider()
        _render_metrics(result)
        st.divider()
        _render_dictionary(result)
        st.divider()
        _render_bigrams(result)

    decompressed = st.session_state.get('comp_decompressed', '')
    if decompressed:
        st.divider()
        st.subheader("Decompressed Output")
        st.text_area("Recovered Text", value=decompressed, height=150, disabled=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    st.title('AI Text Tools')

    tab1, tab2 = st.tabs(["\u270d\ufe0f Text Summarizer", "\U0001f5dc\ufe0f Text Compression"])

    with tab1:
        summarizer_tab()

    with tab2:
        compression_tab()


if __name__ == '__main__':
    main()
