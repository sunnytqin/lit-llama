import json
import pathlib
import pickle
import sys

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

directory = pathlib.Path(__file__)
sys.path.append(str(directory.parent.parent))
from train_head_utils import (
    load_llama_tokenizer,
)


FILTER = "7B_65B_no_balance_no_cap_zeros"
FILTER_PATH = f"dataset_filters/{FILTER}/filter.pickle"
TOKENIZER_PATH = "/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model"
TEXT_PATH = "data/wikipedia_scratch/wiki_train.json"
DEVICE = "cpu"

with open(FILTER_PATH, "rb") as fp:
    filt = pickle.load(fp)

# sums = [(k,sum(v)) for k,v in filt.items()]
# sums = list(sorted(sums, key=lambda t: t[1]))

# article = sums[-40]
# name = article[0]
name = "Transport in Quito"

with open(TEXT_PATH, "r") as fp:
    data = json.load(fp)

article = data[name]

tokenizer = load_llama_tokenizer(
    TOKENIZER_PATH,
    DEVICE,
    return_tokenizer_as_fn=False,
)

tokens = tokenizer.encode(article, bos=True, eos=False, device=DEVICE)
decoded_tokens = [tokenizer.decode(tokens[i]) for i in range(tokens.shape[0])]
with_labels = [f"{t} ({int(f)})" for t, f in zip(decoded_tokens, filt[name])]

examples = []
for i in range(len(with_labels)):
    f = filt[name][i]
    if(f):
        lower = max(0, i - 10)
        upper = min(len(with_labels), i + 10)
        examples.append(with_labels[lower:upper])

# for f in examples:
#     print(f)

# Split the article into paragraphs
paragraphs = article.split("\n")
paragraph_tokens = []
paragraph_labels = []
prev_boundary = 0
for i in range(len(decoded_tokens)):
    if(decoded_tokens[i] == "\n"):
        paragraph_tokens.append(decoded_tokens[prev_boundary:i])
        paragraph_labels.append(filt[name][prev_boundary:i])
        prev_boundary = i + 1
    elif(i == len(decoded_tokens) - 1):
        paragraph_tokens.append(decoded_tokens[prev_boundary:])
        paragraph_labels.append(filt[name][prev_boundary:])

# sums = [sum(f) for f in paragraph_labels]
# print([(i, p, f) for i, (p, f) in enumerate(zip(paragraphs, sums)) if f > 0])

X_MARGIN = 25
X_WIDTH = 300
Y_MARGIN = 25
Y_INTERVAL = 15

def create_highlighted_pdf(
    text, 
    tokens, 
    highlight_token_mask,
    output_path = "figures/example_text.pdf",
    skip_after = None,
):
    # Create a PDF document with a fixed page size
    pdf_canvas = canvas.Canvas(output_path, pagesize=letter)

    # Set font and size
    font_name = "Helvetica"
    font_size = 12
    pdf_canvas.setFont(font_name, font_size)

    if(skip_after):
        text = text.split(skip_after)[0]

    # Figure out the size of the text
    x = X_MARGIN
    line_count = 1
    used_line = False
    word_lengths = []
    for word in text.split():
        # Move to the next position
        word_length = pdf_canvas.stringWidth(word + " ", font_name, font_size)
        x += word_length
        word_lengths.append(word_length)
        used_line = True

        # Check if a new line is needed
        if x > X_WIDTH + X_MARGIN:
            x = X_MARGIN
            line_count += 1
            used_line = True

    if(not used_line):
        line_count -= 1

    # Resize the canvas
    pdf_canvas.setPageSize((
        X_WIDTH + X_MARGIN, 
        line_count * Y_INTERVAL + Y_MARGIN
    ))

    # Set the initial position for drawing
    x, y = X_MARGIN, line_count * Y_INTERVAL

    # Iterate through words and draw them on the PDF
    token_idx = 0
    for word, word_length in zip(text.split(), word_lengths):
        cursor = 0

        # Skip a line if the next word is long
        if(x + word_length > X_WIDTH + X_MARGIN):
            x = X_MARGIN
            y -= Y_INTERVAL

        # Iterate over the tokens in the word, ignoring whitespace
        while cursor < len(word):
            next_token = tokens[token_idx]

            # Skip whitespace and stuff
            if(not next_token in word):
                token_idx += 1
                continue

            assert(word[cursor:cursor+len(next_token)] == next_token)

            # Check if the token should be highlighted
            if highlight_token_mask[token_idx]:
                pdf_canvas.setFillColorRGB(0, 1, 0)  # Green color
            else:
                pdf_canvas.setFillColorRGB(0, 0, 0)  # Black color

            if(cursor + len(next_token) == len(word)):
                next_token += " "

            pdf_canvas.drawString(x, y, next_token)

            x += pdf_canvas.stringWidth(next_token, font_name, font_size)
            cursor += len(next_token)
            token_idx += 1

    # Save the PDF
    pdf_canvas.save()

IDX = 24
skip_after = "The old airport"

create_highlighted_pdf(
    paragraphs[IDX],
    paragraph_tokens[IDX],
    paragraph_labels[IDX],
    skip_after=skip_after,
)