from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your desired model

# Original sentence and model-generated paraphrases
original_sentence = "The robot fails to grasp the cup securely."
# paraphrases = [
#     "The robot is unable to grasp the cup securely.",
#     "The robot fails to hold the mug firmly.",
#     "The robot could not grip the cup properly.",
#     "The robot is unsuccessful in securely holding the mug.",
#     "The robot fails to achieve a secure grasp on the cup.",
#     "The robot struggles to grasp the mug firmly.",
#     "The robot was unable to securely grip the cup.",
#     "The robot does not manage to hold the mug tightly.",
#     "The robot failed to properly secure its grip on the cup.",
#     "The robot is unable to maintain a secure grasp on the mug.",
#     "The robot does not succeed in firmly gripping the cup.",
#     "The robot has trouble holding the mug securely.",
#     "The robot is unsuccessful in gripping the cup tightly.",
#     "The robot fails to get a proper hold of the mug.",
#     "The robot could not grasp the cup with a secure grip."
# ]
paraphrases = [
    "The robot attempts to grab the cup from the inside.",
    "The robot is trying to pick up the mug by its inner surface.",
    "The robot aims to grasp the cup from within.",
    "The robot is working on grabbing the mug by its interior.",
    "The robot tries to hold the cup by its inner section.",
    "The robot attempts to seize the mug using its rim.",
    "The robot is making an effort to grasp the inner part of the cup.",
    "The robot focuses on holding the mug by the rim and inside.",
    "The robot is trying to take hold of the cup from the inside.",
    "The robot endeavors to grip the mug from the inner area.",
    "The robot is aiming to clutch the cup using its interior.",
    "The robot tries to pick the mug up by holding the rim.",
    "The robot makes an effort to grab hold of the cup using its inner surface.",
    "The robot attempts to grasp the mug by its rim and inner part.",
    "The robot works on seizing the cup through its interior."
]

# Encode the original sentence and paraphrases
original_embedding = model.encode(original_sentence, convert_to_tensor=True)
paraphrase_embeddings = model.encode(paraphrases, convert_to_tensor=True)

# Compute cosine similarity between the original sentence and each paraphrase
cosine_similarities = util.pytorch_cos_sim(original_embedding, paraphrase_embeddings)

# Print results
threshold = 0.75  # Define a threshold for acceptable similarity
print("Paraphrase Evaluation:\n")
for paraphrase, similarity in zip(paraphrases, cosine_similarities[0]):
    is_reasonable = "✅" if similarity >= threshold else "❌"
    print(f"Paraphrase: {paraphrase}\nSimilarity: {similarity:.4f} {is_reasonable}\n")

