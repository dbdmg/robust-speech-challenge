from datasets import load_dataset, load_metric, Audio
import os
import codecs
import pandas as pd

print("Loading dataset Common Voice..")
#common_voice = load_dataset("common_voice", "it", split="train")
common_voice = load_dataset("mozilla-foundation/common_voice_7_0", "it", use_auth_token=True, split="train")
# concat all the sentences (used to train kenLM)
print("Sentence writing..")
with open("text.txt", "w") as file:
  file.write(" ".join(common_voice["sentence"]))



print("Loading dataset LibriSpeech..")
libriSpeech = load_dataset("multilingual_librispeech", "italian", split="train")
print("Sentence writing..")
with open("text.txt", "a+") as file:
  file.write(" ".join(libriSpeech["text"]))


print("Processing TED it-en..")
text = open("text.txt", "a+")
files = os.listdir("datasets/TED/it-en/data/train/vtt")
for file in files:
    with codecs.open("datasets/TED/it-en/data/train/vtt/" + file, 'r', encoding='utf-8', errors='ignore') as fdata:
        for i,l in enumerate(fdata):
            if i > 3 and l[0] != "\n" and l[0] != "0" and l[0] != "1" and l[0] != "2" and l[0] != "3" and not l.startswith("Traduzione") and not l.startswith("Traduttore") and not l.startswith("Revisione") and not l.startswith("Revisore"):
                text.write(" " + l[:-1])

print("Processing TED it-es..")
files = os.listdir("datasets/TED/it-es/data/train/vtt")
for file in files:
    with codecs.open("datasets/TED/it-es/data/train/vtt/" + file, 'r', encoding='utf-8', errors='ignore') as fdata:
        for i,l in enumerate(fdata):
            if i > 6 and l[0] != "\n" and l[0] != "0" and l[0] != "1" and l[0] != "2" and l[0] != "3" and not l.startswith("Traduzione") and not l.startswith("Traduttore") and not l.startswith("Revisione") and not l.startswith("Revisore"):
                text.write(" " + l[:-1])


print("Processing VoxForge..")
folders = os.listdir("datasets/VoxForge")
for folder in folders:
    if folder != "._.DS_Store" and folder != ".DS_Store":
        with open("datasets/VoxForge/" + folder + "/etc/prompts-original") as f:
            lines = f.readlines()
            for l in lines:
                text.write(" " + l[8:-1])


print("Processing M-AILABS Speech..")
folders = os.listdir("datasets/M-AILABS_Speech_Dataset/it_IT/by_book/male")
for folder in folders:
    if folder != "._.DS_Store" and folder != ".DS_Store":
        books = os.listdir("datasets/M-AILABS_Speech_Dataset/it_IT/by_book/male/" + folder)
        for book in books:
            if book != "info.txt" and book != "._.DS_Store" and book != ".DS_Store" and book != "._info.txt":
                df = pd.read_csv("datasets/M-AILABS_Speech_Dataset/it_IT/by_book/male/" + folder + "/" + book + "/metadata.csv", sep="|", names = ["id", "original", "clean"])
                text.write(" ".join(list(df["clean"])))
folders = os.listdir("datasets/M-AILABS_Speech_Dataset/it_IT/by_book/female")
for folder in folders:
    if folder != "._.DS_Store" and folder != ".DS_Store":
        books = os.listdir("datasets/M-AILABS_Speech_Dataset/it_IT/by_book/female/" + folder)
        for book in books:
            if book != "info.txt" and book != "._.DS_Store" and book != ".DS_Store" and book != "._info.txt":
                df = pd.read_csv("datasets/M-AILABS_Speech_Dataset/it_IT/by_book/female/" + folder + "/" + book + "/metadata.csv", sep="|", names = ["id", "original", "clean"])
                text.write(" ".join(list(df["clean"])))
folders = os.listdir("datasets/M-AILABS_Speech_Dataset/it_IT/by_book/mix")
for folder in folders:
    if folder != "._.DS_Store" and folder != ".DS_Store" and folder != "info.txt" and folder != "._info.txt":
        df = pd.read_csv("datasets/M-AILABS_Speech_Dataset/it_IT/by_book/mix/" + folder + "/metadata.csv", sep="|", names = ["id", "original", "clean"])
        text.write(" ".join(list(df["clean"])))


print("Processing EuroParl-ST..")
sentences = []
folders = os.listdir("datasets/EuroParl-ST/v1.1/it")
for folder in folders:
    if folder != "audios" and folder != "speeches.cer" and folder != ".DS_Store" and folder != "._.DS_Store":
        with open("datasets/EuroParl-ST/v1.1/it/" + folder + "/train/speeches.it") as file:
            lines = file.readlines()
            for line in lines:
                if line[:-1] not in sentences:
                    sentences.append(line[:-1])
text.write(" ".join(sentences))


print("Processing MSPKA..")
sentences = []
folders = os.listdir("datasets/MSPKA")
for folder in folders:
    if folder != "__MACOSX":
        with open("datasets/MSPKA/" + folder + "/list_sentences") as f:
            lines = f.readlines()
            for line in lines:
                sentences.append(line[line.find(")")+1:-1])
text.write(" ".join(sentences))

print("Processing EMOVO..")
sentences = "Gli operai si alzano presto. I vigili sono muniti di pistola. La cascata fa molto rumore. L’autunno prossimo Tony partirà per la Spagna nella prima metà di ottobre. Ora prendo la felpa di là ed esco per fare una passeggiata. Un attimo dopo s’è incamminato ... ed è inciampato. Vorrei il numero telefonico del Signor Piatti. La casa forte vuole col pane. La forza trova il passo e l’aglio rosso. Il gatto sta scorrendo nella pera  Insalata pastasciutta coscia d’agnello limoncello. Uno quarantatré dieci mille cinquantasette venti. Sabato sera cosa farà? Porti con te quella cosa?"
text.write(" " + sentences)


text.close()

