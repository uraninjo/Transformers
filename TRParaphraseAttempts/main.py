import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from googletrans import Translator
import pandas as pd
from deep_translator import GoogleTranslator
from termcolor import colored


def translate(sentence, target="en", source="auto"):
    return GoogleTranslator(source=source, target=target).translate(sentence)


def paraphrase(input_text, num_return_sequences, num_beams):
    model_name = "tuner007/pegasus_paraphrase"
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    print(torch_device)
    batch = tokenizer(
        [input_text],
        truncation=True,
        padding="longest",
        max_length=60,
        return_tensors="pt",
    ).to(torch_device)
    translated = model.generate(
        **batch,
        max_length=60,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        temperature=1.5
    )
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


default = """Türk Kurtuluş Savaşı sürecinde Ankara Hükûmeti'ni kurdu, Türk Orduları Başkomutanı olarak Sakarya Meydan Muharebesi'ndeki başarısından dolayı 19 Eylül 1921 tarihinde "Gazi" unvanını aldı ve mareşalliğe yükseldi; askerî ve siyasi eylemleriyle İtilaf Devletleri ve destekçilerine karşı zafer kazandı.Savaşın ardından Cumhuriyet Halk Partisi'ni Halk Fırkası adıyla kurdu ve ilk genel başkanı oldu. 29 Ekim 1923'te cumhuriyetin ilanının akabinde cumhurbaşkanı seçildi. 1938'deki ölümüne dek dört dönem bu görevi yürüterek Türkiye'de en uzun süre cumhurbaşkanlığı yapmış kişi oldu."""
inpuT = str(input("Paraphrase edilecek metni girin: "))
contextss = default if input == "" else inpuT
contexts = contextss.split(".")
output = ""
for context in contexts:
    try:
        print(colored("input:", "green"), end=" ")
        print(context)
        output += translate(
            paraphrase(translate(context), 10, 10)[0], target="tr", source="en"
        )
    except Exception as e:
        print(colored("HATA:", "red"), end=" ")
        print(e)


print(
    """
"""
)
print(colored("ILK HALI:", "blue"), end=" ")
print(contextss)
print(
    """
"""
)
print(colored("SON HALI:", "blue"), end=" ")
print(output)
