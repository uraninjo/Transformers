from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3
import speech_recognition as sr


def speak(text):
    engine = pyttsx3.init("sapi5")
    engine.setProperty("rate", 160)
    engine.setProperty("voice", engine.getProperty("voices")[0].id)
    engine.say(text)
    engine.runAndWait()


def get_audio_eng():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio, language="en-US")
            print(said)
        except Exception as e:
            print("Exception: " + str(e))
    return said.lower()


tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/DialoGPT-large"
)  # large-medium-small available
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large").to("cuda")


stepq = input(
    """Do you want to let the bot train with chat history?
Type '1' if you say yes, otherwise type '0': """
)
if stepq == "0" or stepq == "1":
    step = 0
    speak("Hello Sir.")
    while True:
        print("Me: ", end="")
        text = get_audio_eng()
        if text == "destroy yourself":
            break
        if text != "":

            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(
                text + tokenizer.eos_token, return_tensors="pt"
            ).to("cuda")

            # append the new user input tokens to the chat history
            bot_input_ids = (
                torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
                if step > 0
                else new_user_input_ids
            )

            # generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = model.generate(
                bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
            ).to("cuda")

            # pretty print last ouput tokens from bot
            print("Alita: ", end="")
            response = "{}".format(
                tokenizer.decode(
                    chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                    skip_special_tokens=True,
                )
            )
            print(response)
            speak(response)
            step = int(stepq)
        else:
            continue
else:
    speak("Error has occured!")
