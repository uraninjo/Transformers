from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3
import speech_recognition as sr


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
    while True:
        print("Me: ", end="")
        text = input("")
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

            step = int(stepq)
        else:
            continue
else:
    print("Error has occured!")
