# ChatBotwTransformers

I have tried microsoft/DialoGPT-large model for a chatbot.

I used cuda to speed it up, get faster results. If your gpu doesn't support cuda, you may basically delete .to('cuda') part from code...

The training part is a little bit problematic but it demonstrates how to use that kind of model for chatbot or voice assistant.

For the training part, I added a choice part which you could choose to train or not. Without the training part, It may work properly...
